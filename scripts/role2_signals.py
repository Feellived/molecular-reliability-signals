"""컨포멀, 적용가능도메인, 모델 불일치와 최종 담당 2 신호를 생성한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def infer_task(labels: pd.Series) -> str:
    values = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
    return "classification" if len(values) <= 2 and values <= {0, 1} else "regression"


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def deterministic_uniform(row_uid: object) -> float:
    digest = hashlib.sha256(f"mist-aps-v1|{row_uid}".encode()).digest()[:8]
    return (int.from_bytes(digest, "big") + 0.5) / 2**64


def randomized_aps_score(p1: np.ndarray, labels: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    p_true = np.where(labels == 1, p1, 1 - p1)
    p_other = 1 - p_true
    before = np.where(p_true >= p_other, 0.0, p_other)
    return before + (1 - uniform) * p_true


def build_conformal(source: pd.DataFrame, regular: pd.DataFrame, augmented: pd.DataFrame, alpha: float):
    task = infer_task(source["Y_final"])
    y = pd.to_numeric(source["Y_final"]).to_numpy()
    calib = source["split"].eq("calib").to_numpy()
    pred_regular = regular["pred_chemberta_regular"].to_numpy(float)
    pred_augmented = augmented["pred_chemberta_augmented"].to_numpy(float)
    result = pd.DataFrame({"row_uid": source["row_uid"]})

    if task == "regression":
        spread = np.abs(pred_augmented - pred_regular)
        floor = max(float(np.quantile(spread[calib], 0.25)), float(np.std(y[calib])) * 1e-3, 1e-8)
        scale = spread + floor
        qhat = conformal_quantile(np.abs(y[calib] - pred_augmented[calib]) / scale[calib], alpha)
        half_width = qhat * scale
        result["conformal_lower"] = pred_augmented - half_width
        result["conformal_upper"] = pred_augmented + half_width
        result["conformal_width"] = 2 * half_width
        result["conformal_scale"] = scale
        result["conformal_qhat"] = qhat
        result["conformal_true_score"] = np.abs(y - pred_augmented) / scale
        metadata = {"task": task, "method": "normalized_split_conformal", "alpha": alpha, "qhat": qhat}
        return result, metadata

    p1 = np.clip(pred_augmented, 1e-7, 1 - 1e-7)
    labels = y.astype(int)
    uniform = np.asarray([deterministic_uniform(uid) for uid in source["row_uid"]])
    scores = randomized_aps_score(p1, labels, uniform)
    calib_scores = scores[calib]
    qhat = conformal_quantile(calib_scores, alpha)
    p0 = 1 - p1
    score0 = np.where(p0 >= p1, (1 - uniform) * p0, p1 + (1 - uniform) * p0)
    score1 = np.where(p1 >= p0, (1 - uniform) * p1, p0 + (1 - uniform) * p1)
    include0, include1 = score0 <= qhat, score1 <= qhat
    result["aps_prediction_set"] = np.where(
        include0 & include1, "[0,1]", np.where(include0, "[0]", np.where(include1, "[1]", "[]"))
    )
    result["aps_set_size"] = include0.astype(int) + include1.astype(int)
    result["aps_true_pvalue"] = [
        (1 + np.sum(calib_scores >= score)) / (len(calib_scores) + 1) for score in scores
    ]
    result["aps_calibrated_margin"] = result["aps_true_pvalue"] - alpha
    result["aps_qhat"] = qhat
    metadata = {"task": task, "method": "randomized_APS", "alpha": alpha, "qhat": qhat}
    return result, metadata


def array_to_bit_vector(row: np.ndarray) -> ExplicitBitVect:
    vector = ExplicitBitVect(int(row.shape[0]))
    vector.SetBitsFromList(np.flatnonzero(row).astype(int).tolist())
    return vector


def build_ad(source: pd.DataFrame, artifact_path: Path, k_neighbors: int, threshold: float) -> pd.DataFrame:
    with np.load(artifact_path, allow_pickle=False) as data:
        row_uid = data["row_uid"].astype(str)
        fingerprints = data["fingerprints"].astype(np.uint8)
    assert source["row_uid"].astype(str).equals(pd.Series(row_uid))
    train_indices = np.flatnonzero(source["split"].eq("train").to_numpy())
    train_position = {int(global_index): position for position, global_index in enumerate(train_indices)}
    all_vectors = [array_to_bit_vector(row) for row in fingerprints]
    train_vectors = [all_vectors[index] for index in train_indices]
    effective_k = min(k_neighbors, max(1, len(train_vectors) - 1))
    knn, density = [], []
    for index, query in enumerate(all_vectors):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query, train_vectors))
        if index in train_position:
            similarities[train_position[index]] = -1
        similarities = similarities[similarities >= 0]
        top = np.partition(similarities, len(similarities) - effective_k)[-effective_k:]
        knn.append(float(top.mean()))
        density.append(int(np.sum(similarities >= threshold)))
    return pd.DataFrame(
        {
            "row_uid": row_uid,
            "ad_knn_tanimoto_top5_mean": knn,
            "ad_local_density_count_s040": density,
            "ad_local_density_fraction_s040": np.asarray(density) / len(train_vectors),
        }
    )


def run_dataset(dataset: str, processed_dir: Path, output_dir: Path, artifact_dir: Path, config: dict) -> None:
    source = pd.read_csv(processed_dir / dataset / "splits.csv")
    dataset_output = output_dir / dataset
    fingerprint = pd.read_csv(dataset_output / "fingerprint_predictions.csv")
    regular = pd.read_csv(dataset_output / "chemberta_regular_predictions.csv")
    augmented = pd.read_csv(dataset_output / "chemberta_augmented_predictions.csv")
    frames = (fingerprint, regular, augmented)
    assert all(source["row_uid"].astype(str).equals(frame["row_uid"].astype(str)) for frame in frames)

    conformal, metadata = build_conformal(
        source, regular, augmented, float(config["conformal_alpha"])
    )
    conformal.to_csv(dataset_output / "conformal_predictions.csv", index=False)
    (dataset_output / "conformal_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ad = build_ad(
        source,
        artifact_dir / dataset / "morgan_r2_2048_chiral.npz",
        int(config["ad_k_neighbors"]),
        float(config["ad_density_similarity_threshold"]),
    )
    ad.to_csv(dataset_output / "applicability_domain.csv", index=False)

    fp_pred = fingerprint["pred_fp_primary"].to_numpy(float)
    chem_pred = augmented["pred_chemberta_augmented"].to_numpy(float)
    disagreement = pd.DataFrame(
        {"row_uid": source["row_uid"], "model_disagreement_abs": np.abs(fp_pred - chem_pred)}
    )
    if infer_task(source["Y_final"]) == "classification":
        disagreement["model_class_mismatch"] = (
            (fp_pred >= 0.5).astype(int) != (chem_pred >= 0.5).astype(int)
        ).astype(int)
        disagreement["model_disagreement_probability_gap"] = disagreement["model_disagreement_abs"]
    disagreement.to_csv(dataset_output / "model_disagreement.csv", index=False)

    combined = pd.DataFrame({"row_uid": source["row_uid"]})
    for frame in (*frames, conformal, ad, disagreement):
        columns = [column for column in frame if column != "row_uid" and column not in combined]
        combined = pd.concat([combined, frame[columns].reset_index(drop=True)], axis=1)
    combined.to_csv(dataset_output / "role2_signals.csv", index=False)
    print(f"[DONE] {dataset}: {len(combined)} rows, {len(combined.columns) - 1} signals")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/role2.yaml"))
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = args.dataset or sorted(path.parent.name for path in args.processed_dir.glob("*/splits.csv"))
    for dataset in datasets:
        run_dataset(dataset, args.processed_dir, args.output_dir, args.artifact_dir, config)


if __name__ == "__main__":
    main()

"""ChemBERTa 다중 seed 분산으로 회귀 컨포멀 구간을 다시 계산한다."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def infer_task(labels: pd.Series) -> str:
    values = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
    return "classification" if len(values) <= 2 and values <= {0, 1} else "regression"


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("유효한 calibration score가 없습니다.")
    level = min(1.0, math.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def build_ensemble_conformal(
    source: pd.DataFrame,
    multiseed: pd.DataFrame,
    alpha: float = 0.1,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> tuple[pd.DataFrame, dict]:
    """증강 ChemBERTa seed 평균을 중심, seed 표준편차를 적응형 척도로 사용한다."""
    if infer_task(source["Y_final"]) != "regression":
        raise ValueError("회귀 데이터만 지원합니다.")
    if source["row_uid"].astype(str).duplicated().any():
        raise ValueError("source의 row_uid가 중복됐습니다.")
    if multiseed["row_uid"].astype(str).duplicated().any():
        raise ValueError("multiseed의 row_uid가 중복됐습니다.")

    prediction_columns = [f"pred_chemberta_augmented_seed_{seed}" for seed in seeds]
    missing = [column for column in prediction_columns if column not in multiseed]
    if missing:
        raise ValueError(f"다중 seed 예측 열이 없습니다: {missing}")
    lookup = multiseed.assign(_uid=multiseed["row_uid"].astype(str)).set_index("_uid")
    try:
        aligned = lookup.loc[source["row_uid"].astype(str)]
    except KeyError as error:
        raise ValueError("다중 seed 예측에 일부 row_uid가 없습니다.") from error
    if not aligned[prediction_columns].notna().all().all():
        raise ValueError("다중 seed 예측에 결측값이 있습니다.")

    predictions = aligned[prediction_columns].to_numpy(float)
    center = predictions.mean(axis=1)
    ensemble_std = predictions.std(axis=1, ddof=0)
    labels = pd.to_numeric(source["Y_final"], errors="raise").to_numpy(float)
    calibration_mask = source["split"].eq("calib").to_numpy()
    if not calibration_mask.any():
        raise ValueError("calib 행이 없습니다.")

    scale_floor = max(
        float(np.quantile(ensemble_std[calibration_mask], 0.25)),
        float(np.std(labels[calibration_mask], ddof=0)) * 1e-3,
        1e-8,
    )
    scale = ensemble_std + scale_floor
    true_score = np.abs(labels - center) / scale
    qhat = conformal_quantile(true_score[calibration_mask], alpha)
    half_width = qhat * scale
    result = pd.DataFrame(
        {
            "row_uid": source["row_uid"],
            "conformal_center": center,
            "conformal_ensemble_std": ensemble_std,
            "conformal_lower": center - half_width,
            "conformal_upper": center + half_width,
            "conformal_width": 2 * half_width,
            "conformal_scale": scale,
            "conformal_qhat": qhat,
            "conformal_true_score": true_score,
        }
    )
    metadata = {
        "task": "regression",
        "method": "normalized_split_conformal",
        "alpha": alpha,
        "primary_model": "chemberta_augmented_seed_ensemble_mean",
        "center": f"mean of ChemBERTa augmented predictions across seeds {list(seeds)}",
        "adaptive_scale": "ChemBERTa augmented seed ensemble std (ddof=0) + calibration floor",
        "ensemble_variant": "augmented",
        "ensemble_seeds": list(seeds),
        "ensemble_ddof": 0,
        "scale_floor_rule": "max(calib Q25 of ensemble std, std(Y_calib)*1e-3, 1e-8)",
        "scale_floor": scale_floor,
        "qhat": qhat,
        "n_calib": int(calibration_mask.sum()),
    }
    return result, metadata


def load_multiseed_predictions(
    multiseed_dir: Path,
    dataset: str,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> pd.DataFrame:
    """통합 파일 또는 seed별 원본 파일에서 증강 예측을 불러온다."""
    combined_path = multiseed_dir / dataset / "chemberta_multiseed_predictions.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path, low_memory=False)

    frames: list[pd.DataFrame] = []
    for seed in seeds:
        candidates = [
            multiseed_dir
            / f"seed_{seed}"
            / dataset
            / "chemberta_augmented_predictions.csv",
            multiseed_dir
            / "outputs"
            / f"seed_{seed}"
            / dataset
            / "chemberta_augmented_predictions.csv",
        ]
        prediction_path = next((path for path in candidates if path.exists()), None)
        if prediction_path is None:
            checked = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"{dataset} seed {seed}의 증강 ChemBERTa 예측을 찾지 못했습니다: {checked}"
            )
        frame = pd.read_csv(
            prediction_path,
            usecols=["row_uid", "pred_chemberta_augmented"],
        ).rename(
            columns={
                "pred_chemberta_augmented": f"pred_chemberta_augmented_seed_{seed}"
            }
        )
        frames.append(frame)

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="row_uid", how="inner", validate="one_to_one")
    return combined


def replace_conformal_columns(signals: pd.DataFrame, conformal: pd.DataFrame) -> pd.DataFrame:
    """row_uid로 정렬해 기존 conformal_* 열만 새 결과로 교체한다."""
    old_columns = [column for column in signals if column.startswith("conformal_")]
    if not old_columns:
        raise ValueError("role2_signals.csv에 기존 conformal 열이 없습니다.")
    if signals["row_uid"].astype(str).duplicated().any():
        raise ValueError("role2_signals.csv의 row_uid가 중복됐습니다.")
    lookup = conformal.assign(_uid=conformal["row_uid"].astype(str)).set_index("_uid")
    aligned = lookup.loc[signals["row_uid"].astype(str)].reset_index(drop=True)
    insert_at = min(signals.columns.get_loc(column) for column in old_columns)
    before = [column for column in signals.columns[:insert_at] if column not in old_columns]
    after = [column for column in signals.columns[insert_at:] if column not in old_columns]
    new_columns = [column for column in conformal if column != "row_uid"]
    return pd.concat(
        [
            signals[before].reset_index(drop=True),
            aligned[new_columns],
            signals[after].reset_index(drop=True),
        ],
        axis=1,
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.report_dir / f"backup_before_{timestamp}"
    summary_rows: list[dict] = []
    inference_config: dict[str, dict] = {}
    if not args.dry_run:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=False)

    for split_path in sorted(args.processed_dir.glob("*/splits.csv")):
        dataset = split_path.parent.name
        source = pd.read_csv(split_path, low_memory=False)
        if infer_task(source["Y_final"]) != "regression":
            continue
        output = args.output_dir / dataset
        conformal, metadata = build_ensemble_conformal(
            source,
            load_multiseed_predictions(args.multiseed_dir, dataset, tuple(args.seeds)),
            args.alpha,
            tuple(args.seeds),
        )
        metadata["dataset"] = dataset
        metadata["calibration_split"] = "calib"
        test_mask = source["split"].eq("test").to_numpy()
        labels = pd.to_numeric(source["Y_final"], errors="raise").to_numpy(float)
        metadata["calib_coverage"] = float(
            np.mean(
                (labels[source["split"].eq("calib")] >= conformal.loc[source["split"].eq("calib"), "conformal_lower"])
                & (labels[source["split"].eq("calib")] <= conformal.loc[source["split"].eq("calib"), "conformal_upper"])
            )
        )
        metadata["test_coverage"] = float(
            np.mean(
                (labels[test_mask] >= conformal.loc[test_mask, "conformal_lower"])
                & (labels[test_mask] <= conformal.loc[test_mask, "conformal_upper"])
            )
        )
        metadata["test_mean_width"] = float(conformal.loc[test_mask, "conformal_width"].mean())
        inference_config[dataset] = metadata
        summary_rows.append(
            {
                "dataset": dataset,
                "rows": len(source),
                "calib_rows": metadata["n_calib"],
                "test_rows": int(test_mask.sum()),
                "test_coverage": metadata["test_coverage"],
                "test_mean_width": metadata["test_mean_width"],
                "scale_floor": metadata["scale_floor"],
                "qhat": metadata["qhat"],
            }
        )
        if args.dry_run:
            print(f"[DRY-RUN] {dataset}")
            continue

        dataset_backup = backup_dir / dataset
        dataset_backup.mkdir(parents=True)
        for name in ["conformal_predictions.csv", "conformal_metadata.json", "role2_signals.csv"]:
            shutil.copy2(output / name, dataset_backup / name)
        signals = pd.read_csv(output / "role2_signals.csv", low_memory=False)
        updated_signals = replace_conformal_columns(signals, conformal)
        atomic_csv(conformal, output / "conformal_predictions.csv")
        atomic_json(metadata, output / "conformal_metadata.json")
        atomic_csv(updated_signals, output / "role2_signals.csv")
        print(f"[APPLIED] {dataset}")

    summary = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    if len(summary) != 9:
        raise RuntimeError(f"회귀 9종을 예상했지만 {len(summary)}종을 찾았습니다.")
    if not args.dry_run:
        atomic_csv(summary, args.report_dir / "conformal_ensemble_summary.csv")
        atomic_json(
            {
                "method": "normalized split conformal with ChemBERTa augmented seed ensemble dispersion",
                "datasets": inference_config,
                "backup_directory": str(backup_dir),
            },
            args.report_dir / "inference_config.json",
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

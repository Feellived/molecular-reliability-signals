#!/usr/bin/env python
"""분자 하나를 채점하기 위한 정적 산출물을 내보낸다 (연구계획서 7.3절).

지금 파이프라인은 분자 하나를 새로 받아 답할 수 없다. 이유가 셋이다.

  백분위    신호값을 순위로 바꾸려면 비교 대상이 필요한데, 현재 구현은
            데이터셋 전체를 놓고 순위를 매긴다. 분자 하나는 자기 자신과
            순위를 매길 수 없다
  결합 규칙  여러 신호를 하나의 위험 점수로 합치는 능선 회귀를 평가
            스크립트가 매번 새로 적합하고 버린다. 물성마다 계수도 특징
            목록도 다른데 아무데도 남지 않는다
  지문 모델  22종 중 8종은 XGBoost가 대표인데 저장본은 랜덤 포레스트뿐이다

이 셋을 정적인 파일 묶음으로 얼려 데모가 파이프라인을 몰라도 되게 한다.

같이 해결되는 것이 하나 있다. 계획서 5.8절은 백분위 기준을 시험 집합이
아니라 보정 집합에서 고정하도록 정했는데 현재 구현은 메타와 시험을 합쳐
순위를 매긴다. 참조 분포를 내보내는 일이 곧 그 교체이며, 계획서 개정 2가
교체 시점을 도구 제작 단계로 적어둔 것이 이 작업이다.

참조 분할은 메타 보정 분할을 쓴다. 축 신호는 메타와 시험에만 생성되어
컨포멀 보정 분할에는 존재하지 않고, 결합 규칙도 메타에서 학습하므로
참조와 규칙의 출처를 맞추는 편이 일관된다.

적용가능도메인 신호는 새 분자를 학습 집합과 비교해야 산출되므로 학습
분할의 SMILES도 함께 내보낸다. 지문 행렬 대신 SMILES를 저장하는 이유는
용량이 훨씬 작고 사람이 확인할 수 있기 때문이며, 데모는 시작할 때 한 번
지문으로 변환한다.

재적합이 담당2의 모델을 재현하는지도 함께 검증해 기록한다. 변형 채점은
새 분자를 넣을 수 있어야 하므로 처음부터 재적합 모델을 써왔고, 따라서
데모의 예측값도 재적합에서 나온다. 두 예측이 얼마나 일치하는지는 데모
수치를 보고서 수치와 나란히 놓을 때 필요한 정보다.

설정 지문을 함께 적는다. 나중에 A축 변형을 30개로 늘리면 참조 분포가
낡는데, 지문을 대조하지 않으면 데모가 틀린 백분위를 태연히 내놓는다.
데모는 지문이 맞지 않으면 실행을 거부해야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

import generate_variants as gv
from score_variants_fingerprint import (
    MORGAN_BITS, MORGAN_CHIRALITY, MORGAN_RADIUS,
    make_model, make_morgan_matrix,
)
from dataset_repairs import apply_known_repairs

REFERENCE_SPLIT = "meta"
SEEDS = [42, 43, 44, 45, 46]
ALPHA = 0.1
SIGNALS = {
    "base__ad_knn": "적용가능도메인 이웃",
    "base__ad_density": "적용가능도메인 밀도",
    "base__disagreement": "모델 불일치",
    "base__conformal_cb": "컨포멀 ChemBERTa",
    "base__conformal_fp": "컨포멀 지문",
    "axis__cb_augmented__A": "표현 불안정성 A",
    "cond_B__fp_primary__std": "입력 상태 민감성 B (지문)",
    "cond_B__cb_augmented__std": "입력 상태 민감성 B (언어모델)",
}
AD_DEFINITIONS = {
    "ad_knn_tanimoto_top5_mean": "학습 분할에서 가장 가까운 5개 분자와의 Tanimoto 평균",
    "ad_local_density_count_s040": "Tanimoto 0.40 이상인 학습 분자의 개수",
    "ad_local_density_fraction_s040": "그 개수를 학습 분자 수로 나눈 값",
}
AXIS_LABELS = {
    "B1_tautomer": "호변이성질체",
    "B1_protonation": "양성자화",
    "B2_salt": "염 형태",
    "B3_stereo": "입체 표기",
}


def settings_fingerprint() -> dict:
    """변형 생성 설정의 지문. 설정이 바뀌면 참조 분포가 낡는다."""
    settings = {
        "A_AXIS_K": gv.A_AXIS_K,
        "MAX_TAUTOMERS": gv.MAX_TAUTOMERS,
        "PH_MIN": gv.PH_MIN,
        "PH_MAX": gv.PH_MAX,
        "B3_MODE": "full_strip",
        "MORGAN_RADIUS": MORGAN_RADIUS,
        "MORGAN_BITS": MORGAN_BITS,
        "MORGAN_CHIRALITY": MORGAN_CHIRALITY,
    }
    payload = json.dumps(settings, sort_keys=True).encode("utf-8")
    return {"settings": settings, "digest": hashlib.sha256(payload).hexdigest()[:16]}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                             method="higher"))


def normalized_aurc(score, error) -> float:
    curve = lambda x: float(np.mean(
        np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, random = curve(error), float(np.mean(error))
    return np.nan if random - oracle < 1e-12 else (curve(score) - oracle) / (random - oracle)


def build_reference(frame: pd.DataFrame) -> dict:
    """보정 분할의 신호값을 정렬해 저장한다. 새 값은 여기 이진 탐색해 백분위가 된다."""
    reference = frame[frame["split"].eq(REFERENCE_SPLIT)]
    out = {}
    for column, label in SIGNALS.items():
        if column not in frame or reference[column].nunique() <= 1:
            continue
        values = np.sort(reference[column].to_numpy(dtype=float))
        out[column] = {"label": label, "n": len(values),
                       "sorted_values": [round(float(v), 8) for v in values]}
    return out


def to_percentile(values, sorted_reference) -> np.ndarray:
    """보정 분포의 경험 누적분포로 백분위를 매긴다. 기준이 분할 밖에 고정된다."""
    return np.searchsorted(sorted_reference, np.asarray(values, dtype=float),
                           side="right") / len(sorted_reference)


# 분자 하나로는 재현할 수 없는 특징. 담당2의 ChemBERTa 컨포멀 보정값이
# 산출물에 없어 새 분자의 예측 집합 크기를 만들 수 없다.
NOT_REPRODUCIBLE = ("base__conformal_cb",)


def build_combiner(frame: pd.DataFrame, reference: dict, columns=None) -> dict:
    """메타에서 능선 회귀를 적합해 계수를 남긴다. 백분위는 참조 분포로 매긴다."""
    columns = list(reference) if columns is None else [c for c in columns if c in reference]
    matrix = np.column_stack([
        to_percentile(frame[c].to_numpy(float), reference[c]["sorted_values"])
        for c in columns])
    meta = frame["split"].eq(REFERENCE_SPLIT).to_numpy()
    test = frame["split"].eq("test").to_numpy()

    target = rankdata(frame.loc[meta, "abs_error_fp"]) / meta.sum()
    model = Ridge(alpha=1.0).fit(matrix[meta], target)

    error = frame.loc[test, "abs_error_fp"].to_numpy(float)
    label = ((((frame.loc[test, "pred_fp_primary"].to_numpy(float) >= .5).astype(int)
               != pd.to_numeric(frame.loc[test, "Y_final"]).astype(int)).astype(int))
             if frame["task_type"].iloc[0] == "classification"
             else (error >= np.quantile(error, .8)).astype(int))
    score = model.predict(matrix[test])
    return {
        "features": columns,
        "coefficients": [round(float(c), 8) for c in model.coef_],
        "intercept": round(float(model.intercept_), 8),
        "alpha": 1.0,
        "fitted_on": REFERENCE_SPLIT,
        "target": "메타 분할 절대오차의 백분위 순위",
        "test_auprc": (round(float(average_precision_score(label, score)), 6)
                       if label.min() != label.max() else None),
        "test_normalized_aurc": round(float(normalized_aurc(score, error)), 6),
    }


def build_conformal(frame: pd.DataFrame, role2_dir: Path, dataset: str,
                    scores_dir: Path) -> dict:
    """예측 구간을 그리는 데 필요한 값. 분류는 구간 대신 예측 집합을 낸다."""
    task = frame["task_type"].iloc[0]
    aps = scores_dir / "fp_conformal" / dataset / "fp_conformal_metadata.json"
    role2 = pd.read_csv(role2_dir / dataset / "role2_signals.csv",
                        usecols=["row_uid", "std_fp_primary", "split", "Y_final",
                                 "pred_fp_primary"])
    calib = role2["split"].eq("calib").to_numpy()
    if task == "classification":
        record = {"task_type": task, "alpha": ALPHA,
                  "note": "분류는 구간 대신 무작위화 APS 예측 집합을 낸다"}
        if aps.exists():
            record["aps_qhat"] = json.loads(aps.read_text(encoding="utf-8"))["qhat"]
            record["randomization_tag"] = "mist-fp-aps-v1"
        return record
    if not calib.any():
        return {"task_type": task, "note": "보정 분할이 없어 구간을 산출하지 않는다"}

    raw = role2["std_fp_primary"].to_numpy(float)
    floor = float(np.quantile(raw[calib], 0.25))
    if not np.isfinite(floor) or floor <= 0:
        floor = float(np.std(role2.loc[calib, "Y_final"])) * 0.01 or 1e-6
    scale = raw + floor
    truth = pd.to_numeric(role2["Y_final"]).to_numpy(float)
    point = role2["pred_fp_primary"].to_numpy(float)
    qhat = conformal_quantile(np.abs(truth[calib] - point[calib]) / scale[calib], ALPHA)
    return {"task_type": task, "alpha": ALPHA, "qhat": round(qhat, 8),
            "scale_floor": round(floor, 8),
            "scale": "시드 간 표준편차에 보정 분할 25백분위 하한을 더한 값",
            "interval": "예측값 ± qhat × 척도"}


def build_axes(dataset: str, allowance: pd.DataFrame, decision: pd.DataFrame) -> dict:
    """이 물성에 어느 축을 쓸 수 있는지. 이 연구의 고유한 부분이다."""
    out = {}
    for axis, label in AXIS_LABELS.items():
        allowed = allowance.at[dataset, axis] if axis in allowance.columns else "미기재"
        used = decision.at[dataset, f"use_{axis}"] if f"use_{axis}" in decision.columns else "미기재"
        out[label] = {
            "axis": axis,
            "물성_허용성": str(allowed),
            "표본_판정": str(used),
            "사용": bool(str(used) == "사용" and (
                str(allowed) == "허용" or axis == "B3_stereo")),
        }
    out["_설명"] = ("물성 허용성은 그 변형이 이 물성의 측정 정의와 충돌하지 않는지, "
                    "표본 판정은 실제로 변형이 생성되는지를 본다. 둘 다 통과한 축만 쓴다. "
                    "입체 표기는 분자 단위 조건이므로 물성 허용성으로 거르지 않는다.")
    return out


def export_neighbors(dataset: str, processed_dir: Path, out_dir: Path) -> dict:
    """적용가능도메인 신호를 새 분자에 대해 계산하려면 학습 집합이 필요하다."""
    splits = pd.read_csv(processed_dir / dataset / "splits.csv", low_memory=False)
    splits, _ = apply_known_repairs(splits)
    train = splits[splits["split"].eq("train")][["row_uid", "parent_smiles"]]
    train.to_csv(out_dir / "neighbors.csv", index=False)
    return {"n_train": len(train), "definitions": AD_DEFINITIONS,
            "fingerprint": {"radius": MORGAN_RADIUS, "bits": MORGAN_BITS,
                            "chirality": MORGAN_CHIRALITY}}


def verify_refit(dataset: str, scores_dir: Path, role2_dir: Path) -> dict | None:
    """재적합 예측이 담당2 모델을 재현하는지 잰다."""
    path = scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv"
    if not path.exists():
        return None
    refit = pd.read_csv(path, usecols=["row_uid", "pred_fp_primary"]).rename(
        columns={"pred_fp_primary": "refit"})
    role2 = pd.read_csv(role2_dir / dataset / "role2_signals.csv",
                        usecols=["row_uid", "pred_fp_primary", "split"])
    merged = refit.merge(role2, on="row_uid")
    merged = merged[merged["split"].isin(["meta", "test"])]
    if len(merged) < 10:
        return None
    a = merged["refit"].to_numpy(float)
    b = merged["pred_fp_primary"].to_numpy(float)
    span = float(b.max() - b.min())
    return {"n": len(merged), "correlation": round(float(np.corrcoef(a, b)[0, 1]), 6),
            "mean_abs_diff_over_range": round(float(np.abs(a - b).mean() / span), 6)
            if span > 0 else None}


def cache_models(dataset: str, processed_dir: Path, role2_dir: Path, out_dir: Path) -> dict:
    """대표 모델을 재적합해 캐시한다. XGBoost 대표 8종은 저장본이 없다."""
    primary = pd.read_csv(role2_dir / dataset / "role2_signals.csv",
                          usecols=["fp_primary_model"], nrows=1)["fp_primary_model"].iloc[0]
    splits = pd.read_csv(processed_dir / dataset / "splits.csv", low_memory=False)
    splits, _ = apply_known_repairs(splits)
    task = splits["task_type"].iloc[0]
    matrix = make_morgan_matrix(splits["parent_smiles"])
    train = splits["split"].eq("train").to_numpy()
    labels = splits["Y_final"].astype(int if task == "classification" else float).to_numpy()

    target = out_dir / "models"; target.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        model = make_model(primary, task, seed)
        model.fit(matrix[train], labels[train])
        joblib.dump(model, target / f"{primary}_seed_{seed}.joblib", compress=3)
    return {"primary_model": primary, "task_type": task, "seeds": SEEDS,
            "n_train": int(train.sum()),
            "fingerprint": {"radius": MORGAN_RADIUS, "bits": MORGAN_BITS,
                            "chirality": MORGAN_CHIRALITY}}


def main() -> int:
    parser = argparse.ArgumentParser(description="데모용 정적 산출물 내보내기")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--role2-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--with-models", action="store_true",
                        help="지문 모델 재적합 캐시까지 만든다. 오래 걸린다")
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    role2_dir, processed_dir = Path(args.role2_dir), Path(args.processed_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    allowance = pd.read_csv(reports / "06_transformation_allowance_revised.csv").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")

    datasets = args.datasets or sorted(
        p.name for p in evaluation_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))

    records = []
    for index, dataset in enumerate(datasets, 1):
        frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
        target = out_dir / dataset; target.mkdir(parents=True, exist_ok=True)

        reference = build_reference(frame)
        (target / "reference.json").write_text(
            json.dumps(reference, ensure_ascii=False), encoding="utf-8")
        combiner = build_combiner(frame, reference)
        (target / "combiner.json").write_text(
            json.dumps(combiner, ensure_ascii=False, indent=2), encoding="utf-8")
        single = build_combiner(frame, reference,
                                [c for c in reference if c not in NOT_REPRODUCIBLE])
        single["note"] = ("분자 하나로 재현 가능한 특징만 쓴다. 담당2의 ChemBERTa "
                          "컨포멀 보정값이 산출물에 없어 그 신호를 뺐다.")
        (target / "combiner_single.json").write_text(
            json.dumps(single, ensure_ascii=False, indent=2), encoding="utf-8")
        (target / "conformal.json").write_text(
            json.dumps(build_conformal(frame, role2_dir, dataset, Path(args.scores_dir)),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        (target / "axes.json").write_text(
            json.dumps(build_axes(dataset, allowance, decision), ensure_ascii=False,
                       indent=2), encoding="utf-8")

        neighbors = export_neighbors(dataset, processed_dir, target)
        (target / "neighbors.json").write_text(
            json.dumps(neighbors, ensure_ascii=False, indent=2), encoding="utf-8")
        refit = verify_refit(dataset, Path(args.scores_dir), role2_dir)
        if refit:
            (target / "refit_check.json").write_text(
                json.dumps(refit, ensure_ascii=False, indent=2), encoding="utf-8")

        model_info = None
        if args.with_models:
            model_info = cache_models(dataset, processed_dir, role2_dir, target)
            (target / "models.json").write_text(
                json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")

        records.append({"dataset": dataset, "task_type": frame["task_type"].iloc[0],
                        "n_reference": int(frame["split"].eq(REFERENCE_SPLIT).sum()),
                        "n_signals": len(reference),
                        "test_auprc": combiner["test_auprc"],
                        "test_normalized_aurc": combiner["test_normalized_aurc"],
                        "single_auprc": single["test_auprc"],
                        "single_normalized_aurc": single["test_normalized_aurc"],
                        "n_train_neighbors": neighbors["n_train"],
                        "refit_correlation": refit["correlation"] if refit else None,
                        "models_cached": bool(model_info)})
        print(f"[{index}/{len(datasets)}] {dataset:32s} 신호 {len(reference)}종  "
              f"참조 {records[-1]['n_reference']:4d}행  AUPRC {combiner['test_auprc']}",
              flush=True)

    fingerprint = settings_fingerprint()
    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "reference_split": REFERENCE_SPLIT,
        "reference_split_note": (
            "축 신호는 메타와 시험에만 생성되어 컨포멀 보정 분할에는 없다. "
            "결합 규칙도 메타에서 학습하므로 참조와 규칙의 출처를 맞춘다."),
        "percentile_rule": "보정 분포의 경험 누적분포로 고정 (계획서 5.8절, 개정 2 일탈 2 해소)",
        "applicability_domain": AD_DEFINITIONS,
        "refit_note": (
            "변형 채점은 새 분자를 넣을 수 있어야 하므로 재적합 모델을 쓴다. "
            "데모 예측값도 재적합에서 나오며 담당2 모델과의 일치도를 물성마다 기록했다."),
        **fingerprint,
        "datasets": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pd.DataFrame(records)
    print(f"\n설정 지문 {fingerprint['digest']}   물성 {len(table)}종")
    print(f"결합 점수 시험 성적  AUPRC {table['test_auprc'].mean():.4f}   "
          f"정규화 AURC {table['test_normalized_aurc'].mean():.4f}")
    print(f"분자 하나용 규칙      AUPRC {table['single_auprc'].mean():.4f}   "
          f"정규화 AURC {table['single_normalized_aurc'].mean():.4f}")
    corr = table["refit_correlation"].dropna()
    if len(corr):
        print(f"재적합 재현도  상관 중앙값 {corr.median():.4f}  최솟값 {corr.min():.4f}")
    print(f"산출물 {out_dir}")
    if not args.with_models:
        print("모델 캐시는 --with-models로 따로 만든다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

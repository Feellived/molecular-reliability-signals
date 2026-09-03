#!/usr/bin/env python
"""축별 게이트 후보를 결과와 무관하게 정의하고 서로 비교한다 (연구계획서 5.2절).

현행 조건은 축마다 근거의 강도가 다르다. 양성자화는 측정 pH가 정의됐는지를
물어 22종을 15대 7로 가르지만, 호변이성질체는 사실상 대부분 통과하고
입체 표기는 물성 단위 조건이 아예 없다.

여기서 검토하는 후보는 모두 결과를 보기 전에 화학 또는 데이터 출처만으로
정당화된다. 성능이 좋아지는 분할을 찾는 것이 목적이 아니라, 계획서 5.2절이
요구한 축별 조건을 근거 있는 형태로 채우는 것이 목적이다.

  G0  현행. 호변이성질체와 양성자화 각각의 허용 판정
  G1  매질 통제. 호변이성질체 평형도 pH와 용매 극성에 따라 이동하므로,
      측정 조건이 보존되지 않은 물성에서는 평형 자체가 정의되지 않는다.
      양성자화에 건 것과 같은 게이트를 호변이성질체에도 건다
  G2  입체 출처. 원본 등록에서 입체중심 배치가 지정되지 않은 비율이 높은
      물성은 입체 표기를 통제하지 않은 것이므로 B3를 쓰지 않는다
  G3  G1과 G2를 함께 적용

판정은 run_preregistered_ablation.py와 같은 절차를 쓴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import FindMolChiralCenters
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata, wilcoxon

RDLogger.DisableLog("rdApp.*")

MODELS = {"fp_primary": "pred_fp_primary", "cb_augmented": "pred_chemberta_augmented"}
ZERO_TOLERANCE = 1e-12
UNASSIGNED_CUTOFF = 0.5  # 미지정률 분포의 자연 단절 구간 (0.350과 0.747 사이)
BASE_PRE = ["base__ad_knn__pct", "base__ad_density__pct",
            "base__conformal_cb__pct", "base__conformal_fp__pct"]
A_FEAT = ["axis__cb_augmented__A__pct"]
B_FEAT = ["cond_B__fp_primary__std__pct", "cond_B__cb_augmented__std__pct"]


def unassigned_ratio(splits_csv: Path) -> float:
    """원본 등록에서 배치가 지정되지 않은 입체중심의 비율."""
    frame = pd.read_csv(splits_csv, usecols=["parent_smiles"])
    assigned = unassigned = 0
    for smiles in frame["parent_smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        for _, tag in FindMolChiralCenters(mol, includeUnassigned=True,
                                           useLegacyImplementation=False):
            if tag == "?":
                unassigned += 1
            else:
                assigned += 1
    total = assigned + unassigned
    return unassigned / total if total else float("nan")


def gate_axes(dataset, allowance, decision, gate, ratios) -> tuple[str, ...]:
    ok = lambda a: decision.at[dataset, f"use_{a}"] == "사용"
    ph_defined = allowance.at[dataset, "B1_protonation"] == "허용"
    axes = []
    if ok("B1_tautomer"):
        allowed = allowance.at[dataset, "B1_tautomer"] == "허용"
        if gate in ("G1", "G3"):
            allowed = allowed and ph_defined
        if allowed:
            axes.append("B1_tautomer")
    if ok("B1_protonation") and ph_defined:
        axes.append("B1_protonation")
    if ok("B3_stereo"):
        if gate not in ("G2", "G3") or ratios.get(dataset, 0.0) < UNASSIGNED_CUTOFF:
            axes.append("B3_stereo")
    return tuple(axes)


def conditional_std(dataset, scores_dir, axes, row_uids) -> pd.DataFrame:
    """조건이 성립하는 축의 변형만 하나의 표본으로 합쳐 흩어짐을 낸다.

    표본에는 원본 예측을 포함한다. build_conditional_signals.py와 같은 규칙이다.
    B3 입체는 분자당 변형이 정확히 1개여서, 원본을 빼면 표준편차가 정의되지
    않아 축 전체가 0으로 사라진다.
    """
    out = pd.DataFrame({"row_uid": row_uids})
    for key in MODELS:
        out[f"cond_B__{key}__std"] = 0.0
    if not axes:
        return out
    fp = pd.read_csv(scores_dir / "fingerprint" / dataset / "variant_predictions_fp.csv")
    cb = pd.read_csv(
        scores_dir / "chemberta" / dataset / "variant_predictions_chemberta.csv"
    ).drop(columns=["dataset", "axis", "split", "parent_row_uid"], errors="ignore")
    variants = fp.merge(cb, on="variant_uid")
    variants = variants[variants["axis"].isin(axes)]
    if variants.empty:
        return out

    origin = pd.read_csv(
        scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv"
    ).merge(
        pd.read_csv(
            scores_dir / "chemberta" / dataset / "origin_predictions_chemberta.csv"
        ).drop(columns=["dataset", "split"], errors="ignore"),
        on="row_uid",
    ).set_index("row_uid")

    rows = []
    for row_uid, group in variants.groupby("parent_row_uid"):
        record = {"row_uid": row_uid}
        for key, column in MODELS.items():
            sample = np.append(group[column].to_numpy(float),
                               float(origin.at[row_uid, column]))
            std = float(np.std(sample, ddof=0))
            record[f"cond_B__{key}__std"] = 0.0 if std < ZERO_TOLERANCE else std
        rows.append(record)
    return out.drop(columns=[f"cond_B__{k}__std" for k in MODELS]).merge(
        pd.DataFrame(rows), on="row_uid", how="left").fillna(0.0)


def normalized_aurc(score, error) -> float:
    curve = lambda x: float(np.mean(
        np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, random = curve(error), float(np.mean(error))
    return np.nan if random - oracle < 1e-12 else (curve(score) - oracle) / (random - oracle)


def evaluate_dataset(frame, cond) -> dict:
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("cond_B__")],
                       errors="ignore").merge(cond, on="row_uid", how="left")
    for key in MODELS:
        col = f"cond_B__{key}__std"
        frame[col] = frame[col].fillna(0.0)
        frame[f"{col}__pct"] = frame[col].rank(pct=True, method="average")

    meta, test = frame[frame.split == "meta"], frame[frame.split == "test"]
    error = test.abs_error_fp.to_numpy(float)
    target = rankdata(meta.abs_error_fp) / len(meta)
    label = (((test.pred_fp_primary.to_numpy(float) >= .5).astype(int)
              != pd.to_numeric(test.Y_final).astype(int)).astype(int)
             if frame.task_type.iloc[0] == "classification"
             else (error >= np.quantile(error, .8)).astype(int))

    result = {}
    for name, feats in {"기준": BASE_PRE, "기준+A": BASE_PRE + A_FEAT,
                        "기준+B": BASE_PRE + B_FEAT,
                        "기준+A+B": BASE_PRE + A_FEAT + B_FEAT}.items():
        use = [c for c in feats if c in frame and frame[c].nunique() > 1]
        score = Ridge(alpha=1.0).fit(meta[use].to_numpy(float), target
                                     ).predict(test[use].to_numpy(float))
        result[f"auprc__{name}"] = (average_precision_score(label, score)
                                    if label.min() != label.max() else np.nan)
        result[f"aurc__{name}"] = normalized_aurc(score, error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="축 게이트 후보 비교")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir, scores_dir = Path(args.evaluation_dir), Path(args.scores_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    allowance = pd.read_csv(reports / "06_transformation_allowance_revised.csv").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")

    datasets = sorted(p.name for p in evaluation_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    ratios = {d: unassigned_ratio(Path(args.pipeline_dir) / d / "splits.csv") for d in datasets}
    pd.Series(ratios, name="unassigned_ratio").rename_axis("dataset").to_csv(
        out_dir / "stereo_unassigned_ratio.csv")

    gates = ("G0", "G1", "G2", "G3")
    tables, membership = {}, []
    for gate in gates:
        rows = []
        for dataset in datasets:
            axes = gate_axes(dataset, allowance, decision, gate, ratios)
            membership.append({"gate": gate, "dataset": dataset, "axes": "+".join(axes) or "없음"})
            frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
            cond = conditional_std(dataset, scores_dir, axes, frame["row_uid"])
            rows.append({"dataset": dataset, **evaluate_dataset(frame, cond)})
        tables[gate] = pd.DataFrame(rows)
        tables[gate].to_csv(out_dir / f"gate_{gate}.csv", index=False)
        print(f"{gate} 완료", flush=True)

    pd.DataFrame(membership).to_csv(out_dir / "gate_membership.csv", index=False)

    rng = np.random.default_rng(42)
    summary = {}
    print(f"\n{'게이트':6s}{'축쓰는물성':>10s}{'AUPRC 기준+B':>14s}{'AURC 기준+B':>13s}"
          f"{'B 추가 효과(AURC)':>20s}{'95% 구간':>22s}{'개선':>8s}")
    for gate in gates:
        t = tables[gate]
        n_used = sum(1 for m in membership if m["gate"] == gate and m["axes"] != "없음")
        delta = (t["aurc__기준"] - t["aurc__기준+B"]).dropna()
        boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary[gate] = {"n_datasets_with_axes": n_used,
                         "aurc_gain": float(delta.mean()),
                         "ci": [float(lo), float(hi)],
                         "n_improved": int((delta > 0).sum()), "n": len(delta),
                         "wilcoxon_p": float(wilcoxon(delta)[1]),
                         "auprc_gain": float((t["auprc__기준+B"] - t["auprc__기준"]).dropna().mean())}
        print(f"{gate:6s}{n_used:10d}{t['auprc__기준+B'].mean():14.4f}"
              f"{t['aurc__기준+B'].mean():13.4f}{delta.mean():+20.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22s}{f'{int((delta>0).sum())}/{len(delta)}':>8s}")
    (out_dir / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

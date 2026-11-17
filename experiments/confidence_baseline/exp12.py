#!/usr/bin/env python
"""사전 기록 4의 실험 1(확신도 기준선)과 실험 2(증거의 독립성).

정의는 보관본의 기존 코드를 그대로 불러 쓴다.
  naurc, build_frame, 재현 구성   scripts/run_replication_eval.py
  조건부 B 축 선택                scripts/build_conditional_signals.allowed_axes
한 번만 실행한다. 구성과 판정은 사전기록_4_확신도기준선_라이브러리의존.md에 고정되어 있다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

ARCH = Path(__file__).resolve().parents[2]  # 저장소 루트 (실행 당시 절대경로를 저장소 상대경로로 바꿨다. 논리 변경 없음)
sys.path.insert(0, str(ARCH / "scripts"))
from run_replication_eval import build_frame, naurc, STATS as REP_STATS  # noqa: E402
from build_conditional_signals import allowed_axes  # noqa: E402

OUT = Path(__file__).resolve().parent
PROC = ARCH / "data/processed"
R2 = PROC / "scores_role4_r2"
SPL = PROC / "pipeline_yoonsoo"
N_BOOT, SEED = 2000, 20261117

BASE4 = ["base__ad_knn__pct", "base__ad_density__pct", "base__conformal_cb__pct", "base__conformal_fp__pct"]
B_STD = ["cond_B__fp_primary__std__pct", "cond_B__cb_augmented__std__pct"]
FP_BASE = ["base__ad_knn__pct", "base__ad_density__pct", "base__conformal_fp__pct"]
FP_SHAPE = [f"rich__fp_primary__B__{s}__pct" for s in ("std", "max_dev", "shift", "rel_std", "flip")]
NEW = ("cyp1a2_veith", "cyp2c19_veith")
VEITH3 = ("cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith")
SUBSTRATE = ("cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels")
CYP6 = VEITH3 + SUBSTRATE


def load_r2(dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(R2 / "evaluation" / dataset / "evaluation_signals.csv")
    groups = pd.read_csv(SPL / dataset / "splits.csv", usecols=["row_uid", "scaffold_group"])
    frame = frame.merge(groups, on="row_uid", how="left")
    for column, source in (("conf_fp", "pred_fp_primary"), ("conf_cb", "pred_chemberta_augmented")):
        frame[column] = (frame[source].astype(float) - 0.5).abs()
        frame[f"{column}__pct"] = frame[column].rank(pct=True, method="average")
    return frame


def score(fit: pd.DataFrame, evl: pd.DataFrame, frame: pd.DataFrame, features: list[str], err_col: str) -> np.ndarray:
    """meta(또는 지정한 분할)로 적합해 평가 분할의 점수를 낸다. 기존 코드와 같은 열 선택 규칙."""
    used = [c for c in features if c in frame and frame[c].nunique() > 1]
    target = rankdata(fit[err_col]) / len(fit)
    return Ridge(alpha=1.0).fit(fit[used].to_numpy(float), target).predict(evl[used].to_numpy(float))


def cluster_index(groups: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(groups == g) for g in pd.unique(groups)]


def boot_independent(err, ref, cand, groups, rng, n_boot=N_BOOT) -> np.ndarray:
    index = cluster_index(groups)
    out = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([index[k] for k in rng.integers(0, len(index), len(index))])
        out[i] = naurc(ref[rows], err[rows]) - naurc(cand[rows], err[rows])
    return out


def ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.nanpercentile(values, [2.5, 97.5])]


# ---------------------------------------------------------------- 재현 점검
def checks() -> dict:
    result = {}
    stored = pd.read_csv(R2 / "b1_statistics/b1_by_dataset.csv").set_index("dataset")
    diffs = {}
    for dataset in stored.index:
        f = load_r2(dataset)
        meta, test = f[f.split == "meta"], f[f.split == "test"].reset_index(drop=True)
        err = test.abs_error_fp.to_numpy(float)
        ref = score(meta, test, f, BASE4, "abs_error_fp")
        cand = score(meta, test, f, BASE4 + B_STD, "abs_error_fp")
        diffs[dataset] = (naurc(ref, err) - naurc(cand, err)) - float(stored.at[dataset, "effect__기준+B"])
    result["check1_max_abs_diff"] = max(abs(v) for v in diffs.values())
    result["check1_pass"] = result["check1_max_abs_diff"] < 1e-6

    effects = []
    for dataset in stored.index:
        f = load_r2(dataset)
        meta, test = f[f.split == "meta"], f[f.split == "test"].reset_index(drop=True)
        err = test.abs_error_fp.to_numpy(float)
        ref = score(meta, test, f, FP_BASE, "abs_error_fp")
        cand = score(meta, test, f, FP_BASE + FP_SHAPE, "abs_error_fp")
        effects.append(naurc(ref, err) - naurc(cand, err))
    result["check2_fp_config_mean22"] = float(np.nanmean(effects))
    result["check2_pass"] = abs(result["check2_fp_config_mean22"] - 0.073) <= 0.005

    saved = json.loads((PROC / "scores_new/replication/replication_result.json").read_text(encoding="utf-8"))
    result["check3"] = {}
    ok = True
    for dataset in NEW:
        f = new_frame(dataset)
        eff = rep_effect(f)
        diff = eff - saved[dataset]["1차"]["effect"]
        result["check3"][dataset] = {"effect": eff, "saved": saved[dataset]["1차"]["effect"], "diff": diff}
        ok &= abs(diff) < 1e-4
    result["check3_pass"] = bool(ok)
    result["all_pass"] = bool(result["check1_pass"] and result["check2_pass"] and result["check3_pass"])
    return result


# ---------------------------------------------------------------- 재현 데이터
def new_frame(dataset: str) -> pd.DataFrame:
    reports = PROC / "pipeline_new/reports"
    allowance = pd.read_csv(reports / "06_transformation_allowance_final.csv", encoding="utf-8-sig").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")
    axes = allowed_axes(dataset, allowance, decision)
    task = pd.read_csv(PROC / "pipeline_new" / dataset / "splits.csv", usecols=["task_type"]).task_type.iloc[0]
    frame = build_frame(dataset, PROC / "pipeline_new", PROC / "role2_new", PROC / "scores_new", axes, task)
    frame["conf_fp"] = (frame.pred_fp_primary.astype(float) - 0.5).abs()
    frame["conf_fp__pct"] = frame.conf_fp.rank(pct=True, method="average")
    return frame


REP_BASE = ["ad_knn__pct", "ad_density__pct", "conformal_fp__pct"]
REP_SHAPE = [f"shape_{s}__pct" for s in REP_STATS]


def rep_effect(frame: pd.DataFrame, with_conf: bool = False) -> float:
    meta, test = frame[frame.split == "meta"], frame[frame.split == "test"].reset_index(drop=True)
    err = test.error.to_numpy(float)
    base = REP_BASE + (["conf_fp__pct"] if with_conf else [])
    ref = score(meta, test, frame, base, "error")
    cand = score(meta, test, frame, base + REP_SHAPE, "error")
    return naurc(ref, err) - naurc(cand, err)


# ---------------------------------------------------------------- 실험 1
def experiment1() -> dict:
    rng = np.random.default_rng(SEED)
    classification = [d for d in sorted(p.name for p in (R2 / "evaluation").iterdir() if p.is_dir() and not p.name.startswith("_"))
                      if pd.read_csv(R2 / "evaluation" / d / "evaluation_signals.csv", usecols=["task_type"]).task_type.iloc[0] == "classification"]
    rows, boots = [], {}
    for dataset in classification:
        f = load_r2(dataset)
        meta, test = f[f.split == "meta"], f[f.split == "test"].reset_index(drop=True)
        err = test.abs_error_fp.to_numpy(float)
        conf = ["conf_fp__pct", "conf_cb__pct"]
        s = {n: score(meta, test, f, feats, "abs_error_fp") for n, feats in {
            "기준": BASE4, "기준+B": BASE4 + B_STD, "확신도 기준": BASE4 + conf, "확신도 기준+B": BASE4 + conf + B_STD}.items()}
        a = {n: naurc(v, err) for n, v in s.items()}
        groups = test.scaffold_group.to_numpy()
        boot = boot_independent(err, s["확신도 기준"], s["확신도 기준+B"], groups, rng)
        boot_orig = boot_independent(err, s["기준"], s["기준+B"], groups, rng)
        boots[dataset] = (boot, boot_orig)
        rows.append({"dataset": dataset, "cyp": dataset in CYP6, "n_test": len(test),
                     "effect_orig": a["기준"] - a["기준+B"], "effect_msp": a["확신도 기준"] - a["확신도 기준+B"],
                     "msp_gain": a["기준"] - a["확신도 기준"],
                     "ci_msp_low": ci(boot)[0], "ci_msp_high": ci(boot)[1]})
        print(f"  E1 {dataset:34s} orig {rows[-1]['effect_orig']:+.4f}  msp {rows[-1]['effect_msp']:+.4f}  [{rows[-1]['ci_msp_low']:+.3f},{rows[-1]['ci_msp_high']:+.3f}]", flush=True)
    table = pd.DataFrame(rows)

    def mean_ci(names, which=0):
        stack = np.column_stack([boots[n][which] for n in names])
        return float(np.nanmean(stack)), ci(np.nanmean(stack, axis=1))

    summary = {}
    for label, names in {"CYP 6종": CYP6, "Veith 3종": VEITH3, "기질 3종": SUBSTRATE,
                         "비CYP 분류 7종": tuple(d for d in classification if d not in CYP6)}.items():
        e_msp = table[table.dataset.isin(names)].effect_msp.mean()
        e_orig = table[table.dataset.isin(names)].effect_orig.mean()
        _, c_msp = mean_ci(names, 0)
        _, c_orig = mean_ci(names, 1)
        summary[label] = {"effect_orig": float(e_orig), "ci_orig": c_orig, "effect_msp": float(e_msp), "ci_msp": c_msp,
                          "msp_gain": float(table[table.dataset.isin(names)].msp_gain.mean())}
    table.to_csv(OUT / "실험1_물성별.csv", index=False)

    rep = {}
    rng2 = np.random.default_rng(SEED)
    for dataset in NEW:
        f = new_frame(dataset)
        meta, test = f[f.split == "meta"], f[f.split == "test"].reset_index(drop=True)
        err = test.error.to_numpy(float)
        base = REP_BASE + ["conf_fp__pct"]
        ref = score(meta, test, f, base, "error")
        cand = score(meta, test, f, base + REP_SHAPE, "error")
        boot = boot_independent(err, ref, cand, test.scaffold_group.to_numpy(), rng2)
        rep[dataset] = {"effect_orig": rep_effect(f), "effect_msp": naurc(ref, err) - naurc(cand, err), "ci_msp": ci(boot)}
        print(f"  E1 {dataset:34s} orig {rep[dataset]['effect_orig']:+.4f}  msp {rep[dataset]['effect_msp']:+.4f}  {rep[dataset]['ci_msp']}", flush=True)
    effects = [rep[d]["effect_msp"] for d in NEW]
    lows = [rep[d]["ci_msp"][0] for d in NEW]
    rep["verdict"] = ("소멸" if min(effects) <= 0 else ("유지" if max(lows) > 0 else "약화"))
    return {"summary": summary, "replication": rep, "n_classification": len(classification)}


# ---------------------------------------------------------------- 실험 2
def unit(dataset: str, frame: pd.DataFrame, direction: str) -> dict:
    """지문 전용 구성의 평가 단위. direction: meta→test(기본) 또는 test→meta(교차 적합)."""
    fit_split, eval_split = ("meta", "test") if direction == "meta->test" else ("test", "meta")
    new = dataset in NEW
    err_col = "error" if new else "abs_error_fp"
    base = REP_BASE if new else FP_BASE
    shape = REP_SHAPE if new else FP_SHAPE
    fit, evl = frame[frame.split == fit_split], frame[frame.split == eval_split].reset_index(drop=True)
    err = evl[err_col].to_numpy(float)
    return {"dataset": dataset, "direction": direction, "err": err,
            "ref": score(fit, evl, frame, base, err_col), "cand": score(fit, evl, frame, base + shape, err_col),
            "groups": evl.scaffold_group.astype(str).to_numpy()}


def effect_of(u: dict) -> float:
    return naurc(u["ref"], u["err"]) - naurc(u["cand"], u["err"])


def boot_joint(units: list[dict], rng, n_boot=N_BOOT) -> np.ndarray:
    universe = pd.unique(np.concatenate([u["groups"] for u in units]))
    position = {g: i for i, g in enumerate(universe)}
    codes = [np.array([position[g] for g in u["groups"]]) for u in units]
    out = np.empty((n_boot, len(units)))
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, len(universe), len(universe)), minlength=len(universe))
        for j, (u, c) in enumerate(zip(units, codes)):
            rows = np.repeat(np.arange(len(c)), counts[c])
            out[i, j] = naurc(u["ref"][rows], u["err"][rows]) - naurc(u["cand"][rows], u["err"][rows]) if len(rows) else np.nan
    return out


def experiment2() -> dict:
    rng = np.random.default_rng(SEED)
    veith = []
    for dataset in VEITH3:
        veith.append(unit(dataset, load_r2(dataset), "meta->test"))
    for dataset in NEW:
        veith.append(unit(dataset, new_frame(dataset), "meta->test"))
    point = {u["dataset"]: effect_of(u) for u in veith}
    # (i) 물성별 독립 군집 부트스트랩
    indep = np.column_stack([boot_independent(u["err"], u["ref"], u["cand"], u["groups"], rng) for u in veith])
    # (ii) 5종의 골격 합집합 합동 군집 부트스트랩
    joint = boot_joint(veith, rng)
    result = {"veith": {"point": point, "mean": float(np.mean(list(point.values()))),
                        "ci_independent": ci(np.nanmean(indep, axis=1)), "ci_joint": ci(np.nanmean(joint, axis=1)),
                        "ci_each_independent": {u["dataset"]: ci(indep[:, j]) for j, u in enumerate(veith)}}}

    sub_orig, sub_swap = [], []
    for dataset in SUBSTRATE:
        f = load_r2(dataset)
        sub_orig.append(unit(dataset, f, "meta->test"))
        sub_swap.append(unit(dataset, f, "test->meta"))
    point_o = {u["dataset"]: effect_of(u) for u in sub_orig}
    point_s = {u["dataset"]: effect_of(u) for u in sub_swap}
    units = sub_orig + sub_swap
    joint_s = boot_joint(units, rng)
    # 물성별로 두 방향을 평균한 뒤 3종 평균
    per_dataset = (joint_s[:, :3] + joint_s[:, 3:]) / 2
    pooled_point = float(np.mean([(point_o[d] + point_s[d]) / 2 for d in SUBSTRATE]))
    rng_o = np.random.default_rng(SEED)
    indep_o = np.column_stack([boot_independent(u["err"], u["ref"], u["cand"], u["groups"], rng_o) for u in sub_orig])
    result["substrate"] = {"point_meta_to_test": point_o, "point_test_to_meta": point_s,
                           "mean_meta_to_test": float(np.mean(list(point_o.values()))),
                           "ci_meta_to_test_independent": ci(np.nanmean(indep_o, axis=1)),
                           "mean_crossfit": pooled_point, "ci_crossfit_joint": ci(np.nanmean(per_dataset, axis=1))}
    return result


def main() -> int:
    chk = checks()
    (OUT / "재현점검.json").write_text(json.dumps(chk, ensure_ascii=False, indent=2), encoding="utf-8")
    print("재현 점검:", json.dumps(chk, ensure_ascii=False))
    e1 = experiment1()
    e2 = experiment2()
    out = {"checks": chk, "experiment1": e1, "experiment2": e2}

    # 판정(재현 점검을 통과했을 때만)
    v = {}
    if chk["all_pass"]:
        cyp = e1["summary"]["CYP 6종"]
        lo, point = cyp["ci_msp"][0], cyp["effect_msp"]
        v["실험1"] = "소멸" if lo <= 0 else ("유지" if point >= 0.040 else "약화")
        v["실험1_재현데이터"] = e1["replication"]["verdict"]
        v["실험2a"] = "라이브러리 안에서 양수" if e2["veith"]["ci_joint"][0] > 0 else "불충분"
        v["실험2b"] = "다른 화합물 라이브러리에서도 양수" if e2["substrate"]["ci_crossfit_joint"][0] > 0 else "독립 계열 근거 불충분"
    else:
        v["note"] = "재현 점검 실패. 판정을 적용하지 않는다."
    out["verdict"] = v
    (OUT / "결과.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

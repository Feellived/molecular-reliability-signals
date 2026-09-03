#!/usr/bin/env python
"""분자 하나를 채점하고 판정 문장을 만든다.

engine.py의 부품을 순서대로 엮는다. 반환값은 계획서 7.3절이 정한 형태를
따른다. 세 신뢰성 축을 분리해 담고 단일 점수로 합산하지 않으며, 종합은
숫자가 아니라 문장으로 낸다.

  예측       예측값과 컨포멀 구간
  축         표현 안정성, 입력 상태 민감성, 화학 공간 위치를 각각
  조건       이 물성에서 어느 축을 쓸 수 있고 왜 그런지
  판정       위 값들을 근거로 만든 한 문장

위험 점수도 함께 담지만 화면의 주역이 아니다. 축별 값을 감추고 점수만
보여주면 계획서가 금지한 단일 점수 표시가 된다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from engine import (
    AD_NEIGHBORS, AD_SIMILARITY_CUTOFF, AXIS_KEYS, AxisResult,
    fingerprints, gen_a_axis, gen_protonation, gen_stereo, gen_tautomers,
    load_bundle, predict_chemberta, predict_fingerprint, tanimoto,
)

HIGH, MODERATE = 0.85, 0.65   # 백분위 경계. 상위 15퍼센트를 높음으로 본다


def _has_final_consonant(word: str) -> bool:
    """한글 음절의 받침 유무. 와/과, 은/는 같은 조사 선택에 쓴다."""
    last = word.strip()[-1:]
    if not last or not ("가" <= last <= "힣"):
        return True
    return (ord(last) - 0xAC00) % 28 != 0


def _join_korean(names: list[str]) -> str:
    """이름을 와/과로 잇는다. 마지막 항목 뒤에는 조사를 붙이지 않는다."""
    if not names:
        return ""
    joined = names[0]
    for name in names[1:]:
        joined += ("과 " if _has_final_consonant(joined) else "와 ") + name
    return joined


def _axis_stats(parent: float, values: np.ndarray, spread: float) -> dict:
    """축 하나의 흩어짐. 원본을 표본에 포함한다. 변형이 하나뿐인 축 때문이다."""
    if len(values) == 0:
        return {"dispersion": 0.0, "max_deviation": 0.0, "relative": 0.0}
    sample = np.append(values, parent)
    dispersion = float(np.std(sample, ddof=0))
    return {"dispersion": dispersion,
            "max_deviation": float(np.abs(values - parent).max()),
            "relative": dispersion / spread if spread > 1e-9 else 0.0}


def _verdict(prediction, interval, prediction_set, axes, ad_percentile, task) -> dict:
    """신호를 근거로 판정 문장을 만든다. 계획서 7.3절의 예시 문장 형식을 따른다."""
    hot = [a for a in axes if a.usable and a.percentile is not None and a.percentile >= HIGH]
    warm = [a for a in axes if a.usable and a.percentile is not None
            and MODERATE <= a.percentile < HIGH]
    unusable = [a for a in axes if not a.usable]

    # 좁은 예측이란 회귀는 구간이 좁은 것, 분류는 예측 집합에 라벨이 하나뿐인 것이다.
    if task == "classification":
        narrow = prediction_set is not None and prediction_set["size"] == 1
        narrow_phrase = "예측 집합이 라벨 하나로 좁게 산출되었으나"
    else:
        narrow = (interval is not None and interval.get("relative_width") is not None
                  and interval["relative_width"] < 0.5)
        narrow_phrase = "예측 구간은 좁게 산출되었으나"

    if hot:
        names = _join_korean([a.name for a in hot])
        head = (f"{names}에 따라 예측값이 크게 변동한다. "
                "입력 표기를 확인하기 전에는 이 예측을 사용하지 않는 것이 바람직하다.")
        level = "주의"
    elif warm:
        head = (f"{warm[0].name}에 따라 예측값이 다소 변동한다. "
                "같은 물질의 다른 등록 형태를 함께 확인할 것을 권한다.")
        level = "보통"
    else:
        head = "입력 표기와 화학적 상태를 바꾸어도 예측이 안정적이다."
        level = "안정"

    notes = []
    if narrow and hot:
        notes.append(f"{narrow_phrase} 위 변동이 거기에 반영되어 있지 않다.")
    if ad_percentile is not None and ad_percentile >= HIGH:
        notes.append("학습 데이터에서 먼 골격이므로 예측 자체의 신뢰도도 낮다.")
    if unusable:
        names = ", ".join(a.name for a in unusable)
        notes.append(f"{names} 축은 이 물성의 측정 조건에서 의미를 갖지 않아 판정에서 제외했다.")
    return {"level": level, "headline": head, "notes": notes}


def score(bundle_root: Path, dataset: str, smiles: str) -> dict:
    bundle = load_bundle(str(Path(bundle_root).resolve()), dataset)
    mol = Chem.MolFromSmiles(smiles) if smiles and smiles.strip() else None
    if mol is None or mol.GetNumAtoms() == 0:
        # 빈 문자열은 RDKit이 원자 0개인 분자로 받아들여 통과시킨다.
        raise ValueError(f"SMILES를 해석할 수 없다: {smiles!r}")
    parent = Chem.MolToSmiles(mol)

    # 이 물성에 조건이 성립하는 축만 쓴다
    allowed = bundle.allowed_axes()
    plan, results = {}, []
    for name, info in allowed.items():
        key = AXIS_KEYS.get(name)
        if key is None:
            continue
        if not info["사용"]:
            reason = ("이 물성의 측정 정의와 충돌한다" if info["물성_허용성"] != "허용"
                      else f"변형 생성 판정이 {info['표본_판정']}이다")
            results.append(AxisResult(name, False, reason))
            continue
        if key == "B1_tautomer":
            variants = gen_tautomers(mol, parent)
        elif key == "B1_protonation":
            variants = gen_protonation(smiles, parent)
        else:
            variants = gen_stereo(mol, parent)
        plan[name] = variants
        results.append(AxisResult(name, True, "조건 성립", n_variants=len(variants)))

    representation = gen_a_axis(mol, parent)

    # 원본과 모든 변형을 한 번에 채점한다
    b_variants = [s for group in plan.values() for s in group]
    every = [parent] + representation + b_variants
    fp_seeds = predict_fingerprint(bundle, every)
    fp_pred = fp_seeds.mean(axis=0)
    checkpoint = Path(bundle.manifest.get("chemberta_root", "")) if False else None
    cb_pred = None
    cb_root = Path(__file__).resolve().parents[3] / "Jiye" / "checkpoints" / "chemberta"
    if (cb_root / dataset / "augmented").exists():
        cb_pred = predict_chemberta(cb_root / dataset / "augmented", every)

    spread_fp = float(np.std(fp_pred)) or 1.0
    offset = 1 + len(representation)
    parent_fp = float(fp_pred[0])

    # 표현 불안정성 A는 문자열을 직접 읽는 모델에서만 0이 아니다
    a_axis = AxisResult("표현 불안정성", cb_pred is not None,
                        "언어 모델 확보" if cb_pred is not None else "언어 모델 체크포인트 없음",
                        n_variants=len(representation))
    if cb_pred is not None and representation:
        spread_cb = float(np.std(cb_pred)) or 1.0
        stats = _axis_stats(float(cb_pred[0]), cb_pred[1:offset], spread_cb)
        a_axis.dispersion = stats["dispersion"]
        a_axis.max_deviation = stats["max_deviation"]
        a_axis.percentile = bundle.percentile("axis__cb_augmented__A", stats["dispersion"])

    # 입력 상태 민감성 B는 축별로 산출하고 조건이 성립하는 축을 합쳐 통합한다
    cursor, pooled = offset, []
    for result in results:
        if not result.usable:
            continue
        group = plan[result.name]
        chunk = fp_pred[cursor:cursor + len(group)]
        cursor += len(group)
        stats = _axis_stats(parent_fp, chunk, spread_fp)
        result.dispersion = stats["dispersion"]
        result.max_deviation = stats["max_deviation"]
        result.examples = [{"smiles": s, "prediction": round(float(v), 4)}
                           for s, v in zip(group[:3], chunk[:3])]
        pooled.extend(chunk.tolist())
    pooled_stats = _axis_stats(parent_fp, np.array(pooled), spread_fp)
    cb_pooled = None
    if cb_pred is not None and b_variants:
        cb_spread = float(np.std(cb_pred)) or 1.0
        cb_pooled = _axis_stats(float(cb_pred[0]), cb_pred[offset:], cb_spread)["dispersion"]
    b_percentile = bundle.percentile("cond_B__fp_primary__std", pooled_stats["dispersion"])
    for result in results:
        if result.usable and result.dispersion is not None:
            result.percentile = bundle.percentile("cond_B__fp_primary__std", result.dispersion)

    # 화학 공간 위치
    query = fingerprints([parent])[0]
    similarity = tanimoto(query, bundle.neighbor_fp)
    top = float(np.sort(similarity)[-AD_NEIGHBORS:].mean())
    density = int((similarity >= AD_SIMILARITY_CUTOFF).sum())
    ad_percentile = bundle.percentile("base__ad_knn", -top)

    # 컨포멀 구간
    interval = None
    if bundle.conformal.get("qhat") is not None:
        scale = float(fp_seeds.std(axis=0)[0]) + bundle.conformal["scale_floor"]
        half = bundle.conformal["qhat"] * scale
        interval = {"lower": round(parent_fp - half, 4), "upper": round(parent_fp + half, 4),
                    "width": round(2 * half, 4),
                    "relative_width": round(2 * half / spread_fp, 4) if spread_fp else None,
                    "coverage": 1 - bundle.conformal["alpha"]}

    prediction_set = None
    if bundle.task_type == "classification" and bundle.conformal.get("aps_qhat") is not None:
        labels = _aps_set(bundle.conformal["aps_qhat"], parent_fp,
                          bundle.conformal.get("randomization_tag", "mist-fp-aps-v1"), parent)
        prediction_set = {"labels": labels, "size": len(labels),
                          "coverage": 1 - bundle.conformal["alpha"],
                          "note": "라벨이 둘이면 모델이 어느 쪽인지 가르지 못한 것이다"}

    conformal_signals = _conformal_signals(
        bundle, parent_fp, float(cb_pred[0]) if cb_pred is not None else None,
        interval, float(fp_seeds.std(axis=0)[0]) + bundle.conformal.get("scale_floor", 0.0),
        parent)

    verdict = _verdict(parent_fp, interval, prediction_set, [a_axis] + results,
                       ad_percentile, bundle.task_type)
    return {
        "dataset": dataset, "input_smiles": smiles, "canonical_smiles": parent,
        "task_type": bundle.task_type,
        "prediction": round(parent_fp, 4),
        "interval": interval,
        "prediction_set": prediction_set,
        "reliability_axes": {
            "표현 안정성": _dump(a_axis),
            "입력 상태 민감성": {"percentile": b_percentile,
                          "dispersion": pooled_stats["dispersion"],
                          "axes": [_dump(r) for r in results]},
            "화학 공간 위치": {"nearest5_tanimoto": round(top, 4),
                         "neighbors_over_0.40": density,
                         "percentile": ad_percentile},
        },
        "combined_risk": _combined(bundle, top, density, fp_pred, cb_pred,
                                   pooled_stats, a_axis, cb_pooled, conformal_signals),
        "verdict": verdict,
        "settings_digest": bundle.manifest["digest"],
    }


def _dump(result: AxisResult) -> dict:
    return {"name": result.name, "usable": result.usable, "reason": result.reason,
            "n_variants": result.n_variants, "dispersion": result.dispersion,
            "max_deviation": result.max_deviation, "percentile": result.percentile,
            "examples": result.examples}


def _aps_set(qhat: float, prediction: float, tag: str, smiles: str) -> list[int]:
    """무작위화 APS 예측 집합. 무작위화는 SMILES 해시로 결정한다."""
    probability = float(np.clip(prediction, 0.0, 1.0))
    probs = np.array([1.0 - probability, probability])
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    noise = int(hashlib.sha256(f"{tag}:{smiles}".encode()).hexdigest()[:16], 16) / 2**64
    return [label for label in (0, 1)
            if cumulative[int(np.where(order == label)[0][0])] - noise * probs[label] <= qhat]


def _aps_set_size(qhat: float, prediction: float, tag: str, smiles: str) -> float:
    return float(len(_aps_set(qhat, prediction, tag, smiles)))


def _conformal_signals(bundle, fp_prediction, cb_prediction, interval,
                       fp_scale, smiles) -> dict:
    """기준선의 컨포멀 신호 두 종. 회귀는 구간 폭, 분류는 예측 집합 크기다."""
    tag = bundle.conformal.get("randomization_tag", "mist-fp-aps-v1")
    cb_params = bundle.conformal.get("chemberta", {})
    out = {"base__conformal_fp": None, "base__conformal_cb": None}
    if bundle.task_type == "classification":
        if bundle.conformal.get("aps_qhat") is not None:
            out["base__conformal_fp"] = _aps_set_size(
                bundle.conformal["aps_qhat"], fp_prediction, tag, smiles)
        if cb_params.get("aps_qhat") is not None and cb_prediction is not None:
            out["base__conformal_cb"] = _aps_set_size(
                cb_params["aps_qhat"], cb_prediction, tag, smiles)
    else:
        out["base__conformal_fp"] = interval["width"] if interval else None
        # 회귀의 ChemBERTa 컨포멀은 재현하지 않는다. 담당2의 척도 함수가
        # 우리가 가진 재료와 무관해(지문 시드 표준편차와 상관 0.05) 근사하면
        # 틀린 값을 넣게 된다. 결합 규칙도 이 신호를 뺀 판으로 적합했다.
    return out


def _combined(bundle, top, density, fp_pred, cb_pred, pooled, a_axis,
              cb_pooled, conformal_signals) -> dict | None:
    """결합 규칙. 축별 값을 감추지 않으므로 참고용으로만 담는다."""
    raw = {"base__ad_knn": -top, "base__ad_density": -density,
           "base__disagreement": abs(float(fp_pred[0]) - float(cb_pred[0]))
           if cb_pred is not None else 0.0,
           "cond_B__fp_primary__std": pooled["dispersion"],
           "cond_B__cb_augmented__std": cb_pooled,
           "axis__cb_augmented__A": a_axis.dispersion or 0.0}
    raw.update(conformal_signals)
    combiner = bundle.combiner_single
    values = []
    for column in combiner["features"]:
        value = raw.get(column)
        if value is None:
            return None
        percentile = bundle.percentile(column, value)
        values.append(0.5 if percentile is None else percentile)
    score = float(np.dot(combiner["coefficients"], values)) + combiner["intercept"]
    return {"score": round(score, 4), "features": combiner["features"],
            "percentiles": [round(v, 4) for v in values],
            "test_auprc": combiner["test_auprc"],
            "note": "참고값. 축별 값을 대체하지 않는다"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="분자 하나 채점")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("smiles")
    args = parser.parse_args()
    print(json.dumps(score(args.bundle, args.dataset, args.smiles),
                     ensure_ascii=False, indent=2))

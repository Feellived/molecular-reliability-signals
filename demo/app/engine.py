#!/usr/bin/env python
"""분자 하나를 받아 신뢰성 신호와 판정 문장을 만든다 (연구계획서 7.3절).

파이프라인은 데이터셋 단위로 돌아가므로 분자 하나에는 쓸 수 없다. 이 모듈은
build_demo_bundle.py가 내보낸 정적 산출물만 읽어 같은 절차를 분자 하나에
적용한다. 파이프라인 코드에 의존하지 않으므로 배포 환경에 그대로 옮길 수 있다.

  1  설정 지문 대조. 산출물을 만든 변형 생성 설정과 지금 코드의 설정이
     다르면 백분위 기준이 낡은 것이므로 실행을 거부한다. 조용히 틀린
     백분위를 내놓는 것이 가장 위험한 실패 방식이다
  2  이 물성에 조건이 성립하는 축만 골라 변형 생성
  3  지문 모델과 언어 모델로 원본과 변형을 채점
  4  축별 흩어짐 산출, 보정 분포로 백분위 변환
  5  결합 규칙으로 위험 점수, 컨포멀로 예측 구간
  6  판정 문장 생성

계획서 7.3절은 세 축을 분리해 표시하고 단일 점수로 합산하지 말 것을 정한다.
따라서 반환값은 축별 값을 그대로 담고, 종합은 숫자가 아니라 문장으로 낸다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

MAX_TAUTOMERS = 20
A_AXIS_K = 10
PH_MIN, PH_MAX = 6.4, 8.4
MORGAN_RADIUS, MORGAN_BITS, MORGAN_CHIRALITY = 2, 2048, True
MAX_LENGTH = 128
AD_SIMILARITY_CUTOFF = 0.40
AD_NEIGHBORS = 5

AXIS_KEYS = {"호변이성질체": "B1_tautomer", "양성자화": "B1_protonation",
             "입체 표기": "B3_stereo"}


def settings_digest() -> str:
    """산출물의 지문과 대조할 값. build_demo_bundle.py와 같은 규칙으로 만든다."""
    settings = {"A_AXIS_K": A_AXIS_K, "MAX_TAUTOMERS": MAX_TAUTOMERS,
                "PH_MIN": PH_MIN, "PH_MAX": PH_MAX, "B3_MODE": "full_strip",
                "MORGAN_RADIUS": MORGAN_RADIUS, "MORGAN_BITS": MORGAN_BITS,
                "MORGAN_CHIRALITY": MORGAN_CHIRALITY}
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def _morgan_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, includeChirality=MORGAN_CHIRALITY)


def fingerprints(smiles_list) -> np.ndarray:
    generator = _morgan_generator()
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        rows.append(np.zeros(MORGAN_BITS, dtype=np.uint8) if mol is None
                    else np.frombuffer(
                        generator.GetFingerprintAsNumPy(mol).astype(np.uint8).tobytes(),
                        dtype=np.uint8))
    return np.vstack(rows) if rows else np.zeros((0, MORGAN_BITS), dtype=np.uint8)


def depict(smiles: str, width: int = 260, height: int = 170,
           reference: str | None = None) -> str | None:
    """분자 구조를 SVG로 그린다. 화학 도구인데 문자열만 보이면 읽을 수가 없다.

    reference를 주면 그 분자와 다른 원자를 표시한다. 호변이성질체처럼 거의
    같아 보이는 변형에서 어디가 달라졌는지 눈으로 짚어주기 위해서다.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    highlight = []
    if reference:
        base = Chem.MolFromSmiles(reference)
        if base is not None:
            match = mol.GetSubstructMatch(base)
            matched = set(match)
            highlight = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in matched]
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1.5
    options.multipleBondOffset = 0.16
    options.padding = 0.06
    options.setHighlightColour((0.98, 0.88, 0.80, 1.0))  # 원본과 달라진 원자. 옅은 호박색
    # 기본 원소 색은 원색에 가까워 어수선하다. 채도를 낮춰 정갈하게 만든다.
    # 탄소는 검정으로 두었다가 아래에서 currentColor로 바꿔 어두운 모드를 따르게 한다.
    options.updateAtomPalette({
        6: (0, 0, 0), 7: (0.20, 0.44, 0.56), 8: (0.66, 0.33, 0.17),
        9: (0.30, 0.50, 0.35), 15: (0.72, 0.45, 0.15), 16: (0.62, 0.50, 0.12),
        17: (0.30, 0.50, 0.35), 35: (0.55, 0.35, 0.20), 53: (0.45, 0.30, 0.50),
    })
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight or None)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # 탄소 골격만 CSS가 물려받게 한다. 밝은 배경에서는 잉크색, 어두운 배경에서는
    # 밝은 색으로 자동으로 바뀐다. 헤테로 원자 색은 양쪽에서 다 읽히도록 골랐다.
    for black in ("#000000", "rgb(0,0,0)", "#000"):
        svg = svg.replace(black, "currentColor")
    return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").strip()


def tanimoto(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """질의 하나와 행렬 전체의 Tanimoto 유사도."""
    intersection = matrix @ query
    return intersection / (matrix.sum(axis=1) + query.sum() - intersection + 1e-12)


# --- 변형 생성. generate_variants.py와 같은 규칙을 분자 하나에 적용한다 ---

def gen_a_axis(mol, parent_canonical: str) -> list[str]:
    """등가 SMILES. 정준형 왕복 검사로 다른 입체이성질체가 섞이는 것을 막는다."""
    drawn = Chem.MolToRandomSmilesVect(mol, A_AXIS_K * 3, randomSeed=0xC0FFEE)
    seen, out = set(), []
    for smiles in drawn:
        if smiles in seen:
            continue
        roundtrip = Chem.MolFromSmiles(smiles)
        if roundtrip is None or Chem.MolToSmiles(roundtrip) != parent_canonical:
            continue
        seen.add(smiles); out.append(smiles)
        if len(out) >= A_AXIS_K:
            break
    return out


def gen_tautomers(mol, parent_canonical: str) -> list[str]:
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(MAX_TAUTOMERS)
    out = []
    for candidate in enumerator.Enumerate(mol):
        smiles = Chem.MolToSmiles(candidate)
        if smiles != parent_canonical and smiles not in out:
            out.append(smiles)
    return out


def gen_protonation(smiles: str, parent_canonical: str) -> list[str]:
    try:
        from dimorphite_dl import protonate_smiles
    except ImportError:
        return []
    out = []
    for candidate in protonate_smiles(smiles, ph_min=PH_MIN, ph_max=PH_MAX,
                                      precision=1.0, max_variants=8):
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical != parent_canonical and canonical not in out:
            out.append(canonical)
    return out


def gen_stereo(mol, parent_canonical: str) -> list[str]:
    stripped = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stripped)
    smiles = Chem.MolToSmiles(stripped)
    return [smiles] if smiles != parent_canonical else []


@dataclass
class AxisResult:
    name: str
    usable: bool
    reason: str
    n_variants: int = 0
    dispersion: float | None = None
    percentile: float | None = None
    max_deviation: float | None = None
    examples: list[dict] = field(default_factory=list)
    # 변형 전체의 예측. 제외된 축에서도 비어 있는 채로 존재해야 한다.
    spread: list[dict] = field(default_factory=list)


class Bundle:
    """build_demo_bundle.py 산출물을 읽는다."""

    def __init__(self, root: Path, dataset: str):
        self.root, self.dataset = Path(root), dataset
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        expected = self.manifest["digest"]
        if expected != settings_digest():
            raise RuntimeError(
                f"설정 지문 불일치. 산출물 {expected}, 코드 {settings_digest()}. "
                "변형 생성 설정이 바뀌었으므로 백분위 기준이 낡았다. "
                "build_demo_bundle.py를 다시 실행해 산출물을 갱신해야 한다.")
        base = self.root / dataset
        read = lambda name: json.loads((base / name).read_text(encoding="utf-8"))
        self.reference = read("reference.json")
        self.combiner = read("combiner.json")
        self.combiner_single = read("combiner_single.json")
        # 판정 등급의 경계와 각 구간의 실제 미달률. 없으면 기본값으로 떨어진다.
        calibration = self.root / "verdict_calibration.json"
        self.calibration = (json.loads(calibration.read_text(encoding="utf-8"))
                            if calibration.exists() else None)
        self.conformal = read("conformal.json")
        self.axes = read("axes.json")
        self.models = [joblib.load(p) for p in sorted((base / "models").glob("*.joblib"))]
        if not self.models:
            raise RuntimeError(
                f"{dataset}의 지문 모델 캐시가 없다. "
                "build_demo_bundle.py를 --with-models로 실행해야 한다.")
        import pandas as pd
        neighbors = pd.read_csv(base / "neighbors.csv")
        self.neighbor_fp = fingerprints(neighbors["parent_smiles"])
        self.task_type = json.loads(
            (base / "models.json").read_text(encoding="utf-8"))["task_type"]

    def percentile(self, column: str, value: float) -> float | None:
        entry = self.reference.get(column)
        if entry is None:
            return None
        values = entry["sorted_values"]
        return float(np.searchsorted(values, value, side="right") / len(values))

    def allowed_axes(self) -> dict[str, dict]:
        return {k: v for k, v in self.axes.items() if not k.startswith("_")}


@lru_cache(maxsize=8)
def load_bundle(root: str, dataset: str) -> Bundle:
    """물성별 산출물을 캐시한다. 매 호출마다 모델을 다시 읽으면 8초가 걸린다."""
    return Bundle(Path(root), dataset)


def predict_fingerprint(bundle: Bundle, smiles_list) -> np.ndarray:
    """시드별 예측의 평균. 시드 간 표준편차는 컨포멀 척도로 쓴다."""
    matrix = fingerprints(smiles_list)
    per_seed = []
    for model in bundle.models:
        if bundle.task_type == "classification":
            per_seed.append(model.predict_proba(matrix)[:, 1])
        else:
            per_seed.append(model.predict(matrix))
    return np.vstack(per_seed)


def predict_chemberta(checkpoint: Path, smiles_list) -> np.ndarray | None:
    """언어 모델 예측. 체크포인트가 없으면 A축을 낼 수 없다."""
    if not (checkpoint / "complete.json").exists():
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).eval()
    encoded = tokenizer(list(smiles_list), truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    if logits.shape[-1] == 1:
        scaled = logits.squeeze(-1).numpy()
        return scaled * float(meta.get("target_std", 1.0)) + float(meta.get("target_mean", 0.0))
    return torch.softmax(logits, dim=-1)[:, 1].numpy()

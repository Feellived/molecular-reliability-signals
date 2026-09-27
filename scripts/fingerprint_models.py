"""Morgan 지문과 기본 트리 모델의 공통 기능을 제공한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


@dataclass(frozen=True)
class MorganConfig:
    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True


def make_morgan_matrix(
    smiles: Iterable[str], config: MorganConfig = MorganConfig()
) -> np.ndarray:
    """SMILES를 고정 길이 이진 Morgan 지문으로 변환한다."""
    values = list(smiles)
    matrix = np.zeros((len(values), config.n_bits), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.radius,
        fpSize=config.n_bits,
        includeChirality=config.include_chirality,
    )
    for index, value in enumerate(values):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {index}: {value!r}")
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix


def make_model(model_name: str, task_type: str, seed: int):
    """고정 설정으로 재현 가능한 기준 모델을 만든다."""
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task_type: {task_type}")
    if model_name == "rf":
        common = dict(
            n_estimators=500,
            random_state=seed,
            # Windows의 중첩 프로세스 정체를 피하기 위해 단일 작업자를 쓴다.
            n_jobs=1,
            max_features="sqrt",
            min_samples_leaf=1,
        )
        if task_type == "classification":
            return RandomForestClassifier(class_weight="balanced", **common)
        return RandomForestRegressor(**common)
    if model_name == "xgb":
        common = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        if task_type == "classification":
            return XGBClassifier(eval_metric="logloss", **common)
        return XGBRegressor(objective="reg:squarederror", **common)
    raise ValueError(f"Unsupported model: {model_name}")


def predict_values(model, matrix: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        return model.predict_proba(matrix)[:, 1]
    return model.predict(matrix)


def save_fingerprint_cache(
    path, row_uids: pd.Series, matrix: np.ndarray, config: MorganConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_uid=np.asarray(row_uids.astype(str).tolist(), dtype=str),
        fingerprints=matrix,
        radius=np.array(config.radius),
        n_bits=np.array(config.n_bits),
        include_chirality=np.array(config.include_chirality),
    )

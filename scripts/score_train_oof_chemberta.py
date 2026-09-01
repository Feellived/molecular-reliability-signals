#!/usr/bin/env python
"""train 분할에 폴드 외 ChemBERTa 예측을 붙인다 (연구계획서 5.7절).

담당2의 체크포인트는 train 전체로 미세조정되었으므로 train 분자에 대한 예측은
모델이 외운 값이다. 결합 규칙 학습에 train을 쓰려면 각 분자를 그 분자가
빠진 모델로 예측해야 한다. cv_fold를 다섯 조각으로 나눠 조각마다 따로
미세조정한다.

학습 설정은 담당2의 chemberta_training_executed.py와 동일하다.

  모델        DeepChem/ChemBERTa-10M-MTR
  epoch 3, batch 32, lr 2e-5, max_length 128, warmup 10퍼센트
  증강 버전   분자당 등가 SMILES 3개
  회귀        학습 폴드의 라벨 평균과 표준편차로 표준화하고 예측을 되돌린다

주의. 담당2는 CUDA에서 혼합 정밀도로 학습했고 이 스크립트는 MPS나 CPU에서
단정밀도로 학습한다. 따라서 같은 설정이라도 결과가 완전히 같지는 않다.
meta·test 예측은 담당2 체크포인트를 그대로 써서 정확히 일치하지만 train
폴드 외 예측은 새로 학습한 모델에서 나오므로, 두 집합의 신호 척도가 미세하게
어긋날 수 있다. 이 차이는 산출 후 분포 비교로 확인한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_repairs import apply_known_repairs  # noqa: E402

MODEL_NAME = "DeepChem/ChemBERTa-10M-MTR"
EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
MAX_LENGTH = 128
AUG_PER_MOLECULE = 3
SEED = 42
VERSIONS = ("regular", "augmented")


class SmilesDataset(Dataset):
    def __init__(self, smiles, labels=None):
        self.smiles = [str(s) for s in smiles]
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        item = {"smiles": self.smiles[i]}
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item


def random_smiles_variants(smiles, n, seed):
    """담당2의 증강 함수와 동일하게 동작한다."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [str(smiles)] * n
    variants, seen = [], set()
    for i in range(max(20, n * 10)):
        candidate = Chem.MolToSmiles(
            mol, canonical=False, doRandom=True, isomericSmiles=True
        )
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
        if len(variants) >= n:
            break
    while len(variants) < n:
        variants.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return variants[:n]


def make_collate(tokenizer, task, with_labels):
    def collate(batch):
        encoded = tokenizer(
            [b["smiles"] for b in batch], padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        if with_labels:
            dtype = torch.long if task == "classification" else torch.float32
            encoded["labels"] = torch.tensor([b["label"] for b in batch], dtype=dtype)
        return encoded
    return collate


def train_fold(smiles, labels, task, tokenizer, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2 if task == "classification" else 1,
        problem_type=(
            "single_label_classification" if task == "classification" else "regression"
        ),
        ignore_mismatched_sizes=True,
    ).to(device)
    loader = DataLoader(
        SmilesDataset(smiles, labels), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=make_collate(tokenizer, task, True),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total = EPOCHS * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total * 0.1)), total
    )
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model


@torch.inference_mode()
def predict(model, smiles, task, tokenizer, device, mean, std, batch_size=256):
    model.eval()
    loader = DataLoader(
        SmilesDataset(smiles), batch_size=batch_size, shuffle=False,
        collate_fn=make_collate(tokenizer, task, False),
    )
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        if task == "classification":
            out.extend(torch.softmax(logits, -1)[:, 1].float().cpu().tolist())
        else:
            out.extend((logits.squeeze(-1).float().cpu() * std + mean).tolist())
    return out


def process_dataset(dataset, processed_dir, variants_dir, out_dir, device):
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    splits = pd.read_csv(processed_dir / dataset / "splits.csv", low_memory=False)
    splits, _ = apply_known_repairs(splits)
    variants = pd.read_csv(variants_dir / dataset / "variants.csv")
    task = splits["task_type"].iloc[0]

    is_train = splits["split"].eq("train").to_numpy()
    fold = splits["cv_fold"].to_numpy()
    variant_fold = variants["cv_fold"].to_numpy()

    origin = splits.loc[is_train, ["row_uid", "dataset", "split", "cv_fold"]].copy()
    variant_out = variants[
        ["variant_uid", "parent_row_uid", "dataset", "axis", "split", "cv_fold"]
    ].copy()
    for version in VERSIONS:
        origin[f"pred_chemberta_{version}"] = np.nan
        variant_out[f"pred_chemberta_{version}"] = np.nan

    torch.manual_seed(SEED)
    for k in sorted({int(f) for f in fold if f >= 0}):
        fit_rows = is_train & (fold != k)
        origin_rows = is_train & (fold == k)
        variant_rows = variant_fold == k
        if not origin_rows.any():
            continue
        fit = splits.loc[fit_rows]
        mean, std = 0.0, 1.0
        if task == "regression":
            mean = float(pd.to_numeric(fit["Y_final"]).mean())
            std = float(pd.to_numeric(fit["Y_final"]).std(ddof=0)) or 1.0

        for version in VERSIONS:
            if version == "regular":
                smiles = fit["parent_smiles"].astype(str).tolist()
                labels = pd.to_numeric(fit["Y_final"]).tolist()
            else:
                smiles, labels = [], []
                for i, (_, row) in enumerate(fit.reset_index(drop=True).iterrows()):
                    for variant in random_smiles_variants(
                        row["parent_smiles"], AUG_PER_MOLECULE, SEED + i * 17
                    ):
                        smiles.append(variant)
                        labels.append(float(row["Y_final"]))
            if task == "classification":
                labels = [int(y) for y in labels]
            else:
                labels = [(float(y) - mean) / std for y in labels]

            model = train_fold(smiles, labels, task, tokenizer, device)
            origin.loc[origin["cv_fold"].to_numpy() == k, f"pred_chemberta_{version}"] = (
                predict(model, splits.loc[origin_rows, "parent_smiles"], task,
                        tokenizer, device, mean, std)
            )
            if variant_rows.any():
                variant_out.loc[variant_rows, f"pred_chemberta_{version}"] = predict(
                    model, variants.loc[variant_rows, "variant_smiles"], task,
                    tokenizer, device, mean, std,
                )
            del model
        print(f"    fold {k} 완료", flush=True)

    dataset_out = out_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    origin.to_csv(dataset_out / "origin_predictions_chemberta_oof.csv", index=False)
    variant_out.to_csv(dataset_out / "variant_predictions_chemberta_oof.csv", index=False)
    return {
        "dataset": dataset, "task_type": task, "n_origin": len(origin),
        "n_variant": len(variant_out), "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="train 폴드 외 ChemBERTa 채점")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--variants-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"장치: {device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, dataset in enumerate(args.datasets, 1):
        target = out_dir / dataset / "variant_predictions_chemberta_oof.csv"
        if args.resume and target.exists():
            print(f"[{index}/{len(args.datasets)}] {dataset}: 이미 있음", flush=True)
            continue
        print(f"[{index}/{len(args.datasets)}] {dataset}", flush=True)
        record = process_dataset(
            dataset, Path(args.processed_dir), Path(args.variants_dir), out_dir, device
        )
        records.append(record)
        print(
            f"  → 원본 {record['n_origin']:,} + 변형 {record['n_variant']:,} "
            f"({record['elapsed_sec']}초)",
            flush=True,
        )

    if records:
        summary_dir = out_dir / "_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(
            summary_dir / "train_oof_chemberta_summary.csv", index=False
        )
        print(f"\n완료: {len(records)}종")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

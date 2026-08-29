#!/usr/bin/env python
"""변형 분자에 ChemBERTa 예측값을 붙인다 (연구계획서 5.8절 전단계).

담당2가 학습한 체크포인트를 그대로 불러 추론만 한다. 재학습하지 않으므로
담당2의 원본 예측과 정확히 같은 모델이다.

물성마다 정규(regular)와 증강(augmented) 두 버전이 있고, 회귀 물성은 학습 때
라벨을 표준화했으므로 체크포인트의 complete.json에 있는 target_mean과
target_std로 원 단위로 되돌린다. 분류는 양성 클래스 확률을 쓴다.

추론 설정은 담당2의 predict_model과 동일하게 맞춘다.

  최대 토큰 길이  128
  입력            variant_smiles를 그대로 토크나이즈
  분류 출력       softmax 후 1번 클래스 확률
  회귀 출력       스칼라 출력 × target_std + target_mean

이 스크립트는 torch를 쓰므로 conda 환경(rdkit·xgboost)과 같은 프로세스에서
돌리면 OpenMP 런타임이 충돌한다. Drive 바깥의 전용 venv에서 실행한다.

  ~/.venvs/mist-torch/bin/python scripts/score_variants_chemberta.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_repairs import apply_known_repairs  # noqa: E402
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 128
VERSIONS = ("regular", "augmented")


class SmilesDataset(Dataset):
    def __init__(self, smiles):
        self.smiles = [str(value) for value in smiles]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> str:
        return self.smiles[index]


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict(
    checkpoint: Path, smiles, task_type: str, device: torch.device, batch_size: int
):
    """체크포인트 하나로 SMILES 목록을 추론한다."""
    meta = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    model.to(device).eval()

    def collate(batch):
        return tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    loader = DataLoader(
        SmilesDataset(smiles), batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    target_mean = float(meta.get("target_mean", 0.0))
    target_std = float(meta.get("target_std", 1.0))
    values = []
    for encoded in loader:
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        if task_type == "classification":
            values.extend(torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist())
        else:
            scaled = logits.squeeze(-1).float().cpu()
            values.extend((scaled * target_std + target_mean).tolist())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values


def process_dataset(
    dataset: str,
    variants_dir: Path,
    splits_dir: Path,
    checkpoint_root: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    started = time.perf_counter()
    variants = pd.read_csv(variants_dir / dataset / "variants.csv")
    splits = pd.read_csv(splits_dir / dataset / "splits.csv", low_memory=False)
    splits, repaired = apply_known_repairs(splits)
    if repaired:
        print(f"    알려진 SMILES 교정 적용: {', '.join(repaired)}", flush=True)
    task_type = splits["task_type"].iloc[0]

    variant_out = variants[
        ["variant_uid", "parent_row_uid", "dataset", "axis", "split"]
    ].copy()
    origin_out = splits[["row_uid", "dataset", "split"]].copy()

    for version in VERSIONS:
        checkpoint = checkpoint_root / dataset / version
        if not (checkpoint / "complete.json").exists():
            raise FileNotFoundError(f"체크포인트 없음: {checkpoint}")
        variant_out[f"pred_chemberta_{version}"] = predict(
            checkpoint, variants["variant_smiles"], task_type, device, batch_size
        )
        origin_out[f"pred_chemberta_{version}"] = predict(
            checkpoint, splits["parent_smiles"], task_type, device, batch_size
        )

    dataset_out = out_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    variant_out.to_csv(dataset_out / "variant_predictions_chemberta.csv", index=False)
    origin_out.to_csv(dataset_out / "origin_predictions_chemberta.csv", index=False)

    return {
        "dataset": dataset,
        "task_type": task_type,
        "n_variant": len(variants),
        "n_origin": len(splits),
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="변형 분자 ChemBERTa 채점")
    parser.add_argument("--variants-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True, help="checkpoints/chemberta")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    variants_dir = Path(args.variants_dir)
    splits_dir = Path(args.splits_dir)
    checkpoint_root = Path(args.checkpoint_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(args.device)
    print(f"장치: {device}", flush=True)

    datasets = args.datasets or sorted(
        path.name
        for path in variants_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    summaries = []
    for index, dataset in enumerate(datasets, 1):
        target = out_dir / dataset / "variant_predictions_chemberta.csv"
        if args.resume and target.exists():
            print(f"[{index}/{len(datasets)}] {dataset}: 이미 있음, 건너뜀", flush=True)
            continue
        summary = process_dataset(
            dataset,
            variants_dir,
            splits_dir,
            checkpoint_root,
            out_dir,
            device,
            args.batch_size,
        )
        summaries.append(summary)
        print(
            f"[{index}/{len(datasets)}] {dataset}: "
            f"변형 {summary['n_variant']:,} + 원본 {summary['n_origin']:,} "
            f"× 2버전 ({summary['elapsed_sec']}초)",
            flush=True,
        )

    if not summaries:
        print("새로 채점한 물성이 없다.")
        return 0

    summary_dir = out_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summaries)
    frame.to_csv(summary_dir / "chemberta_scoring_summary.csv", index=False)
    print()
    print(
        f"완료: {len(frame)}종, 변형 {int(frame['n_variant'].sum()):,}건 × 2버전, "
        f"총 {frame['elapsed_sec'].sum() / 60:.1f}분"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

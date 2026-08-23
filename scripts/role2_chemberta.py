"""22개 ADMET 데이터셋의 ChemBERTa 정규·증강 모델을 학습한다."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def infer_task(labels: pd.Series) -> str:
    values = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
    return "classification" if len(values) <= 2 and values <= {0, 1} else "regression"


def random_smiles_variants(smiles: str, count: int, seed: int) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [str(smiles)] * count
    variants: list[str] = []
    seen: set[str] = set()
    for offset in range(max(20, count * 10)):
        random.seed(seed + offset)
        value = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
        if value not in seen:
            seen.add(value)
            variants.append(value)
        if len(variants) == count:
            break
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return (variants + [canonical] * count)[:count]


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], labels: list[float] | None = None) -> None:
        self.smiles = smiles
        self.labels = labels

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> dict[str, object]:
        item: dict[str, object] = {"smiles": self.smiles[index]}
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item


class ChemBertaRunner:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(config["model_name"]))

    def collate(self, task: str, with_labels: bool = True):
        def _collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
            encoded = self.tokenizer(
                [str(item["smiles"]) for item in batch],
                padding=True,
                truncation=True,
                max_length=int(self.config["max_length"]),
                return_tensors="pt",
            )
            if with_labels:
                dtype = torch.long if task == "classification" else torch.float32
                encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=dtype)
            return encoded

        return _collate

    def training_rows(self, train: pd.DataFrame, augmented: bool) -> tuple[list[str], list[float]]:
        if not augmented:
            return train["parent_smiles"].astype(str).tolist(), train["Y_final"].tolist()
        smiles_rows: list[str] = []
        labels: list[float] = []
        count = int(self.config["augmentation_per_molecule"])
        seed = int(self.config["seed"])
        for index, row in train.reset_index(drop=True).iterrows():
            variants = random_smiles_variants(row["parent_smiles"], count, seed + index * 17)
            smiles_rows.extend(variants)
            labels.extend([row["Y_final"]] * len(variants))
        return smiles_rows, labels

    @torch.inference_mode()
    def predict(self, model, frame: pd.DataFrame, task: str, mean: float, std: float) -> np.ndarray:
        loader = DataLoader(
            SmilesDataset(frame["parent_smiles"].astype(str).tolist()),
            batch_size=int(self.config["batch_size"]) * 2,
            collate_fn=self.collate(task, with_labels=False),
        )
        predictions: list[float] = []
        model.eval()
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = model(**batch).logits.float()
            if task == "classification":
                values = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            else:
                values = logits.view(-1).cpu().numpy() * std + mean
            predictions.extend(values.tolist())
        return np.asarray(predictions)

    def run(self, dataset: str, processed_dir: Path, output_dir: Path, checkpoint_dir: Path, augmented: bool) -> None:
        version = "augmented" if augmented else "regular"
        prediction_path = output_dir / dataset / f"chemberta_{version}_predictions.csv"
        model_dir = checkpoint_dir / dataset / version
        done_path = model_dir / "complete.json"
        if prediction_path.exists() and done_path.exists():
            print(f"[SKIP] {dataset}/{version}")
            return

        frame = pd.read_csv(processed_dir / dataset / "splits.csv")
        train = frame[frame["split"].eq("train")].copy()
        task = infer_task(frame["Y_final"])
        mean, std = 0.0, 1.0
        if task == "regression":
            mean = float(train["Y_final"].mean())
            std = max(float(train["Y_final"].std(ddof=0)), 1e-8)

        smiles, raw_labels = self.training_rows(train, augmented)
        labels = raw_labels if task == "classification" else [(float(y) - mean) / std for y in raw_labels]
        loader = DataLoader(
            SmilesDataset(smiles, labels),
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            collate_fn=self.collate(task),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.config["model_name"]),
            num_labels=2 if task == "classification" else 1,
            problem_type="single_label_classification" if task == "classification" else "regression",
            ignore_mismatched_sizes=True,
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]))
        epochs = int(self.config["epochs"])
        total_steps = epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        losses: list[float] = []
        started = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    loss = model(**batch).loss
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                running += float(loss.item())
            losses.append(running / max(1, len(loader)))
            print(f"[{dataset}/{version}] epoch {epoch}/{epochs} loss={losses[-1]:.5f}")

        values = self.predict(model, frame, task, mean, std)
        result = pd.DataFrame({"row_uid": frame["row_uid"], f"pred_chemberta_{version}": values})
        if task == "classification":
            result[f"pred_label_chemberta_{version}"] = (values >= 0.5).astype(int)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(prediction_path, index=False)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(model_dir)
        done_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "version": version,
                    "task": task,
                    "target_mean": mean,
                    "target_std": std,
                    "losses": losses,
                    "seconds": time.time() - started,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        del model, optimizer, scheduler, scaler, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/role2.yaml"))
    parser.add_argument("--dataset", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    datasets = args.dataset or sorted(path.parent.name for path in args.processed_dir.glob("*/splits.csv"))
    runner = ChemBertaRunner(config)
    for dataset in datasets:
        runner.run(dataset, args.processed_dir, args.output_dir, args.checkpoint_dir, augmented=False)
        runner.run(dataset, args.processed_dir, args.output_dir, args.checkpoint_dir, augmented=True)


if __name__ == "__main__":
    main()

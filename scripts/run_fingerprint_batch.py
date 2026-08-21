"""22개 물성의 지문 모델을 실행하며 중단 후 재개를 지원한다."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_validation import discover_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_is_complete(dataset: str, split_path: Path, output_dir: Path) -> bool:
    prediction_path = output_dir / dataset / "fingerprint_predictions.csv"
    metrics_path = output_dir / dataset / "fingerprint_metrics.json"
    if not prediction_path.exists() or not metrics_path.exists():
        return False
    try:
        expected_rows = len(pd.read_csv(split_path, usecols=["row_uid"]))
        predictions = pd.read_csv(prediction_path, usecols=["row_uid"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        len(predictions) == expected_rows
        and predictions["row_uid"].is_unique
        and metrics.get("dataset") == dataset
        and metrics.get("rows") == expected_rows
    )


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    split_files = discover_split_files(args.processed_dir)
    available = {path.parent.name: path for path in split_files}
    datasets = args.datasets or sorted(available)
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    batch_dir = args.output_dir / "_batch"
    log_dir = batch_dir / "logs"
    manifest_path = batch_dir / "fingerprint_manifest.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "artifact_dir": str(args.artifact_dir),
        "datasets": {},
    }

    runner = Path(__file__).with_name("run_fingerprint_dataset.py")
    failures = 0
    for position, dataset in enumerate(datasets, start=1):
        split_path = available[dataset]
        if not args.force and result_is_complete(dataset, split_path, args.output_dir):
            print(f"[{position:02d}/{len(datasets):02d}] {dataset}: already complete")
            manifest["datasets"][dataset] = {"status": "skipped_complete"}
            manifest["updated_at"] = utc_now()
            write_manifest(manifest_path, manifest)
            continue

        print(f"[{position:02d}/{len(datasets):02d}] {dataset}: running", flush=True)
        started = time.perf_counter()
        command = [
            sys.executable,
            str(runner),
            "--dataset",
            dataset,
            "--processed-dir",
            str(args.processed_dir),
            "--output-dir",
            str(args.output_dir),
            "--artifact-dir",
            str(args.artifact_dir),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed = round(time.perf_counter() - started, 3)
        log_path = log_dir / f"{dataset}.log"
        log_path.write_text(
            completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        complete = result_is_complete(dataset, split_path, args.output_dir)
        status = "complete" if completed.returncode == 0 and complete else "failed"
        manifest["datasets"][dataset] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "return_code": completed.returncode,
            "log": str(log_path),
        }
        manifest["updated_at"] = utc_now()
        write_manifest(manifest_path, manifest)
        if status == "complete":
            print(f"[{position:02d}/{len(datasets):02d}] {dataset}: complete ({elapsed:.1f}s)")
        else:
            failures += 1
            print(f"[{position:02d}/{len(datasets):02d}] {dataset}: failed — {log_path}")

    manifest["finished_at"] = utc_now()
    manifest["failure_count"] = failures
    manifest["complete_count"] = sum(
        item["status"] in {"complete", "skipped_complete"}
        for item in manifest["datasets"].values()
    )
    write_manifest(manifest_path, manifest)
    print(f"manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

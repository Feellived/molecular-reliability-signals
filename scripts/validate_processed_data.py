"""모델 학습 전에 모든 전처리 데이터를 검사한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_validation import validate_processed_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--skip-smiles",
        action="store_true",
        help="Skip RDKit parsing for a faster structural-only check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = validate_processed_dir(
        args.processed_dir, check_smiles=not args.skip_smiles
    )
    payload = {
        "processed_dir": str(args.processed_dir.resolve()),
        "dataset_count": len(reports),
        "total_rows": sum(report.rows for report in reports),
        "valid": all(report.valid for report in reports),
        "datasets": [report.to_dict() for report in reports],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

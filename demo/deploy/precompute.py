#!/usr/bin/env python
"""정적 배포용으로 채점 결과를 미리 계산한다.

무료로 쓸 수 있는 것은 정적 Space뿐이다. 서버가 없으므로 모델을 돌릴 수
없고, 대신 결과를 미리 만들어 JSON으로 싣는다. 화면 코드는 그대로 쓴다.

Gradio-lite도 검토했으나 접었다. 브라우저 파이썬에는 torch가 없어 표현
불안정성 축이 통째로 빠지고, XGBoost가 대표인 물성도 쓸 수 없다. 정적
방식은 서버에서 미리 채점하므로 축이 하나도 빠지지 않는다.

고르는 기준은 이야기다. 판정 등급이 고루 섞이고 이탈이 큰 사례가 반드시
들어가도록 물성마다 몇 개씩 뽑는다. 임의 분자를 넣는 기능은 잃지만,
발표에서 보여줄 장면은 오히려 또렷해진다.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from score import score  # noqa: E402

DEFAULT = ("bbb_martins", "herg", "lipophilicity_astrazeneca", "caco2_wang")
KNOWN = {
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1": "살부타몰",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C": "카페인",
    "CC(=O)Oc1ccccc1C(=O)O": "아스피린",
}


def max_shift(result: dict) -> float:
    """어느 변형이 원본에서 가장 멀리 갔는가. 고를 때의 기준이 된다."""
    shifts = [abs(v["shift"])
              for axis in result["reliability_axes"]["입력 상태 민감성"]["axes"]
              for v in axis.get("spread", [])]
    return max(shifts, default=0.0)


def pick(scored: list[dict], keep: int) -> list[dict]:
    """판정 등급을 고루 담되 각 등급 안에서는 이탈이 큰 것부터 고른다."""
    buckets: dict[str, list[dict]] = {}
    for record in scored:
        buckets.setdefault(record["verdict"], []).append(record)
    for group in buckets.values():
        group.sort(key=lambda r: -r["max_shift"])

    chosen, index = [], 0
    while len(chosen) < keep and any(len(g) > index for g in buckets.values()):
        for level in ("주의", "보통", "안정"):
            group = buckets.get(level, [])
            if len(group) > index and len(chosen) < keep:
                chosen.append(group[index])
        index += 1
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="정적 배포용 사전 채점")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--out", required=True, help="정적 파일의 data 폴더")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT))
    parser.add_argument("--sample", type=int, default=40, help="물성당 채점할 후보 수")
    parser.add_argument("--keep", type=int, default=15, help="물성당 최종 수록 수")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []

    for dataset in args.datasets:
        evaluation = pd.read_csv(
            Path(args.evaluation_dir) / dataset / "evaluation_signals.csv",
            usecols=["row_uid", "split"])
        splits = pd.read_csv(Path(args.splits_dir) / dataset / "splits.csv",
                             usecols=["row_uid", "parent_smiles"], low_memory=False)
        test = evaluation[evaluation.split.eq("test")].merge(splits, on="row_uid")

        # 예시로 쓰는 분자는 반드시 넣는다. 나머지는 시험 분할에서 고르게 뽑는다.
        forced = [s for s in KNOWN if s in set(test.parent_smiles)]
        rest = test[~test.parent_smiles.isin(forced)]
        step = max(1, len(rest) // args.sample)
        candidates = forced + rest.parent_smiles.iloc[::step].head(args.sample).tolist()

        scored = []
        for position, smiles in enumerate(candidates, 1):
            try:
                result = score(args.bundle, dataset, smiles)
            except Exception:
                continue
            scored.append({"dataset": dataset, "smiles": smiles, "result": result,
                           "verdict": result["verdict"]["level"],
                           "max_shift": max_shift(result)})
            if position % 10 == 0:
                print(f"  {dataset:26s} {position}/{len(candidates)}", flush=True)

        forced_records = [r for r in scored if r["smiles"] in KNOWN]
        others = pick([r for r in scored if r["smiles"] not in KNOWN],
                      max(0, args.keep - len(forced_records)))
        for order, record in enumerate(forced_records + others):
            key = f"{dataset}__{order:02d}"
            (out / f"{key}.json").write_text(
                json.dumps(record["result"], ensure_ascii=False), encoding="utf-8")
            index.append({
                "id": key, "dataset": dataset, "smiles": record["smiles"],
                "name": KNOWN.get(record["smiles"]),
                "task_type": record["result"]["task_type"],
                "prediction": record["result"]["prediction"],
                "verdict": record["verdict"],
                "max_shift": round(record["max_shift"], 4),
            })
        print(f"{dataset:26s} 후보 {len(scored)} → 수록 {len(forced_records) + len(others)}", flush=True)

    (out / "index.json").write_text(
        json.dumps({"molecules": index}, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in out.glob("*.json"))
    print(f"\n수록 {len(index)}건  용량 {total / 1024 / 1024:.1f}MB  →  {out}")
    print("판정 분포:", pd.Series([m["verdict"] for m in index]).value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""정적 Space에 올릴 폴더를 만든다.

HuggingFace에서 무료로 쓸 수 있는 것은 정적 Space뿐이다. Docker와 Gradio는
유료 요금제가 필요해졌다. 정적 Space는 서버가 없으므로 모델을 돌릴 수 없고,
대신 precompute.py가 미리 만들어둔 결과를 읽는다.

화면 코드는 서버 판과 같은 파일을 그대로 쓴다. app.js가 data/index.json이
있는지로 모드를 판별하므로, 이 폴더에는 그 파일이 있고 서버 판에는 없다는
차이만 있다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

README = """---
title: MIST
emoji: 🧪
colorFrom: green
colorTo: gray
sdk: static
pinned: false
---

## MIST

분자 하나의 ADMET 예측값과 함께, 같은 분자를 다르게 적어 넣었을 때 그 예측이 얼마나 흔들리는지를 보여준다.

기존 신뢰성 지표는 모두 모델 쪽에서 나온다. 학습 데이터와 얼마나 비슷한지, 모델이 얼마나 확신하는지를 묻는다. 이 도구는 입력 쪽을 본다. 같은 약이라도 데이터베이스마다 호변이성질체가 다르게 잡히고 양성자화 상태가 다르게 기록되며 입체 표기가 빠지기도 한다. 완전히 같은 분자도 SMILES 문자열로는 여러 가지로 쓸 수 있다.

세 신뢰성 축을 따로 보여주고 하나의 점수로 합치지 않는다. 그리고 각 물성에서 어느 축을 쓸 수 있는지, 쓸 수 없다면 왜인지를 함께 표시한다. 예를 들어 Caco-2 투과도는 여러 출처를 통합해 측정 pH가 보존되지 않으므로 양성자화 변형이 의미를 갖지 않는다.

이 배포본은 정적 페이지라 미리 채점해둔 %d개 분자를 담고 있다. 임의의 분자를 넣으려면 저장소를 내려받아 서버 판으로 실행한다.

TOBIG's 동아리 연구 과제이며 데이터는 Therapeutics Data Commons의 ADMET Benchmark Group을 쓴다.
"""


def human(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.1f}GB"


def main() -> int:
    parser = argparse.ArgumentParser(description="정적 Space 묶음 생성")
    parser.add_argument("--data", required=True, help="precompute.py 산출물 (data 폴더)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data, out = Path(args.data), Path(args.out)
    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    source = Path(__file__).resolve().parents[1] / "app" / "static"
    for name in ("index.html", "style.css", "app.js"):
        shutil.copy2(source / name, out / name)
    shutil.copytree(data, out / "data")
    (out / "README.md").write_text(README % len(index["molecules"]), encoding="utf-8")

    print(f"정적 묶음 {out}   용량 {human(out)}")
    print(f"  분자 {len(index['molecules'])}건")
    counts: dict[str, int] = {}
    for molecule in index["molecules"]:
        counts[molecule["dataset"]] = counts.get(molecule["dataset"], 0) + 1
    for dataset, n in counts.items():
        print(f"    {dataset:26s} {n}건")
    print("\n올리는 법")
    print(f"  cd {out}")
    print("  git init && git add -A && git commit -m 'MIST demo'")
    print("  git remote add origin https://huggingface.co/spaces/<계정>/mist")
    print("  git push -u origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""배포용 묶음을 만든다. HuggingFace Spaces에 올릴 폴더 하나로 모은다.

전체 산출물은 585MB이고 그중 ld50_zhu의 모델 캐시 하나가 247MB다. 데모에
물성 22종이 다 필요하지 않으므로 이야기가 되는 몇 종만 싣는다. 기본값 넷은
이렇게 골랐다.

  bbb_martins               호변이성질체 하나로 예측이 0.012에서 0.535로 바뀐다
  herg                      안전성 평가에서 가장 널리 쓰이는 물성
  lipophilicity_astrazeneca 회귀. 컨포멀 구간이 나온다
  caco2_wang                양성자화 축이 제외되는 사례. 적용 조건 표시를 보여준다

마지막 하나가 중요하다. 쓸 수 없는 축을 왜 제외했는지 말하는 것이 이 도구의
고유한 부분인데, 22종을 다 실으면 오히려 그 장면이 묻힌다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT = ("bbb_martins", "herg", "lipophilicity_astrazeneca", "caco2_wang")
APP_FILES = ("engine.py", "score.py", "api.py")


def human(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.1f}TB"


def main() -> int:
    parser = argparse.ArgumentParser(description="배포 묶음 생성")
    parser.add_argument("--bundle", required=True, help="build_demo_bundle.py 산출물")
    parser.add_argument("--chemberta", required=True, help="담당2 체크포인트 최상위")
    parser.add_argument("--out", required=True, help="배포 폴더")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT))
    args = parser.parse_args()

    bundle, checkpoints = Path(args.bundle), Path(args.chemberta)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "app").mkdir(parents=True)

    here = Path(__file__).resolve().parent
    for name in APP_FILES:
        shutil.copy2(here.parent / "app" / name, out / "app" / name)
    shutil.copytree(here.parent / "app" / "static", out / "app" / "static")
    for name in ("Dockerfile", "requirements.txt", "README.md"):
        source = here / name
        if source.exists():
            shutil.copy2(source, out / name)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    kept = [r for r in manifest["datasets"] if r["dataset"] in set(args.datasets)]
    missing = set(args.datasets) - {r["dataset"] for r in kept}
    if missing:
        raise SystemExit(f"산출물에 없는 물성: {sorted(missing)}")

    target = out / "bundle"
    target.mkdir()
    for dataset in args.datasets:
        shutil.copytree(bundle / dataset, target / dataset)
        source = checkpoints / dataset / "augmented"
        if not source.exists():
            raise SystemExit(f"체크포인트가 없다: {source}")
        shutil.copytree(source, out / "app" / "checkpoints" / dataset / "augmented")

    # 물성 폴더 밖에 있는 산출물도 함께 옮긴다. 판정 경계 보정이 여기 있으며,
    # 빠지면 배포본이 기본 경계값으로 떨어진다.
    for name in ("verdict_calibration.json",):
        source = bundle / name
        if source.exists():
            shutil.copy2(source, target / name)

    manifest["datasets"] = kept
    manifest["deployment"] = {
        "subset": list(args.datasets),
        "note": ("전체 22종 중 일부만 실었다. 모델 캐시가 물성당 수십 MB라 "
                 "전부 담으면 585MB가 된다. 산출물 자체는 22종 모두 있다."),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"배포 묶음 {out}")
    for label, path in (("앱", out / "app"), ("산출물", target),
                        ("체크포인트", out / "app" / "checkpoints")):
        print(f"  {label:8s} {human(path):>8s}")
    print(f"  {'합계':8s} {human(out):>8s}")
    print(f"\n물성 {len(kept)}종: {', '.join(args.datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

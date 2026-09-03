#!/usr/bin/env bash
# 팀원 작업이 반영된 뒤 데모를 갱신한다.
#
# 변형 재생성, 모델 교체, 판정 보정 중 무엇이 바뀌어도 이 순서를 그대로
# 밟으면 된다. 산출물부터 다시 만들어야 하며, 배포 묶음만 다시 싸면 낡은
# 참조 분포가 그대로 올라간다.
#
#   ./deploy/refresh.sh            산출물만 갱신 (모델 캐시 재사용)
#   ./deploy/refresh.sh --models   지문 모델 캐시까지 다시 적합 (20분)
#   ./deploy/refresh.sh --deploy   갱신 후 Cloud Run에 배포까지
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # Juhyeong/
PY="${MIST_PYTHON:-$HOME/miniforge3/envs/mist/bin/python}"
BUNDLE="$HERE/data/processed/scores_role4/demo_bundle"
STAGE="${MIST_STAGE:-/tmp/mist-space}"
WITH_MODELS=""
DO_DEPLOY=""
for arg in "$@"; do
  case "$arg" in
    --models) WITH_MODELS="--with-models" ;;
    --deploy) DO_DEPLOY="1" ;;
    *) echo "모르는 인자: $arg" >&2; exit 1 ;;
  esac
done

echo "[1/4] 판정 경계 보정"
"$PY" "$HERE/scripts/calibrate_verdict.py" \
  --evaluation-dir "$HERE/data/processed/scores_role4/evaluation" \
  --out "$HERE/data/processed/scores_role4/verdict_calibration.json" | tail -5
cp "$HERE/data/processed/scores_role4/verdict_calibration.json" "$BUNDLE/"

echo
echo "[2/4] 데모 산출물 ${WITH_MODELS:+(모델 캐시 포함)}"
"$PY" "$HERE/scripts/build_demo_bundle.py" \
  --evaluation-dir "$HERE/data/processed/scores_role4/evaluation" \
  --role2-dir "$HERE/../Jiye/outputs" \
  --processed-dir "$HERE/data/processed/pipeline_yoonsoo" \
  --reports-dir "$HERE/data/processed/pipeline_yoonsoo/reports" \
  --scores-dir "$HERE/data/processed/scores_role4" \
  --out-dir "$BUNDLE" $WITH_MODELS | tail -6

echo
echo "[3/4] 배포 묶음"
"$PY" "$HERE/demo/deploy/build_space.py" \
  --bundle "$BUNDLE" \
  --chemberta "$HERE/../Jiye/checkpoints/chemberta" \
  --out "$STAGE" | tail -6

echo
if [ -n "$DO_DEPLOY" ]; then
  echo "[4/4] Cloud Run 배포"
  (cd "$STAGE" && gcloud run deploy mist --source . \
     --project mist-demo-507509 --region asia-northeast3 \
     --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 300 --min-instances 0)
else
  echo "[4/4] 배포는 건너뜀. 올리려면 --deploy 를 붙이거나 다음을 실행한다."
  echo "  cd $STAGE && gcloud run deploy mist --source . \\"
  echo "    --project mist-demo-507509 --region asia-northeast3 \\"
  echo "    --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 300 --min-instances 0"
fi

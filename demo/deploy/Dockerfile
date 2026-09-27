# HuggingFace Spaces용. CPU만 쓴다.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf \
    MIST_BUNDLE=/app/bundle \
    MIST_CHEMBERTA=/app/app/checkpoints

WORKDIR /app

# RDKit이 링크하는 시스템 라이브러리. slim 이미지에는 없고 conda 환경에는
# 딸려 오므로 로컬에서는 드러나지 않는다. 목록은 rdkit 리눅스 휠의 공유
# 라이브러리에서 DT_NEEDED를 뽑아 만들었다. boost는 휠에 동봉되어 있어
# 설치하지 않는다. libgomp는 xgboost와 scikit-learn이 쓴다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libx11-6 libxau6 \
    libxcb1 libxcb-render0 libxcb-shm0 \
    libexpat1 libfreetype6 libfontconfig1 libpng16-16 \
    libcairo2 libpixman-1-0 libuuid1 libbz2-1.0 zlib1g \
    libgomp1 libquadmath0 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# torch는 CPU 판만 받는다. 기본 판은 CUDA 라이브러리까지 끌어와 이미지가 몇 배로 커진다.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app
COPY bundle ./bundle

# 호스트마다 포트 규약이 다르다. Spaces는 7860, Cloud Run은 PORT 환경변수로
# 넘긴다. 환경변수가 있으면 그걸 쓰고 없으면 7860으로 떨어진다.
ENV PORT=7860
EXPOSE 7860
WORKDIR /app/app
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]

# HuggingFace Spaces용. CPU만 쓴다.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf \
    MIST_BUNDLE=/app/bundle \
    MIST_CHEMBERTA=/app/app/checkpoints

WORKDIR /app

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

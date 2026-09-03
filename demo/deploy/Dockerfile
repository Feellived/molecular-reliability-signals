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

# Spaces는 7860 포트를 쓴다.
EXPOSE 7860
WORKDIR /app/app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

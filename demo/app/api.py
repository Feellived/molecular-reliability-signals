#!/usr/bin/env python
"""데모 웹 서버. 채점기에 HTTP 한 겹을 씌우고 정적 화면을 함께 서빙한다.

채점 자체는 score.py가 하고 이 파일은 요청을 받아 넘기기만 한다. 화면을
바꾸는 작업이 채점 로직을 건드리지 않게 하기 위해서다.

산출물은 물성마다 지문 모델 다섯 개와 언어 모델 체크포인트를 읽으므로
첫 호출이 8초쯤 걸린다. engine.load_bundle이 캐시하므로 이후로는 1초 미만이다.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import load_bundle
from score import score

BUNDLE = Path(os.environ.get(
    "MIST_BUNDLE",
    Path(__file__).resolve().parents[2] / "data/processed/scores_role4/demo_bundle"))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="MIST", description="입력 상태 민감성 기반 예측 신뢰성 표시")


class ScoreRequest(BaseModel):
    dataset: str = Field(description="물성 데이터셋 이름")
    smiles: str = Field(description="분자의 SMILES 표기")


@app.get("/api/datasets")
def datasets() -> dict:
    """물성 목록과 각 물성에서 쓸 수 있는 축."""
    import json
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for record in manifest["datasets"]:
        axes = json.loads((BUNDLE / record["dataset"] / "axes.json").read_text(encoding="utf-8"))
        out.append({
            "dataset": record["dataset"],
            "task_type": record["task_type"],
            "usable_axes": [k for k, v in axes.items()
                            if not k.startswith("_") and v["사용"]],
            "excluded_axes": [k for k, v in axes.items()
                              if not k.startswith("_") and not v["사용"]],
        })
    return {"datasets": out, "settings_digest": manifest["digest"]}


@app.post("/api/score")
def score_molecule(request: ScoreRequest) -> dict:
    try:
        return score(BUNDLE, request.dataset, request.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404,
                            detail=f"산출물을 찾을 수 없다: {request.dataset}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict:
    import json
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    return {"status": "ok", "settings_digest": manifest["digest"],
            "n_datasets": len(manifest["datasets"]),
            "cached": load_bundle.cache_info()._asdict()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

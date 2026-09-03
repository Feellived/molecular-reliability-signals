## MIST 데모

분자 하나를 입력하면 예측값과 컨포멀 구간, 축별 흔들림, 그리고 그 물성에 어느 축을 쓸 수 있는지를 보여준다. 연구계획서 7.3절의 분자 수준 신뢰성 표시에 해당한다.

### 구성

```
app/engine.py    변형 생성, 지문·언어 모델 채점, 산출물 읽기
app/score.py     축별 흩어짐과 백분위, 컨포멀 구간, 판정 문장
app/api.py       FastAPI. 채점기에 HTTP 한 겹
app/static/      화면. index.html, style.css, app.js
app/validate.py  파이프라인 값 재현 검증
```

계층이 분리되어 있어 화면을 바꿔도 채점 로직을 건드리지 않는다.

### 실행

```
~/miniforge3/envs/mist/bin/python -m uvicorn api:app --port 8811
```

`app/` 안에서 실행한다. 산출물 경로는 `MIST_BUNDLE` 환경변수로 바꿀 수 있고 기본값은 `data/processed/scores_role4/demo_bundle`이다.

산출물이 없으면 먼저 만든다. 모델 캐시까지 20분쯤 걸린다.

```
python scripts/build_demo_bundle.py \
  --evaluation-dir data/processed/scores_role4/evaluation \
  --role2-dir ../Jiye/outputs \
  --processed-dir data/processed/pipeline_yoonsoo \
  --reports-dir data/processed/pipeline_yoonsoo/reports \
  --scores-dir data/processed/scores_role4 \
  --out-dir data/processed/scores_role4/demo_bundle --with-models
```

### 설정 지문

`manifest.json`에 변형 생성 설정의 지문이 들어 있고 채점기가 시작할 때 대조한다. A축 변형 개수나 pH 창을 바꾸면 참조 분포가 낡으므로 실행을 거부한다. 조용히 틀린 백분위를 내놓는 것이 가장 위험한 실패 방식이라 그렇게 두었다. 설정을 바꾸면 산출물을 다시 만들어야 한다.

### 화면 규칙

계획서 7.3절이 세 신뢰성 축을 분리해 표시하고 단일 점수로 합산하지 말 것을 정한다. 결합 위험 점수는 접힌 영역에 참고값으로만 둔다.

쓰지 않은 축을 왜 제외했는지 보여주는 부분이 이 도구의 고유한 기능이다. 예를 들어 caco2_wang을 고르면 양성자화 축이 측정 pH가 보존되지 않아 제외되었다고 표시된다.

### 검증

```
python app/validate.py
```

시험 분할에 이미 있는 분자를 다시 채점해 저장값과 대조한다. 22종 176건에서 실패 0건이며, 데모 예측은 파이프라인 재적합과 평균 절대차 0.0000으로 일치한다. 적용가능도메인은 최대 절대차 0.000049다. 표현 불안정성 A는 무작위 SMILES 생성 시드가 파이프라인과 달라 변형 집합 자체가 다르므로 대조 대상에서 뺀다.

### 남은 일

배포는 HuggingFace Spaces(Docker)를 염두에 둔다. 모델 캐시가 585MB이고 ld50_zhu 하나가 247MB이므로 물성을 몇 종만 실어 올리는 편이 낫다.

회귀 물성의 ChemBERTa 컨포멀은 담당2의 척도 함수를 재구성할 수 없어 결합 규칙에서 뺐다. 정의를 확인하면 채울 수 있다.

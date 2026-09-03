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

### 배포

Google Cloud Run에 올라가 있다.

  https://mist-700016601256.asia-northeast3.run.app

절차는 deploy/CLOUD_RUN.md에 있다. 최소 인스턴스가 0이라 접속이 없으면 잠들고 과금이 발생하지 않는다. 대신 한동안 쓰지 않으면 첫 접속이 30초에서 1분 걸리므로, 시연 전에 한 번 열어 깨워두는 편이 좋다.

HuggingFace Spaces는 쓰지 않는다. Docker와 Gradio SDK가 유료 요금제를 요구하도록 바뀌었고, 무료로 남은 정적 Space는 서버가 없어 임의 분자를 채점할 수 없다. 정적 배포 경로도 함께 만들어두었으므로(deploy/precompute.py, deploy/build_static.py) 필요하면 미리 채점한 결과만으로 올릴 수 있다.

배포 묶음은 다음으로 만든다. 전체 산출물은 585MB이고 그중 ld50_zhu의 모델 캐시 하나가 247MB이므로, 이야기가 되는 물성 넷만 실어 81MB로 줄인다.

```
python deploy/build_space.py \
  --bundle ../data/processed/scores_role4/demo_bundle \
  --chemberta ../../Jiye/checkpoints/chemberta \
  --out /tmp/mist-space
```

만들어진 폴더를 Space 저장소에 그대로 올리면 된다. `README.md`의 앞머리가 Spaces 설정을 담고 있고 Dockerfile이 7860 포트로 띄운다. `--datasets`로 물성을 바꿀 수 있다.

기본 넷은 이렇게 골랐다. bbb_martins는 호변이성질체 하나로 예측이 0.012에서 0.535로 바뀌는 사례이고, herg는 안전성 평가에서 가장 널리 쓰이며, lipophilicity_astrazeneca는 컨포멀 구간이 나오는 회귀이고, caco2_wang은 양성자화 축이 제외되는 사례다. 마지막 하나가 중요하다. 쓸 수 없는 축을 왜 제외했는지 말하는 것이 이 도구의 고유한 부분이라 그 장면이 빠지면 안 된다.

체크포인트와 산출물 경로는 `MIST_CHEMBERTA`와 `MIST_BUNDLE` 환경변수로 정한다. 저장소 구조에 의존하지 않으므로 배포 환경에서 그대로 돈다.

`requirements.txt`의 scikit-learn 판을 고정해두었다. 지문 모델이 joblib으로 저장되어 있어 판이 다르면 역직렬화가 조용히 실패하거나 다른 결과를 낸다.

### 갱신

팀원 작업이 반영되면 데모도 다시 만들어야 한다. 순서가 있으므로 스크립트로 묶어두었다.

```
./deploy/refresh.sh            산출물만 갱신
./deploy/refresh.sh --models   지문 모델 캐시까지 재적합 (20분)
./deploy/refresh.sh --deploy   갱신 후 Cloud Run 배포까지
```

배포 묶음만 다시 싸면 낡은 참조 분포가 그대로 올라간다. 판정 보정과 산출물부터 다시 만들어야 하며 스크립트가 그 순서를 지킨다.

설정 지문은 변형 생성 설정만 덮는다. A축 개수나 pH 창이 바뀌면 데모가 실행을 거부하지만, 모델이 교체되거나 결합 계수와 판정 보정이 바뀌면 지문은 그대로다. 그래서 물성마다 결합 계수와 참조 신호 목록, 대표 모델을 묶은 재료 지문을 따로 남기고, 산출물 생성 시각을 화면의 참고값 항목에 표시한다.

갱신이 필요한 작업은 변형 재생성, 새 신호 추가, 결합 규칙에 영향을 주는 변경이다. 통계 절차 변경은 보고 신뢰구간만 바꾸므로 데모와 무관하다.

### 남은 일

회귀 물성의 ChemBERTa 컨포멀은 담당2의 척도 함수를 재구성할 수 없어 결합 규칙에서 뺐다. 정의를 확인하면 채울 수 있다.

판정 등급의 경계값이 임의다. 상위 15퍼센트를 주의로 두었는데 실제 오차율에 맞춰 정할 여지가 있다.

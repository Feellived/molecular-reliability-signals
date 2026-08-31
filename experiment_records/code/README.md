# 지문 모델 실행 코드 사본

담당 2 지문 모델 결과의 재현성을 위해 실행 당시 사용한 핵심 코드를 함께 보관한다.

- `fingerprint_models.py`: Morgan fingerprint와 RF/XGBoost 정의
- `run_fingerprint_dataset.py`: 물성 하나의 학습·예측·모델 저장
- `run_fingerprint_batch.py`: 22개 물성 일괄 실행과 재개

실제 개발과 재실행에는 `scripts/` 아래의 같은 이름 파일을 사용한다.

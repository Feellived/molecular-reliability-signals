# 담당 2 실험 결과

이 폴더에는 22개 ADMET 데이터셋의 담당 2 실험 요약과 재현 정보를 저장한다.
행별 예측값, 학습 체크포인트, 원본 데이터는 용량이 크므로 Google Drive의 `tobigs_2026`에서 관리한다.
성능 지표는 각 데이터셋의 `test` 분할에서 계산했다.

## 저장 원칙

- Git: 설정, 실행 코드, 데이터셋별 요약표, 검증 보고서
- Google Drive: `role2_signals.csv`, 모델 체크포인트, Morgan 지문, 원본 데이터
- 결합 키: 모든 파일에서 `row_uid`

## Git 요약 파일

- `dataset_status.csv`: 22종 분할 크기와 작업별 완료 상태
- `execution_summary.json`: 실험 설정과 최종 완료 범위
- `model_summary.csv`: 모델별 test 평균 성능
- `key_findings.json`: 증강 효과와 컨포멀 포함률 핵심 수치

## Drive 산출물

- `outputs/<물성명>/fingerprint_predictions.csv`
- `outputs/<물성명>/chemberta_regular_predictions.csv`
- `outputs/<물성명>/chemberta_augmented_predictions.csv`
- `outputs/<물성명>/conformal_predictions.csv`
- `outputs/<물성명>/applicability_domain.csv`
- `outputs/<물성명>/model_disagreement.csv`
- `outputs/<물성명>/role2_signals.csv`
- `experiment_records/role2_execution_summary.json`
- `analysis/role2_model_metrics.csv`
- `analysis/role2_reliability_analysis.csv`
- `analysis/role2_analysis_report.md`
- `analysis/*.png`

## 재현 코드

- `scripts/role2_chemberta.py`: ChemBERTa 정규·증강 학습
- `scripts/role2_signals.py`: 적응형 컨포멀, AD, 모델 불일치, 최종 신호 생성
- `configs/role2.yaml`: 공통 실험 설정
- `notebooks/02_role2_colab_runner.ipynb`: Colab GPU 실행 순서

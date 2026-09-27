# 담당 2 후속 점검 실행 순서

이 문서는 기존 22종 학습 결과를 삭제하거나 덮어쓰지 않고 후속 점검 결과를 별도 폴더에 만드는 방법을 설명한다.

## 생성되는 결과

- `input_manifest.csv`: 22종 입력 파일 크기·분할 수·SHA256
- `smiles_augmentation_diversity.csv`: 무작위 SMILES 고유 변형 수
- `base_model_meta_oof.csv`: train의 같은 `cv_fold`를 제외하고 예측한 meta 결과
- `meta_level_reliability_oof.csv`: meta 신호를 fold 밖에서 결합한 기준 결과
- `role2_features.csv`: 정답 기반 열을 제거한 담당 4 입력 파일
- `role2_evaluation_only.csv`: 정답·coverage 평가 전용 파일
- `dataset_model_metrics_detailed.csv`: 데이터셋별 모델 지표와 bootstrap 구간
- `xgboost_regression_diagnostics.csv`: 회귀 XGBoost 이상 결과
- `conformal_detailed.csv`: 데이터셋별 coverage·구간 폭·집합 크기
- `reliability_correlations_bh.csv`: 신뢰도 신호와 오차의 상관 및 BH 보정

## Colab 실행

Drive와 GitHub 저장소를 연결한 뒤 저장소 루트에서 실행한다.

```python
!python scripts/role2_followup_audit.py \
  --processed-dir /content/data/processed \
  --output-dir /content/drive/MyDrive/tobigs_2026/outputs \
  --report-dir /content/drive/MyDrive/tobigs_2026/followup \
  --variant-attempts 30
```

두 방식의 `meta cv-fold` 결과 생성:

```python
!python scripts/role2_meta_oof.py \
  --processed-dir /content/data/processed \
  --output-dir /content/drive/MyDrive/tobigs_2026/outputs \
  --meta-output-dir /content/drive/MyDrive/tobigs_2026/followup/meta_oof \
  --limit 200
```

상세 결과 재분석:

```python
!python scripts/role2_result_analysis.py \
  --processed-dir /content/data/processed \
  --output-dir /content/drive/MyDrive/tobigs_2026/outputs \
  --report-dir /content/drive/MyDrive/tobigs_2026/followup/analysis \
  --bootstrap-repeats 2000
```

## 중요한 구분

- `base_model_meta_oof`: 담당 2 지문 모델의 fold 제외 예측
- `meta_level_reliability_oof`: 담당 4 방식에 가까운 신뢰도 결합 기준 실험
- `Y_final`, `aps_true_pvalue`, `aps_calibrated_margin`, `conformal_true_score`: 평가 전용
- `role2_features.csv`: 담당 4 모델 입력용
- 기존 `role2_signals.csv`는 호환성을 위해 유지하지만 그대로 모델 입력에 사용하지 않음

## 현재 로컬 환경 주의

Windows 기본 `python`이 존재하지 않는 Python 3.9 경로를 가리키고 있어 로컬 실행이 불가능한 상태다. RDKit·XGBoost가 준비된 Colab에서 위 셀을 실행하거나 로컬 Python 환경을 다시 만든 뒤 실행한다.

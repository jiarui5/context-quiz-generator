# Unsloth Studio 실행 및 증거 수집 체크리스트

본 프로젝트는 과제에서 권장하는 Unsloth Studio 경로로 실제 파인튜닝을 수행한다.

## 1. Studio 시작

1. [공식 Unsloth Studio Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Unsloth_Studio.ipynb)을 연다.
2. Colab 런타임이 T4 GPU인지 확인한다.
3. `Run all`을 실행하고 표시된 링크로 Unsloth Studio를 연다.

## 2. 모델 로드

- 후보 모델: `Qwen2.5-3B-Instruct`
- 최종 모델명, 크기, 라이선스, VRAM 사용량을 기록한다.
- 모델 로드 화면을 캡처한다.

## 3. 데이터 Import 및 Data Recipes

- `data/final/train.jsonl`과 `validation.jsonl`을 import한다.
- 각 샘플의 `user`와 `assistant` role이 올바른지 확인한다.
- user에는 지문과 질문 생성 지시가, assistant에는 생성된 질문 하나만 있는지 확인한다.
- 데이터 미리보기와 샘플 수를 캡처한다.

## 4. QLoRA 학습

초기 설정:

| 항목 | 초기값 |
|---|---:|
| learning rate | 2e-4 |
| epochs | 1 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| effective batch | 16 |
| target modules | attention + MLP |

학습 중 다음 증거를 저장한다.

- training loss 곡선
- validation loss
- gradient norm
- GPU 사용률
- 최종 학습 시간

## 5. Model Arena 비교

학습에 사용하지 않은 동일한 test context를 베이스 모델과 파인튜닝 모델에 입력한다.

- 질문을 정확히 하나 생성하는지
- 지문과 관련 있는지
- 지문만으로 답할 수 있는지
- 질문이 자연스러운지

비교 화면을 캡처하고 `results/comparison.csv`에 결과를 기록한다.

## 6. Export

- LoRA adapter를 저장한다.
- GGUF로 export한다.
- export 완료 화면과 실제 추론 화면을 캡처한다.

## 7. GitHub에 추가할 증거

완료 후 `images/`에 다음 파일을 추가한다.

- `model-load.png`
- `dataset-preview.png`
- `loss-curve.png`
- `model-arena.png`
- `inference-example.png`
- `export-result.png`


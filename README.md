# 영어 지문 기반 퀴즈 질문 자동 생성 모델

SQuAD의 영어 `context`를 원천 텍스트로 활용하여, 주어진 지문만으로 답할 수 있는 퀴즈 질문을
정확히 하나 생성하도록 베이스 언어 모델을 Unsloth QLoRA로 파인튜닝하는 프로젝트입니다.

> 현재 상태: 데이터 파이프라인 및 Colab 코드 구축 완료, 실제 학습 및 결과 기록 예정

## 1. 프로젝트 소개

긴 영어 지문을 바탕으로 평가·복습용 질문을 직접 작성하는 데에는 많은 시간이 필요합니다.
본 프로젝트는 교사, 학습자, 교육 콘텐츠 제작자가 영어 지문을 입력하면 해당 지문만으로 답할 수
있는 자연스러운 퀴즈 질문 하나를 생성하는 모델을 개발합니다.

## 2. 주제 및 목표

파인튜닝 모델은 다음 기준을 베이스 모델보다 안정적으로 만족하는 것을 목표로 합니다.

1. 질문을 정확히 하나만 생성한다.
2. 질문이 입력 지문과 관련 있다.
3. 질문의 답을 입력 지문에서 찾을 수 있다.
4. 자연스럽고 문법적으로 올바른 영어 질문을 생성한다.
5. 정답, 선택지, 설명 등 불필요한 내용을 출력하지 않는다.

## 3. 주요 기능

- 영어 지문을 입력받아 퀴즈 질문을 정확히 하나 생성
- 입력 지문만으로 답할 수 있는 질문 생성
- 정답·선택지·설명 없이 질문만 출력
- 다양한 Wikipedia 주제의 영어 지문 처리
- 베이스 모델보다 안정적인 출력 형식과 높은 지문 관련성을 목표로 함

## 4. 사용한 데이터셋

- 원천 데이터: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)
- 사용 필드: `context`
- 직접 사용하지 않는 필드: 기존 `question`, `answers`
- 처리 과정: context 중복 제거 → 길이 필터링 → 새 질문 생성 → 품질 검수 → chat 형식 변환

기존 질문을 그대로 학습하지 않고 `context`만 원천 자료로 사용합니다. 각 context에 대해 프로젝트
목적에 맞는 새로운 질문을 생성·검수하여 직접 파인튜닝 데이터셋을 구축합니다.
자세한 내용은 [`data/README.md`](data/README.md)와
[`docs/DATA_GENERATION_GUIDE.md`](docs/DATA_GENERATION_GUIDE.md)를 참고하십시오.

## 5. 사용한 기술 스택

| 구분 | 기술 |
|---|---|
| Base Model | `unsloth/Qwen2.5-3B-Instruct` 후보, 실험 후 확정 |
| Fine-tuning | Unsloth, QLoRA |
| Libraries | PyTorch, Transformers, TRL, PEFT, Datasets |
| Environment | Google Colab NVIDIA GPU |
| Data Source | SQuAD context |
| Language | Python |

## 6. 설치 및 실행 방법

### Google Colab 학습

1. [`finetuning.ipynb`](finetuning.ipynb)를 Google Colab에서 엽니다.
2. `런타임 → 런타임 유형 변경 → T4 GPU`를 선택합니다.
3. 셀을 순서대로 실행하여 라이브러리 설치, 데이터 로드, QLoRA 학습 및 adapter export를 수행합니다.

### 추론 및 비교

1. [`inference.ipynb`](inference.ipynb)를 Google Colab에서 엽니다.
2. 베이스 모델과 저장된 LoRA adapter를 로드합니다.
3. 동일한 test context로 출력을 생성하고 `results/comparison.csv`에 기록합니다.

### 로컬 데이터 전처리

```bash
pip install -r requirements.txt

python scripts/prepare_squad_contexts.py \
  --input data/raw/squad_train.json \
  --output data/processed/contexts.jsonl

python scripts/build_chat_dataset.py \
  --input data/interim/generated_questions.jsonl \
  --output-dir data/final
```

## 7. 파인튜닝 설계 설명

| 항목 | 초기 설정 | 선택 근거 |
|---|---|---|
| 베이스 모델 | `unsloth/Qwen2.5-3B-Instruct` 후보 | 영어 지시 수행 능력과 Colab GPU 환경의 VRAM 균형 |
| 방법 | QLoRA | 4-bit 베이스 모델로 VRAM을 절약하면서 LoRA adapter 학습 |
| learning rate | `2e-4` | LoRA/QLoRA 학습의 일반적인 시작값, 실제 결과에 따라 조정 |
| epochs | `1–3` | 과적합을 방지하면서 출력 행동을 학습하기 위한 범위 |
| LoRA rank | `16` | 표현력과 메모리 사용량의 균형 |
| LoRA alpha | `32` | 초기값으로 `2 × rank` 사용 |
| target modules | attention + MLP 전체 | 질문 생성 행동을 충분히 학습시키기 위함 |
| effective batch | `16` | 안정적인 gradient와 제한된 VRAM의 균형 |

최종 하이퍼파라미터는 training/validation loss와 베이스 모델 비교 결과를 근거로 확정합니다.

## 8. 학습 과정

실제 Unsloth QLoRA 학습 후 다음 자료를 추가합니다.

- training/validation loss 곡선
- GPU 환경과 학습 시간
- 과적합·과소적합 판단
- 하이퍼파라미터 수정 과정
- LoRA adapter 및 GGUF export 결과

## 9. 베이스 모델 vs 파인튜닝 모델 비교

학습에 사용하지 않은 동일한 test context를 두 모델에 입력하여 비교합니다.

| 평가 항목 | 베이스 모델 | 파인튜닝 모델 | 변화 |
|---|---:|---:|---:|
| 단일 질문 준수율 | 학습 후 기록 | 학습 후 기록 | |
| 관련성 평균 점수 | 학습 후 기록 | 학습 후 기록 | |
| 답변 가능성 평균 점수 | 학습 후 기록 | 학습 후 기록 | |
| 자연스러움 평균 점수 | 학습 후 기록 | 학습 후 기록 | |

정량·정성 결과는 [`results/comparison.csv`](results/comparison.csv)에 기록합니다.

## 10. 실행 결과 및 스크린샷

학습 완료 후 [`images/`](images/)에 다음 증거를 추가합니다.

- Colab 학습 완료 화면
- loss 곡선
- Model Arena 또는 베이스 vs 파인튜닝 비교
- 파인튜닝 모델 추론 예시
- 모델 export 및 실행 화면

## 11. 한계점 및 개선 방향

- 합성 질문 생성 과정에서 오류가 포함될 수 있으므로 사람 기반 품질 검수가 필요합니다.
- 사람 평가의 주관성을 줄이기 위해 복수 평가자 또는 자동 평가 방법을 함께 사용할 필요가 있습니다.
- 향후 다양한 길이와 도메인의 문서에서 일반화 성능을 평가할 예정입니다.

## 12. 참고문헌

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

## 프로젝트 구조

```text
.
├── README.md
├── finetuning.ipynb
├── inference.ipynb
├── requirements.txt
├── data/
├── docs/
├── images/
├── results/
└── scripts/
```

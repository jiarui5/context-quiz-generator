# 영어 지문 기반 퀴즈 질문 자동 생성 모델

SQuAD의 영어 `context`를 원천 텍스트로 활용하여, 주어진 지문만으로 답할 수 있는
퀴즈 질문 하나를 생성하도록 베이스 언어 모델을 Unsloth QLoRA로 파인튜닝하는 프로젝트입니다.

> 현재 상태: 프로젝트 설계 및 데이터 파이프라인 구축 단계  
> 실제 학습 로그, loss 곡선, Model Arena 비교 결과 및 export 증거는 학습 후 추가합니다.

## 1. 문제 정의

긴 영어 지문을 읽고 평가 또는 복습에 사용할 질문을 직접 만드는 작업은 시간이 많이 듭니다.
본 프로젝트는 교사와 학습자가 지문을 입력하면, 지문의 핵심 내용을 확인할 수 있고 지문만으로
답할 수 있는 질문을 **정확히 하나** 생성하는 모델을 만드는 것을 목표로 합니다.

### 목표 사용자

- 영어 독해 자료를 만드는 교사
- 읽은 내용을 스스로 점검하려는 학습자
- 문서 기반 퀴즈 초안을 빠르게 만들려는 콘텐츠 제작자

### 성공 기준

학습에 사용하지 않은 지문에서 베이스 모델보다 다음 기준을 더 잘 만족하면 성공으로 판단합니다.

1. 질문을 정확히 하나만 생성한다.
2. 질문이 입력 지문과 관련 있다.
3. 질문의 답을 입력 지문에서 찾을 수 있다.
4. 질문이 자연스럽고 문법적으로 올바르다.
5. 질문에 정답이나 불필요한 설명을 함께 출력하지 않는다.

## 2. 데이터셋 설계

- 원천 데이터: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)
- 사용 필드: `context`
- 사용하지 않는 원본 필드: 기존 `question`, `answers`
- 변환 과정: 중복 context 제거 → 길이 필터링 → 새로운 질문 생성 → 품질 검수 → chat 형식 변환

SQuAD의 기존 질문을 그대로 학습하면 단순한 데이터 재사용이 됩니다. 본 프로젝트는 `context`만
원천 자료로 사용하고, 각 지문에 맞는 새로운 질문을 생성·검수하여 프로젝트 목적에 맞는
파인튜닝 데이터셋을 직접 설계합니다.

학습 샘플 형식:

```json
{"messages":[{"role":"user","content":"Read the passage and generate exactly one quiz question that can be answered using only the passage. Output only the question.\n\nPassage:\n..."},{"role":"assistant","content":"...?"}]}
```

## 3. 모델 및 학습 설계

| 항목 | 초기 설정 | 선택 근거 |
|---|---|---|
| 베이스 모델 | `unsloth/Qwen2.5-3B-Instruct` 후보 | 영어 지시 수행 능력과 Colab T4 환경의 VRAM 균형 |
| 방법 | QLoRA | 4-bit 베이스 모델로 VRAM을 절약하면서 LoRA 어댑터 학습 |
| learning rate | `2e-4` | LoRA/QLoRA의 일반적인 시작값, 실험 후 조정 |
| epochs | `1–3` | 과적합을 피하면서 형식과 작업을 학습시키기 위한 범위 |
| LoRA rank | `16` | 표현력과 메모리 사용량의 균형 |
| LoRA alpha | `32` | 초기값으로 `2 × rank` 사용 |
| target modules | attention + MLP 전체 | 질문 생성 행동을 충분히 학습시키기 위함 |
| effective batch | `16` | 안정적인 gradient와 제한된 VRAM의 균형 |

최종 설정은 실제 loss와 검증 결과를 근거로 확정합니다.

## 4. 실행 방법

### 원천 context 추출

```bash
python scripts/prepare_squad_contexts.py \
  --input data/raw/squad_train.json \
  --output data/processed/contexts.jsonl
```

### 생성된 질문을 chat 데이터로 변환

`data/interim/generated_questions.jsonl`에 `context`, `generated_question` 필드를 준비한 후:

```bash
python scripts/build_chat_dataset.py \
  --input data/interim/generated_questions.jsonl \
  --output-dir data/final
```

## 5. 평가 계획

학습에 사용하지 않은 동일한 test context를 베이스 모델과 파인튜닝 모델에 입력합니다.
사람 평가와 자동 형식 평가를 함께 사용합니다.

| 평가 항목 | 측정 방법 |
|---|---|
| 단일 질문 준수율 | 출력의 질문 수와 불필요한 답변 포함 여부 확인 |
| 지문 관련성 | 1–5점 사람 평가 |
| 지문 기반 답변 가능성 | 1–5점 사람 평가 |
| 자연스러움 | 1–5점 사람 평가 |
| 평균 질문 길이 / 중복률 | 스크립트 기반 정량 분석 |

비교 결과는 `results/comparison.csv`와 `results/comparison-summary.md`에 기록합니다.

## 6. 제출 증거 체크리스트

- [ ] Unsloth 실제 학습 로그
- [ ] training/validation loss 곡선
- [ ] 베이스 vs 파인튜닝 Model Arena 스크린샷
- [ ] 정량·정성 비교 결과
- [ ] LoRA adapter 또는 GGUF export
- [ ] 로컬 또는 Colab 추론 화면
- [ ] 프로젝트 보고서
- [ ] 발표 영상

## 7. 프로젝트 구조

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   └── final/
├── docs/
├── report/
├── results/
└── scripts/
```


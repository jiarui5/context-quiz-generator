# 긴 텍스트 기반 퀴즈 자동 생성기

긴 텍스트 passage를 입력받아, passage 안의 정보만으로 답할 수 있는 퀴즈 질문을 정확히 하나 생성하도록
`unsloth/Qwen2.5-3B-Instruct`를 Unsloth Studio에서 QLoRA로 파인튜닝한 프로젝트입니다.

> 최종 상태: 데이터셋 구축, Unsloth Studio QLoRA 학습, validation loss 확인, LoRA adapter export,
> Hugging Face Hub 업로드, Base Model vs Fine-tuned Model 비교 완료.

## 1. 프로젝트 소개

교육 콘텐츠 제작자, 교사, 학습자는 긴 지문에서 복습용 질문을 직접 만드는 데 많은 시간을 사용합니다.
본 프로젝트는 Wikipedia 기반 SQuAD 지문을 입력하면, 정답이나 해설 없이 자연스러운 quiz question
하나만 출력하는 모델을 만드는 것을 목표로 합니다.

선택 시나리오: **11. 퀴즈/문제 생성기**

## 2. 목표

파인튜닝 모델은 다음 기준을 안정적으로 만족해야 합니다.

1. 질문을 정확히 하나만 생성한다.
2. 질문이 입력 passage와 관련 있다.
3. 질문의 답을 passage에서 찾을 수 있다.
4. 자연스럽고 문법적으로 올바른 영어 질문을 생성한다.
5. 정답, 선택지, 해설 등 불필요한 내용을 출력하지 않는다.

## 3. 주요 기능

- 긴 passage 기반 quiz question 생성
- passage만으로 답할 수 있는 질문 생성
- 정답, 선택지, 해설 없이 질문만 출력
- Unsloth Studio의 Model Arena로 base model과 fine-tuned model 비교
- LoRA adapter를 Hugging Face Hub에 export하여 재사용 가능

## 4. 데이터셋

- 원천 데이터: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)
- 생성 데이터: [jiarui5/squad-context-quiz-generated](https://huggingface.co/datasets/jiarui5/squad-context-quiz-generated)
- 사용 필드: `context`
- 직접 사용하지 않은 필드: 원본 SQuAD의 기존 `question`, `answers`
- 처리 과정: context 중복 제거 -> 길이 필터링 -> 새 질문 생성 -> 품질 검수 -> ChatML 형식 변환

기존 SQuAD 질문을 그대로 학습하지 않고 `context`만 원천 텍스트로 사용했습니다. 각 context에 대해
프로젝트 목적에 맞는 새 질문을 생성하고 검수한 뒤, Unsloth Studio가 인식할 수 있는 `messages`
형식으로 변환했습니다.

최종 데이터 통계:

| Split | Samples |
|---|---:|
| Train | 349 |
| Validation | 43 |
| Test | 45 |

데이터 예시:

```json
{
  "context_id": "context-000695",
  "messages": [
    {
      "role": "user",
      "content": "Read the passage and generate exactly one meaningful quiz question that can be answered using only the passage. Output only the question.\n\nPassage:\n..."
    },
    {
      "role": "assistant",
      "content": "In which season did the advertising prices for American Idol reach their peak?"
    }
  ]
}
```

자세한 데이터 구축 과정은 [`data/README.md`](data/README.md)와
[`docs/DATA_GENERATION_GUIDE.md`](docs/DATA_GENERATION_GUIDE.md)를 참고하십시오.

## 5. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Base Model | `unsloth/Qwen2.5-3B-Instruct` |
| Fine-tuning | Unsloth Studio |
| Method | QLoRA, LoRA adapter |
| Runtime | Google Colab T4 GPU |
| Dataset Source | SQuAD context |
| Export | LoRA Only -> Hugging Face Hub |
| Result Adapter | [jiarui5/context-quiz-qwen2.5-3b-lora](https://huggingface.co/jiarui5/context-quiz-qwen2.5-3b-lora) |

## 6. 설치 및 실행 방법

### Unsloth Studio 학습 재현

1. [Unsloth Studio Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Unsloth_Studio.ipynb)을 엽니다.
2. Runtime을 T4 GPU로 설정하고 `Run all`을 실행합니다.
3. Studio에서 base model로 `unsloth/Qwen2.5-3B-Instruct`를 선택합니다.
4. Dataset 탭에서 `data/final/train.jsonl`을 training dataset으로 업로드합니다.
5. `data/final/validation.jsonl`을 eval dataset으로 업로드합니다.
6. QLoRA 설정으로 학습을 시작합니다.
7. 학습 완료 후 Model Arena에서 base model과 fine-tuned model을 비교합니다.
8. Export에서 `LoRA Only`를 선택하고 Hugging Face Hub로 push합니다.

세부 Studio 절차는 [`docs/STUDIO_WORKFLOW.md`](docs/STUDIO_WORKFLOW.md)를 참고하십시오.

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

### Gemini를 사용한 질문 생성

[`generate_questions_with_gemini.ipynb`](generate_questions_with_gemini.ipynb)를 Google Colab에서
실행하면 `squad_contexts_1200.jsonl`을 읽고 새로운 질문을 생성합니다. API key는 숨김 입력으로
전달하며 notebook이나 저장소에 기록하지 않습니다.

## 7. 파인튜닝 설계

| 항목 | 최종 설정 | 근거 |
|---|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` | 3B급 instruct 모델로 Colab T4에서 학습 가능하며 영어 지시 수행 능력이 충분함 |
| Method | QLoRA 4-bit | 4-bit base model을 고정하고 LoRA adapter만 학습하여 VRAM 사용량을 줄임 |
| Epochs | `1` | 데이터가 349개로 작아 과적합을 피하기 위해 1 epoch부터 시작 |
| Learning rate | `2e-4` | LoRA/QLoRA SFT에서 일반적으로 사용하는 시작값 |
| LoRA rank | `16` | 표현력과 메모리 사용량의 균형 |
| LoRA alpha | `32` | `2 x rank` 설정으로 adapter 업데이트 스케일을 확보 |
| Dropout | `0.0` | Unsloth 기본 권장값을 사용하여 빠른 학습과 재현성을 우선 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | attention과 MLP를 함께 조정하여 질문 생성 행동을 충분히 학습 |
| Batch / grad accum | batch size `2`, grad accumulation `8` | effective batch size 16으로 안정적인 gradient를 확보 |
| Context length | `2048` | 긴 passage를 잘라내지 않고 처리하기 위한 길이 |

## 8. 학습 과정

Unsloth Studio에서 QLoRA 학습을 1 epoch 수행했습니다.

| 항목 | 결과 |
|---|---:|
| Train samples | 349 |
| Validation samples | 43 |
| Epoch | 1.00 |
| Steps | 22 / 22 |
| Final training loss | 0.7436 |
| Validation loss | 약 0.65까지 감소 |
| Elapsed time | 약 3분 40초 |
| Export | LoRA Only, Hugging Face Hub |

Training loss와 evaluation loss 모두 전반적으로 감소했습니다. 데이터 크기가 작기 때문에 epoch를 늘리면
훈련셋 문항 스타일에 과적합될 가능성이 있어 1 epoch에서 중단했습니다. Validation loss도 감소했으므로
심한 과소적합보다는, 작은 데이터셋에서 출력 형식을 학습한 상태로 판단했습니다.

## 9. Base Model vs Fine-tuned Model 비교

학습에 사용하지 않은 test split의 passage 5개와 무관한 일반 질문 2개를 사용해 Unsloth Studio
Model Arena에서 base model과 fine-tuned model을 비교했습니다. 전체 결과는
[`results/comparison.csv`](results/comparison.csv)에 기록했습니다.

| 평가 항목 | Base Model | Fine-tuned Model | 변화 |
|---|---:|---:|---|
| 단일 질문 준수율 | 5/5 (100%) | 5/5 (100%) | 동일 |
| 관련성 평균 점수 | 5.0 | 4.8 | base가 약간 높음 |
| 답변 가능성 평균 점수 | 5.0 | 4.6 | base가 약간 높음 |
| 자연스러움 평균 점수 | 4.6 | 4.8 | fine-tuned가 약간 높음 |

무관한 질문 2개(`What is the capital of France?`, `Explain photosynthesis in one sentence.`)에서는 두 모델
모두 정상적으로 일반 질문에 답했습니다. 따라서 이번 짧은 학습에서는 fine-tuned model이 quiz generator
형식에 과도하게 고정되어 일반 능력을 잃는 현상은 뚜렷하게 관찰되지 않았습니다.

정성적으로는 fine-tuned model이 일부 샘플에서 더 간결하고 구체적인 질문을 생성했습니다. 예를 들어
uranium passage에서는 fine-tuned output이 `in 2012`를 포함해 더 구체적인 질문을 만들었습니다. 반면
Maslow passage에서는 base model이 더 의미 있는 질문을 생성했고, fine-tuned output은 다소 표면적인
질문을 생성했습니다.

따라서 본 실험의 결론은 fine-tuned model이 모든 샘플에서 base model을 압도했다는 것이 아니라,
작은 데이터셋과 1 epoch 학습으로도 요구한 단일 질문 출력 형식을 안정적으로 유지했고, 일부 샘플에서
문장 유창성과 구체성이 개선되었다는 것입니다.

## 10. Export 및 실행 결과

학습된 LoRA adapter는 Hugging Face Hub에 업로드했습니다.

- Adapter: [jiarui5/context-quiz-qwen2.5-3b-lora](https://huggingface.co/jiarui5/context-quiz-qwen2.5-3b-lora)
- Export method: LoRA Only
- Base model: `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`

본 저장소의 실행 증거:

- 데이터 생성 실행 화면: [`images/data-recipe-test-20.png`](images/data-recipe-test-20.png),
  [`images/data-recipe-full-run-1200.png`](images/data-recipe-full-run-1200.png)
- 학습 완료 화면: [`images/training-complete.png`](images/training-complete.png)
- 학습 및 eval loss 화면: [`images/training-eval-loss.png`](images/training-eval-loss.png)
- Model Arena 비교 화면:
  [`01`](images/model-arena-01-campus.png),
  [`02`](images/model-arena-02-maslow.png),
  [`03`](images/model-arena-03-uneven-bars.png),
  [`04`](images/model-arena-04-dollar.png),
  [`05`](images/model-arena-05-uranium.png)
- 망각 체크 화면: [`images/forgetting-check.png`](images/forgetting-check.png)
- Hugging Face export 화면: [`images/huggingface-export.png`](images/huggingface-export.png)
- Base vs Fine-tuned 비교 결과: [`results/comparison.csv`](results/comparison.csv)

학습 완료, loss curve, eval loss, Model Arena, Hugging Face export 화면을 저장소에 포함했습니다.

## 11. 한계점 및 개선 방향

- 평가 샘플이 5개로 작아 통계적으로 강한 결론을 내리기 어렵습니다.
- 생성 데이터가 합성 질문을 포함하므로 사람 검수 또는 복수 평가자 검증이 더 필요합니다.
- Base model 자체가 이미 지시 수행 능력이 좋아 개선 폭이 크지 않았습니다.
- 향후 test sample 수를 늘리고, 무관한 질문을 포함한 망각 체크를 추가해야 합니다.
- GGUF export 또는 Ollama 실행까지 확장하면 로컬 배포 증거를 더 강화할 수 있습니다.

## 12. 참고문헌

- [Unsloth Studio](http://unsloth.ai/docs/new/studio)
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

## 프로젝트 구조

```text
.
├── README.md
├── final_report.md
├── generate_questions_with_gemini.ipynb
├── requirements.txt
├── data/
├── docs/
├── images/
├── results/
└── scripts/
```

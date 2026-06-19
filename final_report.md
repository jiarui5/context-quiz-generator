# 기말 프로젝트 보고서

## 1. 선택한 시나리오 및 문제 정의

본 프로젝트는 기말 프로젝트 시나리오 중 **11. 퀴즈/문제 생성기**를 선택했다. 문제는 긴 passage를
읽고, passage 안의 정보만으로 답할 수 있는 자연스러운 quiz question을 정확히 하나 생성하는 것이다.

일반적인 instruct model도 질문을 만들 수는 있지만, 때로는 여러 질문을 생성하거나, 정답과 해설을 함께
출력하거나, passage에서 직접 답하기 어려운 질문을 만들 수 있다. 따라서 본 프로젝트의 목표는
`passage -> exactly one answerable quiz question` 형식을 더 안정적으로 따르는 모델을 만드는 것이다.

## 2. 서비스 대상 사용자

대상 사용자는 다음과 같다.

- 긴 교육 자료에서 복습 문제를 빠르게 만들고 싶은 교사
- 학습한 지문을 바탕으로 self-quiz를 만들고 싶은 학생
- Wikipedia나 교재 기반 학습 콘텐츠를 제작하는 교육 콘텐츠 제작자

이 사용자는 답변 생성보다 "좋은 질문 하나"를 필요로 한다. 그래서 출력 형식은 정답, 선택지, 해설 없이
질문 하나만 나오도록 설계했다.

## 3. Base Model 선택과 근거

Base model은 `unsloth/Qwen2.5-3B-Instruct`를 선택했다.

선택 근거는 다음과 같다.

- 3B급 모델이라 Google Colab T4 환경에서 QLoRA 학습이 가능하다.
- Instruct model이므로 기본적인 지시 이해 능력이 있다.
- 본 프로젝트는 새로운 지식을 주입하는 것이 아니라, passage 기반 질문 생성 형식을 안정화하는 작업이다.
- 0.5B나 1.5B보다 기본 언어 능력이 높고, 7B/8B보다 Colab VRAM 부담이 작다.

## 4. 데이터셋 설계

원천 데이터는 SQuAD의 `context`이다. 원본 SQuAD의 `question`과 `answers`를 그대로 사용하지 않고,
context만 원천 자료로 사용했다. 각 context에 대해 새 quiz question을 생성하고, 형식 검수 후 ChatML
대화 형식으로 변환했다.

최종 데이터 규모는 다음과 같다.

| Split | Samples |
|---|---:|
| Train | 349 |
| Validation | 43 |
| Test | 45 |

데이터 형식은 다음과 같다.

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

`user`에는 지시문과 passage를 넣고, `assistant`에는 모델이 따라야 할 이상적인 quiz question 하나만
넣었다. 이 구조는 base model의 chat template과 role tagging에 맞춰 Unsloth Studio에서 `messages`
컬럼을 ChatML conversation으로 인식하도록 설계한 것이다.

## 5. 하이퍼파라미터 설정과 근거

학습은 Unsloth Studio에서 QLoRA로 수행했다.

| 항목 | 설정 | 근거 |
|---|---|---|
| Method | QLoRA 4-bit | 제한된 Colab T4 VRAM에서 3B model을 효율적으로 학습하기 위함 |
| Epochs | 1 | 데이터가 작아 과적합을 피하기 위해 1 epoch로 제한 |
| Learning rate | 2e-4 | LoRA/QLoRA SFT의 일반적인 시작값 |
| LoRA rank | 16 | 메모리 사용량과 표현력의 균형 |
| LoRA alpha | 32 | rank의 2배로 adapter 업데이트 스케일을 확보 |
| Dropout | 0.0 | Unsloth 기본 권장 설정을 사용 |
| Target modules | q/k/v/o, gate/up/down projections | attention과 MLP를 함께 조정해 질문 생성 행동을 학습 |
| Batch size | 2 | T4 VRAM 제약 고려 |
| Gradient accumulation | 8 | effective batch size 16 확보 |
| Context length | 2048 | 긴 passage를 처리하기 위한 길이 |

LoRA는 전체 모델을 모두 학습하지 않고 작은 adapter만 학습하므로, 작은 데이터셋에서 기존 언어 능력을
크게 훼손하지 않으면서 출력 행동을 조정하기에 적합하다. QLoRA는 4-bit 양자화를 통해 VRAM 사용량을
줄이므로 Colab 환경에 적합하다.

## 6. 학습 과정

학습은 Unsloth Studio on Google Colab T4 GPU에서 수행했다. `train.jsonl`을 training dataset으로,
`validation.jsonl`을 eval dataset으로 업로드했다.

학습 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| Train samples | 349 |
| Validation samples | 43 |
| Epoch | 1.00 |
| Steps | 22 / 22 |
| Final training loss | 0.7436 |
| Validation loss | 약 0.65까지 감소 |
| Training time | 약 3분 40초 |

Training loss는 초반보다 낮아졌고, evaluation loss도 약 1.0 수준에서 약 0.65 수준까지 감소했다.
데이터셋이 작기 때문에 epoch를 더 늘리면 특정 질문 스타일에 과적합될 수 있다고 판단했다. 따라서
1 epoch에서 멈추고, test split으로 base model과 fine-tuned model을 비교했다.

학습 완료 화면은 `images/training-complete.png`에, loss curve는 `images/training-eval-loss.png`에
저장했다.

## 7. 구현 결과: Base Model vs Fine-tuned Model 비교

학습에 사용하지 않은 test split의 passage 5개를 사용해 Unsloth Studio Model Arena에서 base model과
fine-tuned model을 비교했다. 평가 기준은 다음과 같다.

- 정확히 하나의 질문만 출력하는가
- passage와 관련 있는가
- passage만으로 답할 수 있는가
- 영어가 자연스럽고 문법적으로 올바른가

정량 결과는 다음과 같다.

| 평가 항목 | Base Model | Fine-tuned Model |
|---|---:|---:|
| 단일 질문 준수율 | 5/5 (100%) | 5/5 (100%) |
| 관련성 평균 점수 | 5.0 | 4.8 |
| 답변 가능성 평균 점수 | 5.0 | 4.6 |
| 자연스러움 평균 점수 | 4.6 | 4.8 |

세부 결과는 `results/comparison.csv`에 기록했다. Model Arena 비교 화면은
`images/model-arena-01-campus.png`부터 `images/model-arena-05-uranium.png`까지 5개 이미지로
저장했다.

정성적으로는 다음과 같은 결과가 관찰되었다.

- 두 모델 모두 5개 샘플에서 단일 질문 형식을 지켰다.
- Fine-tuned model은 일부 샘플에서 더 간결하거나 구체적인 질문을 생성했다.
- Base model은 일부 샘플에서 더 의미 있는 질문을 생성했다.
- Fine-tuned model의 개선은 압도적이라기보다 제한적이며, 작은 데이터셋과 짧은 학습의 영향이 있다.

예를 들어 uranium passage에서 fine-tuned model은 `in 2012`를 포함해 더 구체적인 질문을 만들었다.
반면 Maslow passage에서는 base model이 더 의미 있는 질문을 만들었고, fine-tuned model은 표면적인
질문을 생성했다.

## 8. Export 및 배포

학습 결과는 LoRA adapter로 export하고 Hugging Face Hub에 업로드했다.

- Hugging Face Hub: <https://huggingface.co/jiarui5/context-quiz-qwen2.5-3b-lora>
- Export method: LoRA Only
- Base model: `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`

본 adapter는 base model과 함께 로드하여 fine-tuned model의 출력을 재현할 수 있다. 이번 제출에서는
LoRA adapter export와 Hub 업로드를 완료했으며, 향후 GGUF export와 Ollama 실행까지 확장할 수 있다.
Export 완료 화면은 `images/huggingface-export.png`에 저장했다.

## 9. 한계점 및 개선 방향

본 프로젝트의 한계는 다음과 같다.

- 비교 샘플이 5개이므로 평가 규모가 작다.
- Base model 자체가 이미 좋은 질문을 생성할 수 있어 개선 폭이 작았다.
- Fine-tuned model이 모든 샘플에서 base model보다 우수하지는 않았다.
- 무관한 일반 질문을 통한 망각 체크가 추가로 필요하다.
- LoRA adapter export는 완료했지만, GGUF/Ollama 로컬 실행 증거는 추가 보완이 가능하다.

개선 방향은 다음과 같다.

- test sample을 20개 이상으로 늘려 평가 신뢰도를 높인다.
- 질문 난이도, 질문 유형, 답변 가능성 등을 명시한 더 정교한 데이터셋을 만든다.
- 무관한 일반 질문을 섞어 catastrophic forgetting 여부를 확인한다.
- 필요하면 2 epoch 또는 rank 32 실험을 추가하고 validation loss와 output quality를 비교한다.
- GGUF export를 추가하여 Ollama 또는 llama.cpp에서 로컬 실행을 시연한다.

## 10. 참고문헌

- Unsloth Studio: <http://unsloth.ai/docs/new/studio>
- Unsloth Fine-tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>
- LoRA: Hu et al., 2021, <https://arxiv.org/abs/2106.09685>
- QLoRA: Dettmers et al., 2023, <https://arxiv.org/abs/2305.14314>
- SQuAD: Rajpurkar et al., 2016, <https://arxiv.org/abs/1606.05250>
- LIMA: Zhou et al., 2023, <https://arxiv.org/abs/2305.11206>

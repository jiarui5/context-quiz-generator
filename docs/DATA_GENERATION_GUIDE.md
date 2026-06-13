# 새 질문 생성 및 품질 검수 가이드

## 입력 데이터

`scripts/prepare_squad_contexts.py`가 출력하는 각 레코드는 다음 필드를 포함한다.

```json
{"context_id":"...","title":"...","context":"...","word_count":150}
```

## 생성할 필드

각 레코드에 `generated_question`을 생성하고 `context_id`와 `context`를 유지한다.

```json
{"context_id":"...","context":"...","generated_question":"...?"}
```

## 질문 생성 규칙

1. 영어 질문을 정확히 하나 생성한다.
2. 질문은 context만으로 답할 수 있어야 한다.
3. 정답, 선택지, 설명, 접두어를 출력하지 않는다.
4. SQuAD의 기존 question을 그대로 복사하지 않는다.
5. 주변적인 세부 사항보다 텍스트의 핵심 사실을 질문한다.
6. 질문은 `?`로 끝나야 한다.

권장 생성 프롬프트:

```text
Read the passage and generate exactly one meaningful quiz question that can
be answered using only the passage. Focus on an important fact. Output only
the question and do not include the answer.

Passage:
{context}
```

## 사람 기반 표본 품질 검수

최소 100개 샘플을 무작위로 검사하고 다음 항목을 기록한다.

- 질문이 정확히 하나인지
- 텍스트와 관련 있는지
- 텍스트만으로 답할 수 있는지
- 자연스럽고 명확한지
- 정답을 노출하지 않는지

오류율이 높으면 바로 학습하지 않고 생성 프롬프트를 수정하여 다시 생성한다.


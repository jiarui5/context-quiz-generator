# 데이터 디렉터리

본 프로젝트는 SQuAD의 기존 질문과 정답을 직접 학습하지 않고 `context`만 원천 텍스트로 사용한다.

| 디렉터리 | 내용 |
|---|---|
| `raw/` | 원본 SQuAD 파일. 용량 때문에 GitHub에 업로드하지 않음 |
| `processed/` | 중복 제거 및 길이 필터링을 적용한 context |
| `interim/` | 새로 생성한 질문과 품질 검수 전 데이터 |
| `final/` | Unsloth 학습용 train/validation/test JSONL |

대용량 원본 및 전체 학습 데이터는 저작권과 저장소 용량을 고려하여 업로드하지 않는다.
대신 생성 방법, 전처리 스크립트, 소규모 예시 데이터를 제공한다.

`generate_questions_with_gemini.ipynb`는 질문 생성 진행 상황을
`generated_questions_progress.jsonl`에 즉시 저장하고, 완료 후 `final/` 데이터 분할을 생성한다.

## 최종 데이터 통계

| 항목 | 개수 |
|---|---:|
| Data Recipes 생성 결과 | 437 |
| 자동 품질 검사 통과 | 437 |
| train | 349 |
| validation | 43 |
| test | 45 |

생성 데이터 출처:
[jiarui5/squad-context-quiz-generated](https://huggingface.co/datasets/jiarui5/squad-context-quiz-generated)

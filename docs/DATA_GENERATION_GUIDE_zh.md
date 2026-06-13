# 新问题生成与质检指南（中文）

## 输入数据

`scripts/prepare_squad_contexts.py` 输出的每条记录只有：

```json
{"context_id":"...","title":"...","context":"...","word_count":150}
```

## 需要生成的新字段

为每条记录生成 `generated_question`，并保留 `context_id` 与 `context`：

```json
{"context_id":"...","context":"...","generated_question":"...?"}
```

## 问题生成规则

1. 只能生成一道英文问题。
2. 问题必须能仅依靠 context 回答。
3. 不输出答案、选项、解释或前缀。
4. 不直接照抄 SQuAD 原有 question。
5. 优先询问文本中的关键事实，而不是边缘细节。
6. 问题结尾必须是 `?`。

推荐生成提示词：

```text
Read the passage and generate exactly one meaningful quiz question that can
be answered using only the passage. Focus on an important fact. Output only
the question and do not include the answer.

Passage:
{context}
```

## 人工抽样质检

至少随机检查 100 条，并记录：

- 是否只有一道问题
- 是否与文本相关
- 是否可以依据文本回答
- 是否自然、明确
- 是否泄露答案

如果错误率较高，应修改生成提示词后重新生成，而不是直接进入训练。


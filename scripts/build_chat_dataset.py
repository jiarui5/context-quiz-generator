#!/usr/bin/env python3
"""Validate generated questions and convert them to Unsloth chat JSONL splits."""

import argparse
import json
import random
from pathlib import Path

INSTRUCTION = (
    "Read the passage and generate exactly one meaningful quiz question that can "
    "be answered using only the passage. Output only the question."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}") from error


def validate(row):
    context = " ".join(row.get("context", "").split())
    question = " ".join(row.get("generated_question", "").split())
    errors = []

    if len(context.split()) < 50:
        errors.append("context_too_short")
    if not question:
        errors.append("missing_question")
    if question and not question.endswith("?"):
        errors.append("not_a_question")
    if question.count("?") != 1:
        errors.append("not_exactly_one_question")
    if len(question.split()) < 5:
        errors.append("question_too_short")
    if len(question.split()) > 40:
        errors.append("question_too_long")

    return context, question, errors


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    valid_rows = []
    rejected_rows = []
    seen_contexts = set()

    for row in read_jsonl(args.input):
        context, question, errors = validate(row)
        if context in seen_contexts:
            errors.append("duplicate_context")

        if errors:
            rejected_rows.append({**row, "quality_errors": errors})
            continue

        seen_contexts.add(context)
        valid_rows.append(
            {
                "context_id": row.get("context_id"),
                "messages": [
                    {
                        "role": "user",
                        "content": f"{INSTRUCTION}\n\nPassage:\n{context}",
                    },
                    {"role": "assistant", "content": question},
                ],
            }
        )

    random.Random(args.seed).shuffle(valid_rows)
    train_end = int(len(valid_rows) * args.train_ratio)
    validation_end = train_end + int(len(valid_rows) * args.validation_ratio)

    splits = {
        "train": valid_rows[:train_end],
        "validation": valid_rows[train_end:validation_end],
        "test": valid_rows[validation_end:],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", rows)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected_rows)

    print(f"valid samples: {len(valid_rows)}")
    print(f"rejected samples: {len(rejected_rows)}")
    for split_name, rows in splits.items():
        print(f"{split_name}: {len(rows)}")


if __name__ == "__main__":
    main()


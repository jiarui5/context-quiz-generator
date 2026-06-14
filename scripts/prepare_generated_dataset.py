#!/usr/bin/env python3
"""Clean generated questions and create deterministic Unsloth chat splits."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

INSTRUCTION = (
    "Read the passage and generate exactly one meaningful quiz question that can "
    "be answered using only the passage. Output only the question."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def clean_text(text):
    return " ".join(text.strip().split())


def clean_question(text):
    text = clean_text(text)
    return re.sub(r"^(question|quiz question)\s*:\s*", "", text, flags=re.I)


def quality_errors(context, question):
    errors = []
    words = question.split()

    if not context:
        errors.append("missing_context")
    if not question:
        errors.append("missing_question")
    if question and not question.endswith("?"):
        errors.append("missing_question_mark")
    if question.count("?") != 1:
        errors.append("not_exactly_one_question")
    if len(words) < 5:
        errors.append("question_too_short")
    if len(words) > 45:
        errors.append("question_too_long")
    if re.match(r"^(i am|i'm|sure|here is|here's|please provide)", question, re.I):
        errors.append("assistant_chatter")

    return errors


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    source_rows = pq.read_table(args.input).to_pylist()
    accepted = []
    rejected = []
    seen_contexts = set()
    seen_questions = set()

    for row in source_rows:
        context = clean_text(row.get("context", ""))
        question = clean_question(row.get("generated_question", ""))
        errors = quality_errors(context, question)

        if context in seen_contexts:
            errors.append("duplicate_context")
        if question.lower() in seen_questions:
            errors.append("duplicate_question")

        cleaned = {
            "context_id": row.get("context_id"),
            "title": row.get("title"),
            "context": context,
            "generated_question": question,
        }
        if errors:
            rejected.append({**cleaned, "quality_errors": errors})
            continue

        seen_contexts.add(context)
        seen_questions.add(question.lower())
        accepted.append(cleaned)

    random.Random(args.seed).shuffle(accepted)
    train_end = int(len(accepted) * 0.8)
    validation_end = train_end + int(len(accepted) * 0.1)
    raw_splits = {
        "train": accepted[:train_end],
        "validation": accepted[train_end:validation_end],
        "test": accepted[validation_end:],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in raw_splits.items():
        chat_rows = [
            {
                "context_id": row["context_id"],
                "messages": [
                    {
                        "role": "user",
                        "content": f"{INSTRUCTION}\n\nPassage:\n{row['context']}",
                    },
                    {"role": "assistant", "content": row["generated_question"]},
                ],
            }
            for row in rows
        ]
        write_jsonl(args.output_dir / f"{split_name}.jsonl", chat_rows)

    write_jsonl(args.output_dir / "accepted_with_sources.jsonl", accepted)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)

    reasons = Counter(error for row in rejected for error in row["quality_errors"])
    stats = {
        "source_records": len(source_rows),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "splits": {name: len(rows) for name, rows in raw_splits.items()},
        "rejection_reasons": reasons,
    }
    (args.output_dir / "quality_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

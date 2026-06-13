#!/usr/bin/env python3
"""Extract unique, sufficiently long contexts from an official SQuAD JSON file."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def normalize(text):
    return " ".join(text.split())


def main():
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    unique = {}

    for article in source["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = normalize(paragraph["context"])
            word_count = len(context.split())
            if args.min_words <= word_count <= args.max_words:
                unique.setdefault(
                    context,
                    {
                        "context_id": f"context-{len(unique):06d}",
                        "title": title,
                        "context": context,
                        "word_count": word_count,
                    },
                )

    rows = list(unique.values())
    random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"unique contexts after filtering: {len(rows)}")
    if rows:
        counts = [row["word_count"] for row in rows]
        print(f"word count range: {min(counts)}-{max(counts)}")
        print(f"average word count: {sum(counts) / len(counts):.1f}")


if __name__ == "__main__":
    main()


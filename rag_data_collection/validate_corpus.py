"""Report corpus quality problems before embedding."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def validate(path: Path, strict: bool) -> int:
    ids, texts, problems, total = set(), set(), Counter(), 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems["invalid_json"] += 1
                continue
            chunk_id = record.get("chunk_id") or record.get("id")
            text = record.get("text", "")
            normalised = normalise_text(text) if isinstance(text, str) else ""
            if not chunk_id or not normalised:
                problems["missing_id_or_text"] += 1
                continue
            if chunk_id in ids:
                problems["duplicate_chunk_id"] += 1
            ids.add(chunk_id)
            if normalised in texts:
                problems["duplicate_text"] += 1
            texts.add(normalised)
            tokens = normalised.split()
            if len(tokens) < 20:
                problems["too_short"] += 1
            if len(set(tokens)) <= 2 and len(tokens) >= 10:
                problems["repeated_boilerplate"] += 1

    print(f"Corpus rows: {total}")
    print(f"Unique IDs: {len(ids)}; unique texts: {len(texts)}")
    if problems:
        print("Problems:")
        for name, count in sorted(problems.items()):
            print(f"  {name}: {count}")
    else:
        print("No structural or basic-quality problems found.")
    return 1 if strict and problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate PromptBridge corpus.jsonl.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any issue is found.")
    options = parser.parse_args()
    raise SystemExit(validate(Path(options.corpus), options.strict))

"""
Merge multiple JSONL files into one.

Each input line is one JSON record (e.g. one SFT conversation). Blank lines are
skipped and every line is validated as JSON, so a missing trailing newline in
one file can't glue two records together. Optionally de-duplicate by a key.

Usage:
  python merge_jsonl.py \
      --inputs egoexo4d.sft.jsonl egoexolearn.sft.jsonl \
      --output merged.sft.jsonl
  # de-dup by metadata.question_id:
  python merge_jsonl.py --inputs a.jsonl b.jsonl --output m.jsonl --dedup_key question_id
"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSONL files")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input JSONL files, in the order to concatenate")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--dedup_key", type=str, default=None,
                        help="If set, drop records whose value at this key was already "
                             "seen. Looks under top-level and under 'metadata'. "
                             "e.g. question_id")
    return parser.parse_args()


def get_key(record, key):
    if key in record:
        return record[key]
    meta = record.get("metadata") or {}
    return meta.get(key)


def main():
    args = parse_args()
    seen = set()
    written = 0
    per_file = {}
    skipped_blank = 0
    skipped_dup = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.inputs:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        skipped_blank += 1
                        continue
                    record = json.loads(line)  # raises on malformed JSON
                    if args.dedup_key is not None:
                        k = get_key(record, args.dedup_key)
                        if k is not None and k in seen:
                            skipped_dup += 1
                            continue
                        if k is not None:
                            seen.add(k)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    written += 1
            per_file[path] = count

    print("Merge complete.")
    for path, count in per_file.items():
        print(f"  {count:>7} records  <-  {path}")
    print(f"  {'-' * 30}")
    print(f"  {written:>7} records  ->  {args.output}")
    if skipped_blank:
        print(f"  skipped {skipped_blank} blank line(s)")
    if args.dedup_key:
        print(f"  skipped {skipped_dup} duplicate(s) by '{args.dedup_key}'")


if __name__ == "__main__":
    main()

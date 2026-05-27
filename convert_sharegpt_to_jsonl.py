"""
Convert ShareGPT-format JSON to the JSONL format expected by
`generate_training_data_multiturn.py`.

Usage:
  python convert_sharegpt_to_jsonl.py \
      --input /data/wangzhichao/datasets/ShareGPT/ShareGPT_V4.3_unfiltered_cleaned_split.json \
      --output /data/wangzhichao/datasets/ShareGPT/sharegpt_for_adapter.jsonl \
      --system_prompt "You are a helpful assistant." \
      --max_samples 30000
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="ShareGPT JSON (list of {id, conversations: [{from, value}]})")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL, one conversation per line")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful assistant.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap total exported samples; 0 = export all")
    parser.add_argument("--min_assistant_tokens_estimate", type=int, default=20,
                        help="Skip conversations whose total assistant chars < this*4 "
                             "(rough token estimate). Drops trivially short examples.")
    return parser.parse_args()


def normalize_conversation(sharegpt_turns, system_prompt):
    """Convert ShareGPT turn list -> [{role, content}, ...].
    Drops conversations that don't strictly alternate human / gpt."""
    out = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})

    expected = "human"
    for turn in sharegpt_turns:
        role_src = turn.get("from", "").lower()
        value = turn.get("value", "")
        if not value or not isinstance(value, str):
            return None
        if role_src not in ("human", "gpt"):
            continue
        if role_src != expected:
            # Either two consecutive humans or two consecutive gpts -- skip.
            return None
        if role_src == "human":
            out.append({"role": "user", "content": value})
            expected = "gpt"
        else:
            out.append({"role": "assistant", "content": value})
            expected = "human"

    if expected != "human":
        # Last turn was a user turn with no assistant reply; drop it.
        if out and out[-1]["role"] == "user":
            out.pop()
    if not any(t["role"] == "assistant" for t in out):
        return None
    return out


def main():
    args = parse_args()
    print(f"Loading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} raw conversations")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept = 0
    dropped_format = 0
    dropped_short = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for example in data:
            turns = example.get("conversations") or example.get("messages") or []
            conv = normalize_conversation(turns, args.system_prompt)
            if conv is None:
                dropped_format += 1
                continue

            asst_chars = sum(
                len(t["content"]) for t in conv if t["role"] == "assistant"
            )
            if asst_chars < args.min_assistant_tokens_estimate * 4:
                dropped_short += 1
                continue

            record = {
                "question_id": example.get("id", f"sg_{kept}"),
                "conversation": conv,
                "metadata": {"source": "sharegpt"},
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_samples and kept >= args.max_samples:
                break

    print("Done.")
    print(f"  kept           : {kept}")
    print(f"  dropped_format : {dropped_format}")
    print(f"  dropped_short  : {dropped_short}")
    print(f"  output         : {args.output}")


if __name__ == "__main__":
    main()

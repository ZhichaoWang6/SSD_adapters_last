import json
import glob
import os


def safe_div(num, den):
    return num / den if den else 0


def load_jsonl_files(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "summary" not in obj or not obj["summary"]:
                    continue
                row = obj["summary"]
                row["_question_id"] = obj.get("question_id", "")
                row["_source_file"] = path
                row["_line_id"] = line_id
                rows.append(row)
    return rows


def _agg(metric_dicts):
    """Pool accept-length and speedup metrics across questions (or buckets).

    accept_length_with_bonus    = total accepted / total rounds (Kangaroo口径)
    accept_length_without_bonus = (total - rounds) / total rounds (扣掉每轮大模型白送)
    speedup                     = pooled spec tok/s / pooled AR tok/s
    """
    turns = sum(m.get("turns", 0) for m in metric_dicts)
    rounds = sum(m.get("rounds", 0) for m in metric_dicts)
    total = sum(m.get("accept_lengths_sum", 0) for m in metric_dicts)
    spec_tokens = sum(m.get("spec_tokens", 0) for m in metric_dicts)
    ar_tokens = sum(m.get("ar_tokens", 0) for m in metric_dicts)
    spec_decode = sum(m.get("spec_decode_time", 0) for m in metric_dicts)
    ar_decode = sum(m.get("ar_decode_time", 0) for m in metric_dicts)
    spec_tps = safe_div(spec_tokens, spec_decode)
    ar_tps = safe_div(ar_tokens, ar_decode)
    return {
        "turns": turns,
        "rounds": rounds,
        "accept_lengths_sum": total,
        "accept_length_with_bonus": safe_div(total, rounds),
        "accept_length_without_bonus": safe_div(total - rounds, rounds),
        "spec_tokens": spec_tokens,
        "ar_tokens": ar_tokens,
        "spec_decode_time": spec_decode,
        "ar_decode_time": ar_decode,
        "spec_decode_tokens_per_second": round(spec_tps, 2),
        "ar_decode_tokens_per_second": round(ar_tps, 2),
        "speedup": round(safe_div(spec_tps, ar_tps), 4) if ar_tps > 0 else None,
    }


def aggregate_summaries(rows):
    summary = _agg(rows)
    summary["num_questions"] = len(rows)
    summary["short_reply"] = _agg([r["short_reply"] for r in rows if r.get("short_reply")])
    summary["long_reply"] = _agg([r["long_reply"] for r in rows if r.get("long_reply")])
    return summary


def _print_metrics(label, m):
    print(
        f"[{label}] turns={m['turns']} rounds={m['rounds']} sum={m['accept_lengths_sum']} | "
        f"accept_len with_bonus={m['accept_length_with_bonus']:.3f} "
        f"no_bonus={m['accept_length_without_bonus']:.3f} | "
        f"spec {m['spec_decode_tokens_per_second']:.1f} tok/s | "
        f"AR {m['ar_decode_tokens_per_second']:.1f} tok/s | "
        f"speedup {m['speedup'] or 0:.2f}x"
    )


def print_generation_summary(title, summary):
    print(f"\n--- {title} ({summary.get('num_questions', 0)} questions) ---")
    _print_metrics("Overall", summary)
    for label, key in (("Short/NO_REPLY <=5 tok", "short_reply"), ("Long >5 tok", "long_reply")):
        bucket = summary.get(key)
        if bucket and bucket["turns"]:
            _print_metrics(label, bucket)


if __name__ == "__main__":
    # 改成你的输出 jsonl 文件路径
    jsonl_paths = glob.glob("/data/wangzhichao/projects/SSD_adapters/outputs/text_4-3_0.6_stage2_1.jsonl")

    rows = load_jsonl_files(jsonl_paths)
    print(f"Loaded {len(rows)} question summaries from {len(jsonl_paths)} jsonl files.")

    summary = aggregate_summaries(rows)
    print_generation_summary("AGGREGATE RESULTS", summary)

    with open("aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved to aggregate_summary.json")

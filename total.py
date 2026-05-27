import json
import glob
import os


SHORT_REPLY_MAX_TOKENS = 5


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


def aggregate_bucket(rows, bucket_name):
    buckets = [r.get(bucket_name, {}) for r in rows if r.get(bucket_name)]

    turns = sum(b.get("turns", 0) for b in buckets)
    tokens = sum(b.get("tokens", 0) for b in buckets)
    rounds = sum(b.get("rounds", 0) for b in buckets)

    # Pool the per-round accepted-length lists across all questions for this
    # bucket, matching Kangaroo's np.mean(accept_lengths_list).
    accept_lengths = [a for b in buckets for a in b.get("accept_lengths", [])]
    accept_lengths_sum = sum(accept_lengths)

    adapter_correct = sum(b.get("adapter_correct", 0) for b in buckets)
    adapter_total = sum(b.get("adapter_total", 0) for b in buckets)
    adapter_first_correct = sum(b.get("adapter_first_correct", 0) for b in buckets)
    adapter_first_total = sum(b.get("adapter_first_total", 0) for b in buckets)

    # 按你代码里的逻辑，confidence 优先按 adapter_total 加权
    conf_num = 0
    conf_den = 0
    for b in buckets:
        w = b.get("adapter_total", 0)
        if w > 0:
            conf_num += b.get("avg_confidence", 0) * w
            conf_den += w

    avg_confidence = safe_div(conf_num, conf_den)

    return {
        "turns": turns,
        "tokens": tokens,
        "rounds": rounds,
        "accept_lengths_sum": accept_lengths_sum,
        "accept_length_pooled": safe_div(accept_lengths_sum, rounds),
        "progress_per_round": safe_div(
            sum(b.get("progress_per_round", 0) * b.get("rounds", 0) for b in buckets),
            rounds
        ),
        "draft_accept_per_round": safe_div(adapter_correct, rounds),
        "adapter_correct": adapter_correct,
        "adapter_total": adapter_total,
        "adapter_accuracy": safe_div(adapter_correct, adapter_total),
        "adapter_first_correct": adapter_first_correct,
        "adapter_first_total": adapter_first_total,
        "adapter_first_accuracy": safe_div(adapter_first_correct, adapter_first_total),
        "avg_confidence": avg_confidence,
    }


def aggregate_summaries(rows):
    num_turns = sum(r.get("num_turns", 0) for r in rows)

    spec_tokens = sum(r.get("spec_tokens", 0) for r in rows)
    ar_tokens = sum(r.get("ar_tokens", 0) for r in rows)

    spec_decode_time = sum(r.get("spec_decode_time", 0) for r in rows)
    ar_decode_time = sum(r.get("ar_decode_time", 0) for r in rows)

    spec_tps = safe_div(spec_tokens, spec_decode_time)
    ar_tps = safe_div(ar_tokens, ar_decode_time)

    rounds = sum(r.get("rounds", 0) for r in rows)

    adapter_correct = sum(r.get("adapter_correct", 0) for r in rows)
    adapter_total = sum(r.get("adapter_total", 0) for r in rows)
    adapter_first_correct = sum(r.get("adapter_first_correct", 0) for r in rows)
    adapter_first_total = sum(r.get("adapter_first_total", 0) for r in rows)

    lossless_matches = sum(r.get("lossless_matches", 0) for r in rows)
    lossless_total = sum(r.get("lossless_total", 0) for r in rows)

    fallback_to_ar = sum(r.get("history_fallback_to_ar", 0) for r in rows)

    auto_must_reply_turns = sum(r.get("auto_must_reply_turns", 0) for r in rows)
    auto_must_reply_total_tokens = sum(r.get("auto_must_reply_total_tokens", 0) for r in rows)

    auto_context_lens = []
    for r in rows:
        auto_context_lens.extend(r.get("auto_must_reply_context_lens", []))

    summary = {
        "num_questions": len(rows),
        "num_turns": num_turns,

        "spec_tokens": spec_tokens,
        "ar_tokens": ar_tokens,
        "spec_decode_time": spec_decode_time,
        "ar_decode_time": ar_decode_time,

        "spec_decode_tokens_per_second": round(spec_tps, 2),
        "ar_decode_tokens_per_second": round(ar_tps, 2),
        "actual_decode_speedup": round(safe_div(spec_tps, ar_tps), 4) if ar_tps > 0 else None,

        "match_rate": safe_div(lossless_matches, lossless_total),
        "lossless_matches": lossless_matches,
        "lossless_total": lossless_total,
        "history_fallback_to_ar": fallback_to_ar,

        "rounds": rounds,

        # 对已经保存的 question-level summary，只能按 num_turns 加权还原整体均值
        "avg_accept_length": safe_div(
            sum(r.get("avg_accept_length", 0) * r.get("num_turns", 0) for r in rows),
            num_turns
        ),
        "avg_draft_accept_length": safe_div(
            sum(r.get("avg_draft_accept_length", 0) * r.get("num_turns", 0) for r in rows),
            num_turns
        ),

        # 和你原代码一致：progress 按 accept_lengths 总和 / rounds
        "progress_per_round": safe_div(
            sum(r.get("progress_per_round", 0) * r.get("rounds", 0) for r in rows),
            rounds
        ),

        # 和你原代码一致：draft_accept_per_round = adapter_correct / rounds
        "draft_accept_per_round": safe_div(adapter_correct, rounds),

        "adapter_correct": adapter_correct,
        "adapter_total": adapter_total,
        "adapter_first_correct": adapter_first_correct,
        "adapter_first_total": adapter_first_total,

        "adapter_accuracy": safe_div(adapter_correct, adapter_total),
        "adapter_first_accuracy": safe_div(adapter_first_correct, adapter_first_total),

        "auto_must_reply_turns": auto_must_reply_turns,
        "auto_must_reply_context_lens": auto_context_lens,
        "auto_must_reply_total_tokens": auto_must_reply_total_tokens,

        "short_reply": aggregate_bucket(rows, "short_reply"),
        "long_reply": aggregate_bucket(rows, "long_reply"),
    }

    if auto_must_reply_turns > 0:
        summary["auto_must_reply_progress_per_round"] = safe_div(
            sum(
                r.get("auto_must_reply_progress_per_round", 0) * r.get("auto_must_reply_turns", 0)
                for r in rows
                if r.get("auto_must_reply_progress_per_round") is not None
            ),
            auto_must_reply_turns
        )
    else:
        summary["auto_must_reply_progress_per_round"] = None

    return summary


def print_generation_summary(title, summary):
    print(f"\n--- {title} ({summary['num_turns']} turns, {summary['num_questions']} questions) ---")

    print(
        f"Actual decode: Spec {summary['spec_tokens']} tok / {summary['spec_decode_time']:.2f}s = "
        f"{summary['spec_decode_tokens_per_second']:.1f} tok/s | "
        f"AR {summary['ar_tokens']} tok / {summary['ar_decode_time']:.2f}s = "
        f"{summary['ar_decode_tokens_per_second']:.1f} tok/s | "
        f"speedup {summary['actual_decode_speedup'] or 0:.2f}x"
    )

    print(
        f"Progress/round {summary['progress_per_round']:.2f} | "
        f"Draft accepted/round {summary['draft_accept_per_round']:.2f} "
        f"(counts every verified draft token, including EOS)"
    )

    print(
        f"Adapter top1: {summary['adapter_correct']}/{summary['adapter_total']} "
        f"({summary['adapter_accuracy']:.1%}) | "
        f"first draft top1: {summary['adapter_first_correct']}/{summary['adapter_first_total']} "
        f"({summary['adapter_first_accuracy']:.1%})"
    )

    if summary["lossless_total"]:
        print(
            f"Lossless: {summary['lossless_matches']}/{summary['lossless_total']} "
            f"({summary['match_rate']:.1%}) | fallback_to_AR {summary['history_fallback_to_ar']}"
        )

    for label, key in [
        ("Short/NO_REPLY <=5 tok", "short_reply"),
        ("Long >5 tok", "long_reply"),
    ]:
        bucket = summary[key]
        if bucket["turns"]:
            print(
                f"{label}: turns={bucket['turns']} tokens={bucket['tokens']} | "
                f"rounds={bucket['rounds']} sum={bucket.get('accept_lengths_sum', 0)} "
                f"pooled_accept={bucket.get('accept_length_pooled', 0):.2f} | "
                f"progress/round={bucket['progress_per_round']:.2f} | "
                f"draft_accept/round={bucket['draft_accept_per_round']:.2f} | "
                f"adapter={bucket['adapter_correct']}/{bucket['adapter_total']} "
                f"({bucket['adapter_accuracy']:.1%}) | "
                f"first={bucket['adapter_first_correct']}/{bucket['adapter_first_total']} "
                f"({bucket['adapter_first_accuracy']:.1%}) | "
                f"conf={bucket['avg_confidence']:.3f}"
            )


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
"""
Adapter-only (draft-only) autoregressive inference.

Generates text using ONLY the lower base-model layers (0..early_exit_layer-1)
plus the adapter, completely skipping the verify pass. This is FAST but NOT
lossless -- output may differ from what full-model AR would produce.

Use cases:
  - quick MT-Bench style benchmarks of adapter quality + speed
  - smoke tests of whether the adapter learned to speak

For lossless inference use the standard inference.py with speculative
decoding instead.

Usage:
  python adapter_only_inference.py \\
      --model_path /data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt \\
      --adapter_path /path/to/warmup_epoch010/ \\
      --questions /home/user/SSD_adapters/kangaroo_data/question.jsonl \\
      --output /tmp/adapter_only_mtbench.jsonl \\
      --exit_layer 12 \\
      --num_adapter_layers 3 \\
      --max_new_tokens 512 \\
      --max_questions 0
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoProcessor

from kangaroo_model import KangarooQwenModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt",
                    help="Base model path (MMDuet2 / Qwen2.5-VL).")
    ap.add_argument("--adapter_path", type=str, default="/data/wangzhichao/projects/SSD_adapters/adapter_checkpoints/multiturn_text_12-3_with_val_ce/epochs/epoch007_valtop10.6271_valdraft10.7485_vallongdraft10.7485_valoverlap0.5780_valloss1.7882",
                    help="Adapter checkpoint directory (containing adapter_model.bin).")
    ap.add_argument("--questions", type=str,
                    default="/data/wangzhichao/projects/SSD_adapters/question.jsonl",
                    help="MT-Bench style JSONL with 'turns' field.")
    ap.add_argument("--output", type=str, default="./outputs/adapter_only_out.jsonl",
                    help="Where to write generated answers.")
    ap.add_argument("--exit_layer", type=int, default=12)
    ap.add_argument("--num_adapter_layers", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_questions", type=int, default=0,
                    help="Cap on MT-Bench questions to evaluate. 0 = all.")
    ap.add_argument("--all_turns", action="store_true",
                    help="Evaluate every turn in question.jsonl instead of only turns[0].")
    ap.add_argument("--system_prompt", type=str,
                    default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    ap.add_argument("--device", type=str, default="cuda:6")
    ap.add_argument("--ar_baseline", action="store_true",
                    help="Also run a full-model AR baseline per prompt for speed comparison.")
    return ap.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def adapter_only_generate(model, inputs, processor, max_new_tokens):
    """Autoregressive decoding using only lower base layers + adapter.

    Pipeline:
      1. prefill:  base.full_forward(input_ids) -> populates lower-layer KV cache,
                   gives hidden_state at exit_layer for entire input.
      2. adapter prefill on those hidden states -> populates adapter KV cache.
      3. first token = argmax of head(adapter_prefill_output[-1]).
      4. loop:
          - new_token -> base.forward_draft_or_large_model(in_tokens_small)
            -> 1 new exit-layer hidden state, lower-layer KV grows by 1.
          - adapter.forward_early_stop(that hidden_state) -> 1 new adapter output.
          - argmax(head(adapter_output)) -> next token.
          - stop at EOS or max_new_tokens.
    """
    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model
    device = base_model.device

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos = tokenizer.eos_token_id
    eos_set = set(eos) if isinstance(eos, list) else {eos}

    input_ids = inputs["input_ids"]
    batch_size, ctx_len = input_ids.shape
    assert batch_size == 1, "batch_size > 1 not supported"

    # ---- Step 1: prefill the base lower layers (we still run the FULL forward
    # because Qwen2.5-VL's get_rope_index / image-pad embedding logic is hidden
    # inside the top-level forward; running the full thing is the easiest way
    # to populate the lower-layer KV cache correctly). We won't use the upper-
    # layer outputs.
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": inputs.get("attention_mask"),
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "second_per_grid_ts": inputs.get("second_per_grid_ts"),
        "drop_method": "none",
        "drop_threshold": 1.0,
        "drop_absolute": True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    t_prefill = time.perf_counter()
    output = base_model.model(**forward_kwargs)
    base_model.past_key_values = output.past_key_values
    hidden_state_early = output.hidden_states[base_model.early_exit_layer]

    # Adapter prefill on the full early-exit hidden state sequence. Use proper
    # 3D mRoPE positions (the same helper trick we use in inference_kangaroo.py).
    prefill_position_ids, _ = base_model.model.get_rope_index(
        input_ids,
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        inputs.get("second_per_grid_ts"),
        inputs.get("attention_mask"),
    )
    prefill_position_ids = prefill_position_ids.to(device)
    adapter_hidden, adapter_past = adapter_model.forward_early_stop(
        inputs_embeds=hidden_state_early,
        position_ids=prefill_position_ids,
        use_cache=True,
    )

    # First generated token from the adapter's last position.
    first_logits = head_model(adapter_hidden[:, -1:, :]).float()
    next_token = torch.argmax(first_logits[:, -1, :], dim=-1)
    prefill_time = time.perf_counter() - t_prefill

    generated_tokens = [int(next_token.item())]
    if int(next_token.item()) in eos_set:
        return {
            "tokens": generated_tokens,
            "prefill_time": prefill_time,
            "decode_time": 0.0,
            "total_time": prefill_time,
        }

    # ---- Step 2: autoregressive decode (lower layers + adapter, NO verify) ----
    t_decode = time.perf_counter()
    rope_deltas = base_model.model.rope_deltas
    for step in range(1, max_new_tokens):
        # Tokens go to lower base layers (one token at a time).
        in_tok = next_token.unsqueeze(0)  # shape (1, 1)
        hidden_state_early = base_model.forward_draft_or_large_model(
            in_tokens_small=in_tok,
        )  # (1, 1, D)

        # Adapter step. Build proper mRoPE position for this new (text) token.
        adapter_past_len = (
            adapter_past[0][0].shape[2]
            if adapter_past is not None and len(adapter_past) > 0
            else 0
        )
        if rope_deltas is not None:
            delta = (adapter_past_len + rope_deltas).to(device)
        else:
            delta = adapter_past_len
        pos = torch.arange(1, device=device) + delta
        pos = pos.view(1, -1).expand(1, -1)             # (1, 1) before 3D expand
        pos = pos.unsqueeze(0).expand(3, -1, -1)         # (3, 1, 1)

        adapter_hidden, adapter_past = adapter_model.forward_early_stop(
            inputs_embeds=hidden_state_early,
            position_ids=pos,
            past_key_values=adapter_past,
            use_cache=True,
        )
        logits = head_model(adapter_hidden[:, -1:, :]).float()
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated_tokens.append(int(next_token.item()))
        if int(next_token.item()) in eos_set:
            break

    decode_time = time.perf_counter() - t_decode
    return {
        "tokens": generated_tokens,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": prefill_time + decode_time,
    }


@torch.no_grad()
def ar_full_baseline(model, inputs, processor, max_new_tokens):
    """Full-model AR baseline via model.generate() for speed comparison."""
    base_raw_model = model.base_model.model  # underlying Qwen2_5_VLForConditionalGeneration
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    t0 = time.perf_counter()
    out = base_raw_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        drop_method="none",
        drop_threshold=1.0,
        drop_absolute=True,
    )
    elapsed = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    return {
        "tokens": new_ids,
        "total_time": elapsed,
    }


def build_chat_messages(question_turns, system_prompt):
    """MT-Bench is single-turn or multi-turn; build a fresh first-turn message
    list (we only do turn[0] here for simplicity)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_turns[0]},
    ]


def build_chat_messages_from_history(history, system_prompt):
    return [{"role": "system", "content": system_prompt}] + history


def text_quality_stats(text):
    words = text.split()
    lower_words = [w.lower() for w in words]
    unique_ratio = len(set(lower_words)) / max(len(lower_words), 1)

    repeated_bigram = 0
    if len(lower_words) >= 4:
        bigrams = list(zip(lower_words, lower_words[1:]))
        repeated_bigram = len(bigrams) - len(set(bigrams))

    stripped = text.strip()
    return {
        "chars": len(text),
        "words": len(words),
        "unique_word_ratio": unique_ratio,
        "repeated_bigram_count": repeated_bigram,
        "empty": not bool(stripped),
        "looks_truncated": bool(stripped) and stripped[-1] not in ".!?\"')]}",
    }


def main():
    args = parse_args()
    print(f"Loading model from {args.model_path}")
    print(f"Loading adapter from {args.adapter_path}")
    model = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        num_adapter_layers=args.num_adapter_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    print(f"Loading questions from {args.questions}")
    questions = read_jsonl(args.questions)
    if args.max_questions > 0:
        questions = questions[:args.max_questions]
    print(f"Will evaluate {len(questions)} questions")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []
    total_decode_tokens = 0
    total_decode_time = 0.0
    total_ar_tokens = 0
    total_ar_time = 0.0

    for i, q in enumerate(questions):
        history = []
        turn_records = []
        turns_to_run = q["turns"] if args.all_turns else q["turns"][:1]

        print(f"\n--- Question {i+1}/{len(questions)} (id={q.get('question_id')}, cat={q.get('category')}) ---")

        for turn_idx, user_text in enumerate(turns_to_run):
            history.append({"role": "user", "content": user_text})
            if hasattr(model, "reset_status"):
                model.reset_status()
            messages = build_chat_messages_from_history(history, args.system_prompt)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt", padding=True).to(args.device)
            ctx_len = int(inputs["input_ids"].shape[1])

            result = adapter_only_generate(model, inputs, processor, args.max_new_tokens)
            reply_text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
            n_new = len(result["tokens"])
            decode_tps = n_new / result["decode_time"] if result["decode_time"] > 0 else 0.0

            total_decode_tokens += n_new
            total_decode_time += result["decode_time"]

            ar_record = None
            if args.ar_baseline:
                if hasattr(model, "reset_status"):
                    model.reset_status()
                ar = ar_full_baseline(model, inputs, processor, args.max_new_tokens)
                ar_text = tokenizer.decode(ar["tokens"], skip_special_tokens=True)
                ar_n = len(ar["tokens"])
                ar_tps = ar_n / ar["total_time"] if ar["total_time"] > 0 else 0.0
                total_ar_tokens += ar_n
                total_ar_time += ar["total_time"]
                ar_record = {
                    "tokens": ar_n,
                    "total_time": ar["total_time"],
                    "tps": ar_tps,
                    "text": ar_text,
                    "quality": text_quality_stats(ar_text),
                }

            adapter_record = {
                "tokens": n_new,
                "prefill_time": result["prefill_time"],
                "decode_time": result["decode_time"],
                "decode_tps": decode_tps,
                "text": reply_text,
                "quality": text_quality_stats(reply_text),
            }
            turn_record = {
                "turn": turn_idx + 1,
                "question": user_text,
                "ctx_len": ctx_len,
                "adapter_only": adapter_record,
            }
            if ar_record is not None:
                turn_record["ar_baseline"] = ar_record
                turn_record["speedup_decode_tps"] = (
                    decode_tps / ar_record["tps"] if ar_record["tps"] > 0 else None
                )
            turn_records.append(turn_record)
            history.append({"role": "assistant", "content": reply_text})

            print(f"  Turn {turn_idx + 1} Q: {user_text}")
            print(f"  Turn {turn_idx + 1} A (adapter-only, {n_new} tok): {reply_text}")
            if ar_record is not None:
                print(f"  Turn {turn_idx + 1} A (AR full, {ar_record['tokens']} tok): {ar_record['text'][:200]}...")

        record = {
            "question_id": q.get("question_id"),
            "category": q.get("category"),
            "turns": turn_records,
        }
        results.append(record)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    avg_tps = total_decode_tokens / total_decode_time if total_decode_time > 0 else 0
    print(f"Adapter-only: {total_decode_tokens} tokens / {total_decode_time:.2f}s = {avg_tps:.1f} tok/s "
          f"(across {len(questions)} questions)")
    if args.ar_baseline:
        ar_avg_tps = total_ar_tokens / total_ar_time if total_ar_time > 0 else 0
        speedup = avg_tps / ar_avg_tps if ar_avg_tps > 0 else None
        print(f"AR full:       {total_ar_tokens} tokens / {total_ar_time:.2f}s = {ar_avg_tps:.1f} tok/s")
        if speedup:
            print(f"Adapter-only speedup vs AR full: {speedup:.2f}x")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()

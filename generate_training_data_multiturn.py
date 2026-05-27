"""
Generate multi-turn training data for the Kangaroo adapter.

One JSONL line (= one full multi-turn conversation) -> one .ckpt file.

Compared to the per-turn `generate_training_data.py`:
  - 1 forward pass per conversation instead of k (k = #assistant turns)
  - Multi-segment loss_mask: every assistant turn in the conversation
    contributes loss positions simultaneously
  - Optional NO_REPLY down-weighting via --no_reply_mask_weight, so the
    91%-NO_REPLY data imbalance can be tamed without throwing samples away
  - Kept in lock-step with Kangaroo's original ge_data_all_vicuna.py pattern
    (one conversation -> one ckpt with a multi-segment loss_mask)

Saved tensors (same keys as `generate_training_data.py` so train_adapter.py
works without modification):
  - input_ids                  [L]
  - loss_mask                  [L]   float, multi-segment; non-binary if
                                     no_reply weight < 1.0
  - position_ids               [3, L]  (3D mRoPE)
  - hidden_state_layer{N}      [L, D]  per requested early-exit layer
  - hidden_state               [L, D]  final layer
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn adapter training data (one ckpt per conversation)"
    )
    parser.add_argument("--model_path", type=str,
                        default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    parser.add_argument("--data_path", type=str,
                        default="/data/wangzhichao/projects/SSD/SSD3/datasets/egoexolearn_train.json")
    parser.add_argument("--output_dir", type=str,
                        default="/data/wangzhichao/projects/SSD_RE/datasets/training_data/multiturn/")
    parser.add_argument("--exit_layers", type=str, default="2,3,4",
                        help="Comma-separated list of exit layers to save hidden states for")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=32768,
                        help="Skip conversations whose tokenized length exceeds this value")
    parser.add_argument("--no_reply_mask_weight", type=float, default=0.1,
                        help="Loss weight for NO REPLY assistant tokens. "
                             "1.0 = same as real replies; 0.0 = fully ignored; "
                             "0.1 (default) = 10x downweight, keeps a little NO REPLY signal.")
    parser.add_argument("--include_im_end_in_loss", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include each assistant turn's trailing <|im_end|> in the loss. "
                             "Default True (matches Kangaroo's original behaviour and helps "
                             "the adapter learn EOS).")
    parser.add_argument("--min_real_reply_tokens", type=int, default=0,
                        help="Skip conversations with fewer than this many non-NO_REPLY "
                             "assistant tokens. 0 = keep all.")
    parser.add_argument("--gpu", type=str, default="cuda:6")
    return parser.parse_args()


def load_data(data_path):
    if data_path.endswith(".jsonl"):
        rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_assistant_text(turn):
    content = turn.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return " ".join(texts).strip()
    return ""


def build_multiturn_loss_mask(
    input_ids,
    tokenizer,
    include_im_end=True,
    no_reply_mask_weight=0.1,
):
    """Scan an already-tokenized Qwen2.5-VL conversation and return a
    per-token loss mask.

    The mask marks the content of every assistant turn (and optionally that
    turn's trailing <|im_end|>) with weight 1.0, downweighting NO REPLY
    content to `no_reply_mask_weight`. Everything else (system, user, image
    pads, chat-template headers/newlines) gets 0.

    Returns a float tensor of shape [L] and the integer real-reply-token count
    (used for `--min_real_reply_tokens` filtering and ckpt metadata).
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_start_id is not None and im_end_id is not None, \
        "Tokenizer does not have Qwen chat-template special tokens"

    ids = input_ids.tolist()
    L = len(ids)
    mask = torch.zeros(L, dtype=torch.float32)
    real_reply_tokens = 0

    i = 0
    while i < L:
        if ids[i] != im_start_id:
            i += 1
            continue

        # Find the closing <|im_end|> of this block.
        j = i + 1
        while j < L and ids[j] != im_end_id:
            j += 1
        if j >= L:
            break  # unterminated block at the very end; ignore

        # Walk forward token-by-token after <|im_start|> to:
        #   1. determine the role (system / user / assistant)
        #   2. find the position right after the header newline
        # We do this by decoding tokens one at a time and checking text.
        cursor = i + 1
        header_text = ""
        while cursor < j:
            piece = tokenizer.decode([ids[cursor]], skip_special_tokens=False)
            header_text += piece
            cursor += 1
            if "\n" in piece:
                break  # cursor is now at the first content token
        content_start = cursor

        role = header_text.split("\n", 1)[0].strip()
        if role != "assistant":
            i = j + 1
            continue

        content_end = j  # exclusive (j points at <|im_end|>)
        if content_start >= content_end:
            i = j + 1
            continue

        # Decode the content to detect NO REPLY for downweighting.
        content_text = tokenizer.decode(
            ids[content_start:content_end], skip_special_tokens=False,
        ).strip()
        is_no_reply = (content_text == "NO REPLY")
        weight = no_reply_mask_weight if is_no_reply else 1.0

        mask[content_start:content_end] = weight
        if include_im_end:
            mask[content_end] = weight

        if not is_no_reply:
            real_reply_tokens += (content_end - content_start)
            if include_im_end:
                real_reply_tokens += 1

        i = j + 1

    return mask, real_reply_tokens


@torch.no_grad()
def process_conversation(model, processor, conversation, exit_layers, max_seq_len,
                         no_reply_mask_weight, include_im_end_in_loss,
                         min_real_reply_tokens):
    """Run the full multi-turn conversation through one forward pass and
    package everything needed for adapter training into a single dict."""
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    seq_len = inputs.input_ids.shape[1]
    if seq_len > max_seq_len:
        print(f"  [SKIP] seq_len={seq_len} exceeds max_seq_len={max_seq_len}")
        return None

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    loss_mask, real_reply_tokens = build_multiturn_loss_mask(
        inputs.input_ids[0],
        tokenizer,
        include_im_end=include_im_end_in_loss,
        no_reply_mask_weight=no_reply_mask_weight,
    )

    if real_reply_tokens < min_real_reply_tokens:
        print(f"  [SKIP] real_reply_tokens={real_reply_tokens} < "
              f"min={min_real_reply_tokens}")
        return None
    if float(loss_mask.sum()) <= 0:
        print("  [SKIP] loss_mask is all zero")
        return None

    forward_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "second_per_grid_ts": inputs.get("second_per_grid_ts"),
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
        "drop_method": "none",
        "drop_threshold": 1.0,
        "drop_absolute": True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    try:
        outputs = model(**forward_kwargs)
    except Exception as exc:
        print(f"  [ERROR] forward pass failed: {exc}")
        return None

    result = {
        "input_ids": inputs["input_ids"].cpu()[0],
        "loss_mask": loss_mask,
        "seq_len": int(seq_len),
        "answer_tokens": int((loss_mask > 0).sum().item()),
        "real_reply_tokens": int(real_reply_tokens),
        "loss_mask_sum_weighted": float(loss_mask.sum().item()),
    }

    try:
        position_ids, _ = model.get_rope_index(
            inputs["input_ids"],
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            inputs.get("second_per_grid_ts"),
            inputs.get("attention_mask"),
        )
        result["position_ids"] = position_ids.cpu()[:, 0]  # (3, L)
    except Exception as exc:
        print(f"  [WARN] get_rope_index failed: {exc}; saving without position_ids")

    for layer in exit_layers:
        if layer < len(outputs.hidden_states):
            result[f"hidden_state_layer{layer}"] = outputs.hidden_states[layer].float().cpu()[0]

    result["hidden_state"] = outputs.hidden_states[-1].float().cpu()[0]

    for key, value in result.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"  [SKIP] {key} contains NaN/Inf")
                return None

    return result


def main():
    args = parse_args()
    exit_layers = [int(x) for x in args.exit_layers.split(",") if x.strip()]

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval().to(args.gpu)

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)

    end = args.end if args.end is not None else len(data)
    data = data[args.start:end]
    print(f"Processing {len(data)} samples (index {args.start}..{end})")

    os.makedirs(args.output_dir, exist_ok=True)

    total_seen = 0
    total_saved = 0
    total_skipped = 0
    total_real_reply_tokens = 0
    total_no_reply_tokens = 0

    for sample_i, example in enumerate(tqdm(data)):
        total_seen += 1
        global_idx = args.start + sample_i
        conversation = example.get("conversation", example.get("messages", []))
        if not conversation:
            total_skipped += 1
            continue

        result = process_conversation(
            model=model,
            processor=processor,
            conversation=conversation,
            exit_layers=exit_layers,
            max_seq_len=args.max_seq_len,
            no_reply_mask_weight=args.no_reply_mask_weight,
            include_im_end_in_loss=args.include_im_end_in_loss,
            min_real_reply_tokens=args.min_real_reply_tokens,
        )
        if result is None:
            total_skipped += 1
            continue

        save_path = os.path.join(args.output_dir, f"data_{global_idx}.ckpt")
        torch.save(result, save_path)
        total_saved += 1
        total_real_reply_tokens += result["real_reply_tokens"]
        total_no_reply_tokens += result["answer_tokens"] - result["real_reply_tokens"]

    print("\nDone.")
    print(f"  Total conversations seen   : {total_seen}")
    print(f"  Saved                      : {total_saved}")
    print(f"  Skipped                    : {total_skipped}")
    print(f"  Total real-reply tokens    : {total_real_reply_tokens}")
    print(f"  Total NO_REPLY tokens      : {total_no_reply_tokens}")
    print(f"  Output dir                 : {args.output_dir}")
    print(f"  Exit layers                : {exit_layers}")
    print(f"  no_reply_mask_weight       : {args.no_reply_mask_weight}")
    print(f"  include_im_end_in_loss     : {args.include_im_end_in_loss}")


if __name__ == "__main__":
    main()

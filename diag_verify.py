"""
Decisive diagnostic for the spec/AR divergence on long content.

Hypothesis under test: the BATCHED verify pass (feeding N drafted-token hidden
states to the verify layers at once) does NOT equal the SEQUENTIAL AR pass
(one token at a time), because of a causal-mask / cache_position issue in the
batched path. If true, batched verify lets earlier tokens attend to later
(future) drafted tokens, contaminating their logits.

What it does (one responding turn, greedy):
  1. Prefill on a forced-content prompt.
  2. AR reference: decode K tokens one-by-one, recording each token id AND the
     verify-layer logits argmax at each step.
  3. Replay: take the SAME K tokens, run draft-layers one-by-one to collect the
     exited hidden states (exactly like spec), then run ONE batched verify on
     all K hidden states. Compare its per-position argmax to the AR argmax.
  4. Print the first position where batched-verify argmax != AR argmax.

If they differ at some position -> batched verify is the bug (causal mask).
If they match everywhere -> the bug is elsewhere (KV trim across rounds), and
we instrument the multi-round path next.

Run:
  python diag_verify.py \
      --llm_pretrained /data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt \
      --adapter_path   /path/to/adapter/epochXXX \
      --test_fname     /path/to/ego_dataset.json \
      --device cuda:5 --exit_layer 4 --num_adapter_layers 3
"""
import argparse, json
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from kangaroo_model import KangarooQwenModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_pretrained", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--test_fname", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--exit_layer", type=int, default=4)
    p.add_argument("--num_adapter_layers", type=int, default=3)
    p.add_argument("--k", type=int, default=12, help="how many content tokens to test")
    p.add_argument("--system_prompt", default=(
        "You are a helpful assistant. Your task is to answer questions based on "
        "continuously incoming video frames. Your responses should include "
        "information from the video since your last reply (if any). If the "
        "information in this segment of the video cannot answer the question, "
        "output \"NO REPLY\"."))
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    model = KangarooQwenModel(
        base_model_path=args.llm_pretrained,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        num_adapter_layers=args.num_adapter_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    proc = AutoProcessor.from_pretrained(args.llm_pretrained)
    tok = proc.tokenizer
    base = model.base_model
    head = model.head_model
    dev = args.device

    # Build a forced-content prompt from the first dataset sample's frames if we
    # can; otherwise a text-only question. We just need a prompt where the base
    # produces a multi-token content answer.
    data = json.load(open(args.test_fname))
    sample = data[0] if isinstance(data, list) else data
    print("sample type:", type(sample),
          "keys:", list(sample.keys()) if isinstance(sample, dict) else "-")

    conv = [{"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe what happens, in one sentence."}]}]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    img, vid = process_vision_info(conv)
    inputs = proc(text=[text], images=img, videos=vid,
                  padding=True, return_tensors="pt").to(dev)
    ctx = inputs["input_ids"].shape[1]

    def prefill():
        model.reset_status()
        fk = {k: inputs.get(k) for k in
              ["input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
               "image_grid_thw", "video_grid_thw", "second_per_grid_ts"]}
        fk = {k: v for k, v in fk.items() if v is not None}
        fk.update(use_cache=True, output_hidden_states=True, return_dict=True,
                  drop_method="none", drop_threshold=1.0, drop_absolute=True)
        out = base.model(**fk)
        base.past_key_values = out.past_key_values
        return out

    # ---- AR reference: token ids + per-step verify argmax ----
    out = prefill()
    first = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
    ar_ids = [first]
    nxt = first
    eos = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
    for _ in range(args.k):
        in_t = torch.tensor([[nxt]], device=dev)
        dh = base.forward_draft_or_large_model(in_tokens_small=in_t)
        _, hn = base.forward_draft_or_large_model(in_features_large=dh)
        nxt = int(torch.argmax(head(hn).float()[:, -1, :], dim=-1).item())
        ar_ids.append(nxt)
        if nxt in eos:
            break
    print("AR ids :", ar_ids)
    print("AR text:", tok.decode(ar_ids))

    K = len(ar_ids) - 1  # we have K transitions to verify
    if K < 2:
        print("answer too short to test batched verify; pick a sample with longer content")
        return

    # ---- Replay spec-style: draft layers one-by-one to gather exited hiddens,
    # then ONE batched verify, compare per-position argmax to AR. ----
    prefill()
    start_index = ctx  # first content token sits at ctx
    exited = None
    # feed ar_ids[0..K-1] through draft layers (these are the inputs whose
    # NEXT token AR predicted as ar_ids[1..K])
    for j in range(K):
        in_t = torch.tensor([[ar_ids[j]]], device=dev)
        h = base.forward_draft_or_large_model(in_tokens_small=in_t)
        exited = h if exited is None else torch.cat([exited, h], dim=1)
    _, hn = base.forward_draft_or_large_model(in_features_large=exited)
    batched_argmax = torch.argmax(head(hn).float(), dim=-1)[0].tolist()

    print("\npos | AR_next | batched_verify | match")
    first_mismatch = None
    for j in range(K):
        ar_next = ar_ids[j + 1]
        bv = batched_argmax[j]
        m = (ar_next == bv)
        if not m and first_mismatch is None:
            first_mismatch = j
        print(f"{j:3d} | {ar_next:6d} {tok.decode([ar_next])!r:12s} | "
              f"{bv:6d} {tok.decode([bv])!r:12s} | {m}")

    print()
    if first_mismatch is None:
        print(">>> batched verify MATCHES AR everywhere. Bug is NOT batched verify; "
              "it is the multi-round KV trim path. Next: instrument across rounds.")
    else:
        print(f">>> FIRST MISMATCH at position {first_mismatch}. Batched verify != "
              f"sequential AR => the verify causal mask / cache_position is the bug. "
              f"Earlier tokens are seeing future drafted tokens.")


if __name__ == "__main__":
    main()

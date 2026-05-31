"""
Sanity-dump trigger positions: print the token context around each assistant
turn boundary and what the BASE model predicts there, so we can verify the
trigger-position alignment is correct before trusting any miss-rate numbers.

The base model scoring miss=0.775 on its OWN training distribution is a red
flag for a misalignment (off-by-one position, or wrong notion of "the trigger
token"). This dumps the raw evidence.

Usage:
  python dump_trigger_context.py \
      --basepath /path/MMDuet2_ckpt \
      --datadir  /path/ego_gate_ckpts \
      --val_ratio 0.05 --val_seed 42 \
      --num_show 8 --context 6
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basepath", required=True)
    p.add_argument("--datadir", required=True)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--val_seed", type=int, default=42)
    p.add_argument("--no_reply_text", type=str, default="NO REPLY")
    p.add_argument("--num_show", type=int, default=8, help="How many RESPOND turns to dump")
    p.add_argument("--context", type=int, default=6, help="Tokens of context each side")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return p.parse_args()


def split_val(datadir, val_ratio, val_seed):
    import random
    files = sorted(glob.glob(os.path.join(datadir, "*.ckpt")))
    shuffled = list(files)
    random.Random(val_seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    return shuffled[:n_val]


def find_segments(loss_mask):
    pos = (loss_mask > 0).to(torch.int8)
    padded = torch.cat([torch.zeros(1, dtype=torch.int8), pos, torch.zeros(1, dtype=torch.int8)])
    diff = padded[1:] - padded[:-1]
    starts = torch.nonzero(diff == 1).flatten().tolist()
    ends = torch.nonzero(diff == -1).flatten().tolist()
    return list(zip(starts, ends))


def load_lm_head(basepath, hidden_size, vocab_size, dtype):
    from safetensors import safe_open
    head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    with open(os.path.join(basepath, "model.safetensors.index.json")) as f:
        import json
        head_file = json.load(f)["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(basepath, head_file), framework="pt", device="cpu") as f:
        head.weight.data = f.get_tensor("lm_head.weight").to(dtype)
    return head.to(dtype)


@torch.no_grad()
def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    cfg = AutoConfig.from_pretrained(args.basepath)
    tok = AutoTokenizer.from_pretrained(args.basepath)
    head = load_lm_head(args.basepath, cfg.hidden_size, cfg.vocab_size, dtype).to(args.device)
    norm = args.no_reply_text.strip()
    no_tok = tok(norm, add_special_tokens=False).input_ids[0]

    files = split_val(args.datadir, args.val_ratio, args.val_seed)
    shown = 0
    C = args.context

    def tokstr(tid):
        return repr(tok.decode([int(tid)]))

    for fp in files:
        if shown >= args.num_show:
            break
        d = torch.load(fp, map_location="cpu", weights_only=False)
        ids = d["input_ids"]
        hid = d["hidden_state"].to(args.device, dtype)  # final layer (1?,L,D) -> (L,D)
        if hid.dim() == 3:
            hid = hid[0]
        for s, e in find_segments(d["loss_mask"]):
            if s == 0:
                continue
            text = tok.decode(ids[s:e].tolist(), skip_special_tokens=True).strip()
            if text == norm:
                continue  # only dump RESPOND turns (the ones base is missing)
            shown += 1
            print("\n" + "=" * 90)
            print(f"file={os.path.basename(fp)}  segment=[{s},{e})  GT_text={text[:60]!r}")
            lo = max(0, s - C)
            hi = min(len(ids), s + C)
            print(f"  context tokens [{lo},{hi}) (segment start s={s} marked >>):")
            for j in range(lo, hi):
                mark = ">>" if j == s else "  "
                # base prediction made AT position j (predicts token j+1)
                logits = head(hid[j].unsqueeze(0)).float()[0]
                p = F.softmax(logits, dim=-1)
                top5 = torch.topk(p, 5)
                pred_str = ", ".join(
                    f"{tokstr(t)}:{pr:.2f}" for t, pr in zip(top5.indices.tolist(), top5.values.tolist())
                )
                print(f"   {mark} pos {j:>4} tok={tokstr(ids[j]):<14} | base@pos predicts next: [{pred_str}]  p(NO)={p[no_tok]:.3f}")
            # The position the eval script scores: s-1 (predicts token at s)
            print(f"  --> eval scores position s-1={s-1}: base p(NO) there ="
                  f" {F.softmax(head(hid[s-1].unsqueeze(0)).float()[0], dim=-1)[no_tok]:.3f}"
                  f"  (GT first token at s = {tokstr(ids[s])})")
            if shown >= args.num_show:
                break


if __name__ == "__main__":
    main()

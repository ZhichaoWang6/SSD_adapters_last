"""
Offline trigger evaluation for the proactive-response (gate) adapter.

In the gated architecture the adapter alone decides, at each turn boundary,
whether to stay silent ("NO REPLY") or respond. The response *content* is still
produced by lossless speculative decoding (base verifies), so the ONLY new
error source is this trigger decision. This script measures it directly on the
held-out ckpts, without running full generation.

For each assistant turn it compares:
  - GT class            : is the turn's text "NO REPLY" (silent) or a response?
  - adapter prediction  : the adapter's first-token logits at the position that
                          predicts the turn's first token.

The asymmetric cost matters:
  - MISS (GT=respond, pred=NO REPLY)  -> answer is lost. DANGEROUS.
  - FALSE ALARM (GT=NO REPLY, pred=respond) -> one wasted base prefill. Cheap.
So "respond" is treated as the positive class and we report miss rate / recall
explicitly, plus a sweep over the p(NO REPLY) threshold so you can pick an
operating point that drives the miss rate down.

Usage:
  python eval_trigger.py \
      --basepath /path/MMDuet2_ckpt \
      --datadir  /path/ego_gate_ckpts \
      --adapter_path /path/stage2_gate_4-3/epochs/epochXXX \
      --exit_layer 4 --num_adapter_layers 3 \
      --val_ratio 0.05 --val_seed 42 --eval_split val
"""

import argparse
import glob
import json
import os
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from adapter import AdapterModel, create_adapter_config


def parse_args():
    p = argparse.ArgumentParser(description="Offline trigger (NO REPLY vs respond) evaluation")
    p.add_argument("--basepath", required=True, help="Base MMDuet2 ckpt (for lm_head + config + tokenizer)")
    p.add_argument("--datadir", required=True, help="Dir of .ckpt training files")
    p.add_argument("--source", choices=["adapter", "base"], default="adapter",
                   help="Whose trigger decision to evaluate. 'adapter' = early-exit "
                        "hidden_state_layer{N} through the adapter (the deployed gate). "
                        "'base' = the full model's final hidden_state through lm_head "
                        "(the reference: how well the base model itself triggers). Same "
                        "split / NO-token / miss-FA accounting, so the two are directly "
                        "comparable.")
    p.add_argument("--adapter_path", default=None,
                   help="Dir with adapter_model.bin + adapter_config.json. Required for "
                        "--source adapter; ignored for --source base.")
    p.add_argument("--exit_layer", type=int, default=4)
    p.add_argument("--num_adapter_layers", type=int, default=3)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--val_seed", type=int, default=42)
    p.add_argument("--eval_split", choices=["val", "train", "all"], default="val",
                   help="Which split to evaluate. Reproduces train_adapter.py's "
                        "sorted+shuffle(seed) split. Default: val (held-out).")
    p.add_argument("--no_reply_text", type=str, default="NO REPLY")
    p.add_argument("--no_first_token_override", type=int, default=None,
                   help="Force the NO-REPLY first-token id (else inferred from tokenizer "
                        "and validated against the data).")
    p.add_argument("--thresholds", type=str, default="0.5,0.7,0.9,0.95,0.99",
                   help="Comma-separated p(NO REPLY) thresholds to sweep. Predict NO "
                        "REPLY iff p_no >= t (higher t -> respond more -> fewer misses).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--output_json", type=str, default="./outputs/trigger_eval.json")
    return p.parse_args()


def load_lm_head(basepath, hidden_size, vocab_size, dtype):
    from safetensors import safe_open
    head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    index_path = os.path.join(basepath, "model.safetensors.index.json")
    with open(index_path) as f:
        head_file = json.load(f)["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(basepath, head_file), framework="pt", device="cpu") as f:
        head.weight.data = f.get_tensor("lm_head.weight").to(dtype)
    return head.to(dtype)


def load_adapter(basepath, adapter_path, num_adapter_layers, dtype):
    cfg = create_adapter_config(basepath, num_adapter_layers=num_adapter_layers)
    meta_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if "use_mlp" in meta:
            cfg.use_mlp = meta["use_mlp"]
        if "num_adapter_layers" in meta and meta["num_adapter_layers"] != num_adapter_layers:
            raise ValueError(
                f"adapter_config.json num_adapter_layers={meta['num_adapter_layers']} "
                f"!= --num_adapter_layers {num_adapter_layers}")
    adapter = AdapterModel(cfg)
    sd = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu", weights_only=True)
    cleaned = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = adapter.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise ValueError(f"adapter load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return adapter.to(dtype).eval()


def split_files(datadir, val_ratio, val_seed, which):
    import random
    files = sorted(glob.glob(os.path.join(datadir, "*.ckpt")))
    if not files:
        raise ValueError(f"No .ckpt in {datadir}")
    shuffled = list(files)
    random.Random(val_seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    if which == "val":
        return shuffled[:n_val]
    if which == "train":
        return shuffled[n_val:]
    return shuffled


def find_segments(loss_mask):
    """Contiguous runs where loss_mask > 0. Returns list of (start, end_exclusive)."""
    pos = (loss_mask > 0).to(torch.int8)
    padded = torch.cat([torch.zeros(1, dtype=torch.int8), pos, torch.zeros(1, dtype=torch.int8)])
    diff = padded[1:] - padded[:-1]
    starts = torch.nonzero(diff == 1).flatten().tolist()
    ends = torch.nonzero(diff == -1).flatten().tolist()
    return list(zip(starts, ends))


@torch.no_grad()
def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = args.device

    cfg = AutoConfig.from_pretrained(args.basepath)
    tok = AutoTokenizer.from_pretrained(args.basepath)
    head = load_lm_head(args.basepath, cfg.hidden_size, cfg.vocab_size, dtype).to(device)
    if args.source == "adapter":
        if args.adapter_path is None:
            raise ValueError("--source adapter requires --adapter_path")
        adapter = load_adapter(args.basepath, args.adapter_path, args.num_adapter_layers, dtype).to(device)
    else:
        adapter = None

    files = split_files(args.datadir, args.val_ratio, args.val_seed, args.eval_split)
    print(f"Evaluating {len(files)} ckpt(s) [{args.eval_split} split] | source={args.source}")
    norm_no_reply = args.no_reply_text.strip()

    # The "NO REPLY" first-token id. Use the override if given, else the
    # tokenizer's encoding; validated below against the data-derived mode.
    if args.no_first_token_override is not None:
        no_first_token = args.no_first_token_override
    else:
        no_first_token = tok(norm_no_reply, add_special_tokens=False).input_ids[0]

    # Collected per-turn records: (gt_is_no_reply, p_no scalar, argmax_token)
    records = []
    no_first_counter = Counter()  # input_ids at start of GT no-reply segments (validation)
    layer_key = f"hidden_state_layer{args.exit_layer}"

    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        input_ids = d["input_ids"]
        loss_mask = d["loss_mask"]
        if args.source == "adapter":
            if layer_key not in d:
                raise KeyError(f"{fp} has no {layer_key}; regenerate with --exit_layers {args.exit_layer}")
            early = d[layer_key].unsqueeze(0).to(device, dtype)  # (1, L, D)
            pos_ids = d.get("position_ids")
            if pos_ids is not None:
                pos_ids = pos_ids[:, None, :].to(device)  # (3, 1, L)
            hidden = adapter(inputs_embeds=early, position_ids=pos_ids)  # (1, L, D)
        else:
            # base reference: the full model's final hidden state -> lm_head is
            # exactly the base model's own next-token (trigger) prediction.
            if "hidden_state" not in d:
                raise KeyError(f"{fp} has no 'hidden_state' (final layer)")
            hidden = d["hidden_state"].unsqueeze(0).to(device, dtype)  # (1, L, D)

        segments = find_segments(loss_mask)
        trig_positions, gt_no_reply_flags, seg_starts = [], [], []
        for s, e in segments:
            if s == 0:
                continue  # no position predicts token 0
            text = tok.decode(input_ids[s:e].tolist(), skip_special_tokens=True).strip()
            is_no = (text == norm_no_reply)
            trig_positions.append(s - 1)
            gt_no_reply_flags.append(is_no)
            seg_starts.append(s)
            if is_no:
                no_first_counter[int(input_ids[s].item())] += 1

        if not trig_positions:
            continue
        trig_hidden = hidden[0, trig_positions, :]            # (T, D)
        logits = head(trig_hidden).float()                    # (T, V)
        p_no = F.softmax(logits, dim=-1)[:, no_first_token].tolist()  # (T,)
        argmax_tok = logits.argmax(dim=-1).tolist()
        for i in range(len(trig_positions)):
            records.append({
                "gt_no_reply": gt_no_reply_flags[i],
                "argmax": argmax_tok[i],
                "p_no": p_no[i],
            })

    if not records:
        print("No assistant turns found. Nothing to evaluate.")
        return

    # Validate the chosen NO-REPLY first token against the data-derived mode.
    print(f"NO-REPLY first token id = {no_first_token} ({tok.decode([no_first_token])!r}); "
          f"GT no-reply start-token distribution: {dict(no_first_counter.most_common(3))}")
    if no_first_counter:
        data_mode = no_first_counter.most_common(1)[0][0]
        if data_mode != no_first_token:
            print(f"  [WARN] data mode start-token is {data_mode} "
                  f"({tok.decode([data_mode])!r}) but using {no_first_token}. "
                  f"Re-run with --no_first_token_override {data_mode} if the table looks wrong.")

    n_total = len(records)
    n_gt_respond = sum(1 for r in records if not r["gt_no_reply"])
    n_gt_no = n_total - n_gt_respond
    print(f"Turns: {n_total} | GT respond={n_gt_respond} ({n_gt_respond/n_total:.1%}) "
          f"| GT NO REPLY={n_gt_no} ({n_gt_no/n_total:.1%})")

    def confusion(pred_no_fn):
        TP = FN = FP = TN = 0  # positive class = respond
        for r in records:
            pred_no = pred_no_fn(r)
            if not r["gt_no_reply"]:      # GT respond
                if pred_no:
                    FN += 1              # MISS
                else:
                    TP += 1
            else:                         # GT no reply
                if pred_no:
                    TN += 1
                else:
                    FP += 1              # false alarm
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        return {
            "miss_rate": FN / (TP + FN) if (TP + FN) else 0.0,
            "false_alarm": FP / (FP + TN) if (FP + TN) else 0.0,
            "respond_recall": recall,
            "respond_precision": prec,
            "respond_f1": f1,
            "accuracy": (TP + TN) / n_total,
            "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        }

    results = {}
    # Hard argmax decision
    results["argmax"] = confusion(lambda r: r["argmax"] == no_first_token)
    # p(NO REPLY) threshold sweep
    for t in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        results[f"p_no>={t}"] = confusion(lambda r, t=t: r["p_no"] >= t)

    print("\n  policy            miss   falseAlarm  recall  prec   F1     acc   (TP/FN/FP/TN)")
    print("  " + "-" * 84)
    for name, m in results.items():
        print(f"  {name:<16} {m['miss_rate']:6.3f} {m['false_alarm']:10.3f} "
              f"{m['respond_recall']:7.3f} {m['respond_precision']:5.3f} "
              f"{m['respond_f1']:5.3f} {m['accuracy']:6.3f}   "
              f"({m['TP']}/{m['FN']}/{m['FP']}/{m['TN']})")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    out = {
        "n_turns": n_total, "gt_respond": n_gt_respond, "gt_no_reply": n_gt_no,
        "no_first_token": no_first_token, "eval_split": args.eval_split,
        "source": args.source, "adapter_path": args.adapter_path,
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.output_json}")
    print("\nReminder: MISS (GT=respond, pred=NO REPLY) loses an answer and is "
          "unrecoverable. Pick a threshold that drives miss_rate low; false alarms "
          "only cost one wasted base prefill.")


if __name__ == "__main__":
    main()

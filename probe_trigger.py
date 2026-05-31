"""
Linear-probe diagnostic: does the early-exit hidden state CARRY the trigger
signal at all?

The gate adapter's trigger decision is failing (high miss rate, no threshold
separates respond vs NO REPLY). Two very different causes:
  (2) imbalance kills the CE objective -> features HAVE the signal, fixable by
      a binary head / class weighting.
  (3) the early-exit layer is too shallow -> features DON'T have the signal,
      only fixable by exiting deeper (regenerate data at a larger exit_layer).

This script settles it. It trains the simplest possible classifier (logistic
regression with pos_weight) on the RAW base hidden_state_layer{N} at each turn's
trigger position, and reports val AUC + miss/false-alarm at the best threshold.

  high AUC (>~0.85)  -> signal is THERE; a binary head / weighting will help.
  low  AUC (~0.5-0.65) -> signal is NOT in this layer; exit deeper.

Usage:
  python probe_trigger.py \
      --basepath /path/MMDuet2_ckpt \
      --datadir  /path/ego_gate_ckpts \
      --exit_layer 4 \
      --val_ratio 0.05 --val_seed 42
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Linear probe: is the trigger signal in the early-exit features?")
    p.add_argument("--basepath", required=True, help="Base ckpt (for tokenizer only)")
    p.add_argument("--datadir", required=True)
    p.add_argument("--exit_layer", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--val_seed", type=int, default=42)
    p.add_argument("--no_reply_text", type=str, default="NO REPLY")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def split_files(datadir, val_ratio, val_seed):
    import random
    files = sorted(glob.glob(os.path.join(datadir, "*.ckpt")))
    if not files:
        raise ValueError(f"No .ckpt in {datadir}")
    shuffled = list(files)
    random.Random(val_seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    return shuffled[n_val:], shuffled[:n_val]  # train, val


def find_segments(loss_mask):
    pos = (loss_mask > 0).to(torch.int8)
    padded = torch.cat([torch.zeros(1, dtype=torch.int8), pos, torch.zeros(1, dtype=torch.int8)])
    diff = padded[1:] - padded[:-1]
    starts = torch.nonzero(diff == 1).flatten().tolist()
    ends = torch.nonzero(diff == -1).flatten().tolist()
    return list(zip(starts, ends))


def collect(files, exit_layer, tok, norm_no_reply):
    """Return (X, y): trigger-position hidden states and labels (1=respond)."""
    layer_key = f"hidden_state_layer{exit_layer}"
    feats, labels = [], []
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        if layer_key not in d:
            raise KeyError(f"{fp} missing {layer_key}; regenerate with --exit_layers {exit_layer}")
        input_ids = d["input_ids"]
        hid = d[layer_key]  # (L, D)
        for s, e in find_segments(d["loss_mask"]):
            if s == 0:
                continue
            text = tok.decode(input_ids[s:e].tolist(), skip_special_tokens=True).strip()
            is_respond = 0.0 if text == norm_no_reply else 1.0
            feats.append(hid[s - 1])           # position that predicts the first token
            labels.append(is_respond)
    if not feats:
        raise ValueError("No trigger positions found.")
    X = torch.stack(feats).float()
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y


def auc_score(scores, y):
    """ROC-AUC via rank statistic (Mann-Whitney U)."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y == 1].sum().item()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def metrics_at(scores, y, thr):
    pred = (scores >= thr).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    miss = fn / (tp + fn) if (tp + fn) else 0.0
    fa = fp / (fp + tn) if (fp + tn) else 0.0
    return miss, fa, tp, fn, fp, tn


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.basepath)
    norm = args.no_reply_text.strip()
    train_f, val_f = split_files(args.datadir, args.val_ratio, args.val_seed)
    print(f"train ckpts={len(train_f)}  val ckpts={len(val_f)}")

    Xtr, ytr = collect(train_f, args.exit_layer, tok, norm)
    Xva, yva = collect(val_f, args.exit_layer, tok, norm)
    print(f"train turns={len(ytr)} (respond={int(ytr.sum())}, {ytr.mean():.1%}) | "
          f"val turns={len(yva)} (respond={int(yva.sum())}, {yva.mean():.1%})")

    dev = args.device
    # Standardize features on train stats.
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = ((Xtr - mu) / sd).to(dev)
    Xva = ((Xva - mu) / sd).to(dev)
    ytr, yva = ytr.to(dev), yva.to(dev)

    D = Xtr.shape[1]
    w = torch.zeros(D, 1, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)
    pos_weight = ((ytr == 0).sum() / (ytr == 1).sum().clamp_min(1)).to(dev)
    print(f"pos_weight (neg/pos) = {pos_weight.item():.2f}")

    for ep in range(args.epochs):
        opt.zero_grad()
        logits = (Xtr @ w).squeeze(1) + b
        loss = F.binary_cross_entropy_with_logits(logits, ytr, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    with torch.no_grad():
        va_scores = torch.sigmoid((Xva @ w).squeeze(1) + b).cpu()
    yva_c = yva.cpu()
    auc = auc_score(va_scores, yva_c)

    print(f"\n  Linear-probe VAL AUC = {auc:.3f}")
    print("\n  thr     miss   falseAlarm   (TP/FN/FP/TN)")
    print("  " + "-" * 48)
    for thr in [0.3, 0.5, 0.7, 0.9]:
        miss, fa, tp, fn, fp, tn = metrics_at(va_scores, yva_c, thr)
        print(f"  {thr:<5}  {miss:6.3f} {fa:10.3f}     ({tp}/{fn}/{fp}/{tn})")

    print("\n  Verdict:")
    if auc >= 0.85:
        print(f"  AUC={auc:.3f} HIGH -> the trigger signal IS in layer {args.exit_layer}.")
        print("  Cause is the imbalanced CE objective, not the features.")
        print("  -> Add a binary head + pos_weight (or downweight NO REPLY) and retrain.")
    elif auc >= 0.70:
        print(f"  AUC={auc:.3f} MODERATE -> partial signal in layer {args.exit_layer}.")
        print("  A binary head may help but a deeper exit_layer will likely help more.")
    else:
        print(f"  AUC={auc:.3f} LOW -> layer {args.exit_layer} does NOT carry the trigger signal.")
        print("  Class weighting/binary head won't save it. Regenerate data at a")
        print("  deeper --exit_layers (e.g. 8, 12, 16) and re-probe to find where the")
        print("  signal appears.")


if __name__ == "__main__":
    main()

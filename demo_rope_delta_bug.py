"""
Debug script: compare adapter position_ids across the original buggy
fallback, my v1 synthetic fix, and the v2 get_rope_index-based fix,
using a real multimodal training ckpt that already has the ground-truth
position_ids saved.

Run on your machine (no GPU / model load needed; just reads the ckpt):
    python debug_rope_positions.py /path/to/multimodal_ckpt/data_0.ckpt
"""

import argparse
import torch


def detect_regions(input_ids, position_ids):
    """Walk the input and split it into contiguous regions of the same
    'kind' inferred from position_ids:
      - 'text_a': leading text (T = H = W = input_index)
      - 'image' : image / video block (T, H, W disagree, or repeat)
      - 'text_b': text after images (T = H = W, but shifted by rope_deltas)
    Returns a list of (kind, start, end_exclusive)."""
    T = position_ids[0]
    H = position_ids[1]
    W = position_ids[2]
    L = T.shape[0]

    regions = []
    cur_kind = None
    cur_start = 0
    for i in range(L):
        if T[i] == H[i] == W[i] == i:
            kind = 'text_a'  # text with no rope_delta yet
        elif T[i] == H[i] == W[i]:
            kind = 'text_b'  # text with rope_delta applied (T=H=W but != i)
        else:
            kind = 'image'
        if cur_kind is None:
            cur_kind = kind
        elif kind != cur_kind:
            regions.append((cur_kind, cur_start, i))
            cur_kind = kind
            cur_start = i
    regions.append((cur_kind, cur_start, L))
    return regions


def build_buggy_v0(seq_len):
    """Original adapter fallback: 1D arange, broadcast to 3 channels."""
    pos = torch.arange(seq_len)
    return pos.unsqueeze(0).expand(3, seq_len)  # (3, L)


def build_v1_fix(seq_len, rope_deltas):
    """My first 'fix': arange + rope_deltas, still text-style 3 channel."""
    pos = torch.arange(seq_len) + rope_deltas
    return pos.unsqueeze(0).expand(3, seq_len)  # (3, L)


def build_v2_fix(saved_position_ids):
    """v2 fix: take the real position_ids that get_rope_index produced."""
    return saved_position_ids


def diff_stats(name, proper, candidate, regions):
    """Print per-region max abs diff between two position tensors."""
    print(f"\n--- {name} vs proper ---")
    for kind, s, e in regions:
        p = proper[:, s:e]
        c = candidate[:, s:e]
        diff = (p.long() - c.long()).abs()
        max_diff = int(diff.max())
        first_token = list(p[:, 0].tolist())
        cand_first = list(c[:, 0].tolist())
        match_flag = "✓ match" if max_diff == 0 else "✗ wrong"
        print(f"  {kind:8s} [{s:5d}..{e:5d}]  max|delta|={max_diff:6d}   "
              f"proper start (T,H,W)={first_token}   "
              f"candidate start (T,H,W)={cand_first}   {match_flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/wangzhichao/projects/SSD_adapters/datasets/multiturn_281230_noreplymask0.1/data_0.ckpt",help="Path to a multimodal training ckpt with saved position_ids")
    args = ap.parse_args()

    print(f"Loading ckpt: {args.ckpt}")
    d = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    if 'position_ids' not in d:
        print("ERROR: this ckpt has no 'position_ids' field. Need a multimodal ckpt from generate_training_data_multiturn.py")
        return

    input_ids = d['input_ids']  # (L,)
    proper = d['position_ids']  # (3, L)
    L = input_ids.shape[0]

    print(f"\nSeq length: {L}")
    print(f"input_ids[:10]: {input_ids[:10].tolist()}")
    print(f"position_ids[:, :10]:\n{proper[:, :10]}")

    # Estimate rope_deltas from the last text-style position
    # For text_b at input index L-1, proper position = (L-1) + rope_deltas
    # Find a representative text_b position
    last_idx = L - 1
    text_b_position = int(proper[0, last_idx])
    rope_deltas = text_b_position - last_idx
    print(f"\nEstimated rope_deltas: {rope_deltas}")
    print(f"  (computed from last token: proper position {text_b_position} - input index {last_idx})")

    regions = detect_regions(input_ids, proper)
    print(f"\nDetected {len(regions)} region(s):")
    for kind, s, e in regions:
        print(f"  {kind:8s} [{s:5d}..{e:5d}]  length={e - s}")
        # Sample a few position values
        sample_indices = [s, min(s + 3, e - 1), (s + e) // 2, e - 1]
        sample_indices = sorted(set(sample_indices))
        for i in sample_indices:
            print(f"    pos[:, {i:5d}] = {proper[:, i].tolist()}  "
                  f"(input_idx={i}, T={int(proper[0,i])}, H={int(proper[1,i])}, W={int(proper[2,i])})")

    # Three candidate implementations
    v0_buggy = build_buggy_v0(L)
    v1_synth = build_v1_fix(L, rope_deltas)
    v2_proper = build_v2_fix(proper)

    diff_stats("v0 (original buggy: arange)", proper, v0_buggy, regions)
    diff_stats("v1 (my first fix: arange + rope_deltas)", proper, v1_synth, regions)
    diff_stats("v2 (final fix: get_rope_index)", proper, v2_proper, regions)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: where each version is wrong")
    print("=" * 70)
    for name, candidate in [
        ("v0 buggy", v0_buggy),
        ("v1 synth", v1_synth),
        ("v2 proper", v2_proper),
    ]:
        per_region_wrong = []
        for kind, s, e in regions:
            d_abs = (proper[:, s:e].long() - candidate[:, s:e].long()).abs()
            if int(d_abs.max()) > 0:
                per_region_wrong.append(kind)
        if per_region_wrong:
            print(f"  {name:10s}: WRONG on {sorted(set(per_region_wrong))}")
        else:
            print(f"  {name:10s}: CORRECT everywhere ✓")


if __name__ == "__main__":
    main()
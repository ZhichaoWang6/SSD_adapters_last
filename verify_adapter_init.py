"""Verify that an adapter freshly initialized from base layers has byte-identical
weights to the corresponding base-model layers.

Usage:
    python verify_adapter_init.py \
        --basepath /path/to/Qwen2.5-VL \
        --exit_layer 12 \
        --num_adapter_layers 3
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

from kangaroo_model import KangarooQwenModel


def load_base_layer_weights(base_path, layer_idx):
    """Return {param_name_without_prefix: tensor} for base.model.layers[layer_idx].*"""
    prefix = f"model.layers.{layer_idx}."
    index_path = os.path.join(base_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = {fn for k, fn in weight_map.items() if k.startswith(prefix)}
    else:
        files = {"model.safetensors"}

    out = {}
    for fn in files:
        path = os.path.join(base_path, fn)
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basepath", default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    p.add_argument("--exit_layer", type=int, default=12)
    p.add_argument("--num_adapter_layers", type=int, default=3)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    args = p.parse_args()

    model = KangarooQwenModel(
        base_model_path=args.basepath,
        adapter_model_path=None,
        early_exit_layer=args.exit_layer,
        num_adapter_layers=args.num_adapter_layers,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    # mapping: adapter-layer parameter suffix -> base-layer parameter key
    name_map = {
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.q_proj.bias":   "self_attn.q_proj.bias",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.k_proj.bias":   "self_attn.k_proj.bias",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.v_proj.bias":   "self_attn.v_proj.bias",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "input_layernorm.weight":          "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "gate_proj.weight":        "mlp.gate_proj.weight",
        "up_proj.weight":          "mlp.up_proj.weight",
        "down_proj.weight":        "mlp.down_proj.weight",
    }

    ok = True
    for i in range(args.num_adapter_layers):
        base_idx = args.exit_layer + i
        base_w = load_base_layer_weights(args.basepath, base_idx)
        adapter_layer = model.adapter_model.layers[i]
        adapter_sd = adapter_layer.state_dict()
        print(f"\n=== adapter.layers[{i}] vs base.layers[{base_idx}] ===")
        for ak, bk in name_map.items():
            if ak not in adapter_sd:
                continue
            if bk not in base_w:
                print(f"  [skip] {ak}: base key '{bk}' missing")
                continue
            a = adapter_sd[ak].to("cpu", torch.float32)
            b = base_w[bk].to("cpu", torch.float32)
            same = torch.equal(a, b)
            ok &= same
            print(f"  {ak:45s} match={same}  max|delta|={(a - b).abs().max().item():.3e}")

    # Also verify the shared final norm copied from base.model.norm
    print(f"\n=== adapter.norm vs base.model.norm ===")
    index_path = os.path.join(args.basepath, "model.safetensors.index.json")
    norm_tensor = None
    if os.path.exists(index_path):
        wm = json.load(open(index_path))["weight_map"]
        norm_file = wm.get("model.norm.weight")
        if norm_file is not None:
            with safe_open(os.path.join(args.basepath, norm_file), framework="pt", device="cpu") as f:
                norm_tensor = f.get_tensor("model.norm.weight")
    else:
        single = os.path.join(args.basepath, "model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as f:
            norm_tensor = f.get_tensor("model.norm.weight")
    if norm_tensor is not None:
        a = model.adapter_model.norm.weight.detach().to("cpu", torch.float32)
        b = norm_tensor.to(torch.float32)
        same = torch.equal(a, b)
        ok &= same
        print(f"  norm.weight match={same}  max|delta|={(a - b).abs().max().item():.3e}")

    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
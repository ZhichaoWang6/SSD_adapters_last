"""
Decisive multi-round diagnostic for the long-content lossless failure.

Drives the REAL kangaroo_speculative_generate and the REAL AR baseline on a
prompt that forces long content, then compares their token ids and reports the
first divergence. Also re-runs spec with speculative_steps=1 (one drafted token
per round => spec verify becomes per-token, like AR). The pattern tells us where
the bug is:

  * steps=1 matches AR, steps=6 diverges  -> multi-token round / KV-trim bug
  * steps=1 also diverges                 -> per-round re-entry / cache bug
  * both match                            -> bug was the gate path after all

Build a forced-content prompt: we append a fake prior assistant content turn so
the model is "mid-answer" and continues with content rather than NO REPLY.

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
from inference_kangaroo import kangaroo_speculative_generate
from ar_generate import autoregressive_manual_baseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_pretrained", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--test_fname", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--exit_layer", type=int, default=4)
    p.add_argument("--num_adapter_layers", type=int, default=3)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--turn_idx", type=int, default=-1,
                   help="which conversation turn to use as the prompt boundary; "
                        "-1 = use the last user turn in the sample")
    return p.parse_args()


@torch.no_grad()
def run_spec(model, proc, inputs, steps, exit_layer):
    model.reset_status()
    eos = model.base_model.model.generation_config.eos_token_id
    out_ids, _, stats = kangaroo_speculative_generate(
        model=model, inputs=inputs, processor=proc, past_key_values=None,
        max_new_tokens=256, early_exit_layer=exit_layer,
        speculative_steps=steps, threshold=0.0,  # threshold 0 => never early-stop draft
        eos_token_ids=eos,
    )
    ctx = inputs["input_ids"].shape[1]
    return out_ids[0, ctx:].tolist()


@torch.no_grad()
def run_ar(model, proc, inputs, exit_layer):
    model.reset_status()
    eos = model.base_model.model.generation_config.eos_token_id
    txt, _, _ = autoregressive_manual_baseline(
        model=model, inputs=inputs, processor=proc, max_new_tokens=256,
        early_exit_layer=exit_layer, eos_token_ids=eos,
    )
    return txt


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
    dev = args.device

    data = json.load(open(args.test_fname))
    sample = data[args.sample_idx]
    conv = sample["conversation"]
    print(f"sample {sample.get('question_id')} has {len(conv)} turns")

    # Find a boundary that forces CONTENT: keep history up to and including the
    # turn that in the real run produced content, then add_generation_prompt.
    # Simplest robust choice: build the full conversation up to the LAST user
    # turn and let the model answer. If that yields NO REPLY, the user can pass
    # --turn_idx to pick an earlier boundary that had real content.
    if args.turn_idx >= 0:
        hist = conv[:args.turn_idx + 1]
    else:
        # last user turn
        last_user = max(i for i, t in enumerate(conv) if t.get("role") == "user")
        hist = conv[:last_user + 1]

    text = proc.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    img, vid = process_vision_info(hist)
    inputs = proc(text=[text], images=img, videos=vid,
                  padding=True, return_tensors="pt").to(dev)
    print("context_len:", inputs["input_ids"].shape[1])

    ar_txt = run_ar(model, proc, inputs, args.exit_layer)
    ar_ids = tok.encode(ar_txt, add_special_tokens=False)
    print("\nAR text:", ar_txt)
    print("AR ids :", ar_ids[:40])

    for steps in (1, 6):
        spec_ids = run_spec(model, proc, inputs, steps, args.exit_layer)
        spec_txt = tok.decode(spec_ids, skip_special_tokens=True)
        match = (spec_txt.strip() == ar_txt.strip())
        # first divergence position
        fd = None
        for i in range(min(len(spec_ids), len(ar_ids))):
            if spec_ids[i] != ar_ids[i]:
                fd = i
                break
        print(f"\n=== spec steps={steps} ===")
        print("spec text:", spec_txt)
        print("MATCH AR?", match, "| first divergence idx:", fd)
        if fd is not None:
            lo = max(0, fd - 2)
            print(f"  AR  [{lo}:{fd+3}] :", ar_ids[lo:fd+3],
                  [tok.decode([x]) for x in ar_ids[lo:fd+3]])
            print(f"  spec[{lo}:{fd+3}] :", spec_ids[lo:fd+3],
                  [tok.decode([x]) for x in spec_ids[lo:fd+3]])

    print("\nInterpretation:")
    print("  steps=1 matches, steps=6 diverges -> multi-token round / KV-trim bug")
    print("  steps=1 also diverges            -> per-round re-entry / cache bug")
    print("  both match                       -> earlier non-lossless was the gate path")


if __name__ == "__main__":
    main()

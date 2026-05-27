"""
Speculative decoding inference for Qwen2.5-VL with Kangaroo adapter.

This replaces model.generate() with a custom draft-verify loop:
1. Prefill: Run full model on all input tokens (text + visual) normally
2. Draft: Run early layers + adapter to generate candidate tokens
3. Verify: Run remaining layers to check draft tokens
4. Accept tokens until first mismatch (lossless for greedy decoding)

Adapted from Kangaroo's inference_kangaroo.py for Qwen2.5-VL.
"""

import argparse
import copy
import time

import torch
from transformers.cache_utils import DynamicCache


_PRINTED_TEXT_POSITION_VERIFY = False
_PRINTED_MM_POSITION_VERIFY = False


def _verify_adapter_prefill_positions(prefill_position_ids):
    """One-time-per-modality runtime sanity check on the adapter's prefill
    position_ids. Prints once for the first pure-text input, and once
    again for the first multimodal input where rope_deltas takes effect.
    This way streaming-VL pipelines (text-only first turn, then frames
    accumulate) reveal both code paths.
    """
    global _PRINTED_TEXT_POSITION_VERIFY, _PRINTED_MM_POSITION_VERIFY

    pos = prefill_position_ids.detach().cpu()
    if pos.dim() == 3 and pos.shape[1] == 1:
        pos = pos[:, 0, :]
    L = pos.shape[-1]
    T_ch, H_ch, W_ch = pos[0], pos[1], pos[2]

    # Region detection (same logic as debug_rope_positions.py)
    regions = []
    cur_kind = None
    cur_start = 0
    for i in range(L):
        t, h, w = int(T_ch[i]), int(H_ch[i]), int(W_ch[i])
        if t == h == w == i:
            kind = 'text_a'
        elif t == h == w:
            kind = 'text_b'
        else:
            kind = 'image'
        if cur_kind is None:
            cur_kind = kind
        elif kind != cur_kind:
            regions.append((cur_kind, cur_start, i))
            cur_kind = kind
            cur_start = i
    regions.append((cur_kind, cur_start, L))

    n_text_a = sum(1 for k, _, _ in regions if k == 'text_a')
    n_image = sum(1 for k, _, _ in regions if k == 'image')
    n_text_b = sum(1 for k, _, _ in regions if k == 'text_b')
    is_multimodal = (n_image > 0)

    # Skip if we've already printed the verification for this modality.
    if is_multimodal:
        if _PRINTED_MM_POSITION_VERIFY:
            return
        _PRINTED_MM_POSITION_VERIFY = True
    else:
        if _PRINTED_TEXT_POSITION_VERIFY:
            return
        _PRINTED_TEXT_POSITION_VERIFY = True

    last_text_b_idx = None
    for kind, s, e in reversed(regions):
        if kind == 'text_b':
            last_text_b_idx = e - 1
            break
    estimated_rope_delta = int(T_ch[last_text_b_idx] - last_text_b_idx) if last_text_b_idx is not None else 0

    buggy = torch.arange(L)
    rows = []
    for kind, s, e in regions:
        proper_t = T_ch[s:e]
        proper_h = H_ch[s:e]
        proper_w = W_ch[s:e]
        proper_max_d = max(
            int((proper_t - buggy[s:e]).abs().max()),
            int((proper_h - buggy[s:e]).abs().max()),
            int((proper_w - buggy[s:e]).abs().max()),
        )
        rows.append((kind, s, e, proper_max_d))

    print("=" * 78)
    label = "MULTIMODAL" if is_multimodal else "PURE-TEXT"
    print(f"[adapter prefill] {label} position_id verification (first {label.lower()} input)")
    print(f"  seq_len:              {L}")
    print(f"  regions detected:     {len(regions)}  "
          f"(text_a={n_text_a}, image={n_image}, text_b={n_text_b})")
    print(f"  T/H/W differ?:        "
          f"{bool((T_ch != H_ch).any() or (T_ch != W_ch).any())}  "
          f"(True means image tokens have proper 3D mRoPE)")
    print(f"  estimated rope_delta: {estimated_rope_delta}  "
          f"(from last text_b position - input index)")
    print(f"  max position value:   {int(pos.max())}")
    print(f"  first 5 positions:    T={T_ch[:5].tolist()}  "
          f"H={H_ch[:5].tolist()}  W={W_ch[:5].tolist()}")
    print(f"  last 5 positions:     T={T_ch[-5:].tolist()}  "
          f"H={H_ch[-5:].tolist()}  W={W_ch[-5:].tolist()}")

    print("-" * 78)
    print("  Region summary (first 6 + last 3 shown if many):")
    if len(rows) > 9:
        show = rows[:6] + [("...", -1, -1, -1)] + rows[-3:]
    else:
        show = rows
    for entry in show:
        kind, s, e, max_d_vs_buggy = entry
        if kind == "...":
            print(f"    ... ({len(rows) - 9} more regions) ...")
            continue
        delta_str = f"max|delta| vs buggy arange = {max_d_vs_buggy}"
        if kind == 'text_a':
            note = "(buggy would also be correct here)"
        else:
            note = "(buggy is WRONG here -> fix matters)"
        print(f"    {kind:8s} [{s:5d}..{e:5d}]  len={e-s:5d}   {delta_str:35s}  {note}")

    if not is_multimodal:
        print("  >>> Pure-text input (no images / video). All three position variants")
        print("      would produce identical results. The fix has no effect here.")
        print("      (verification will fire again the first time a multimodal input arrives)")
    else:
        bad = sum(1 for _, _, _, d in rows if d > 0)
        total = len(rows)
        if bad > 0:
            print(f"  >>> Multimodal input: buggy version would be WRONG on {bad}/{total} regions")
            print(f"      Current code (v2 get_rope_index) gives the proper positions. ✓")
        else:
            print("  >>> All regions agree with arange; nothing for the fix to correct.")
    print("=" * 78)


def _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens):
    """Build comprehensive timing and acceptance statistics."""
    decode_time = total_time - prefill_time
    avg_accept = sum(accept_length_list) / len(accept_length_list) if accept_length_list else 0
    avg_draft_accept = sum(max(0, a - 1) for a in accept_length_list) / len(accept_length_list) if accept_length_list else 0
    tokens_per_second = num_new_tokens / total_time if total_time > 0 else 0
    decode_tokens_per_second = num_new_tokens / decode_time if decode_time > 0 else 0
    return {
        'accept_lengths': accept_length_list,
        'avg_accept_length': avg_accept,
        'avg_draft_accept_length': avg_draft_accept,
        'total_rounds': len(accept_length_list),
        'total_tokens': num_new_tokens,
        'total_time': total_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'draft_times': draft_times,
        'verify_times': verify_times,
        'avg_draft_time': sum(draft_times) / len(draft_times) if draft_times else 0,
        'avg_verify_time': sum(verify_times) / len(verify_times) if verify_times else 0,
        'tokens_per_second': tokens_per_second,
        'decode_tokens_per_second': decode_tokens_per_second,
    }


@torch.no_grad()
def kangaroo_speculative_generate(
    model,
    inputs,
    processor,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
    do_sample: bool = False,
    past_key_values=None,
):
    # =======================
    adapter_correct = 0
    adapter_total = 0
    adapter_first_correct = 0
    adapter_first_total = 0
    adapter_confidences = []
    adapter_accept_probs = []
    #============================

    # print("going into kangaroo speculative generation...")
    assert not do_sample, "Only greedy decoding is supported for speculative decoding"

    if past_key_values is not None:
        cached_len = 0
        if len(past_key_values.key_cache) > 0 and len(past_key_values.key_cache[0]) > 0:
            cached_len = past_key_values.key_cache[0].shape[2]
        if cached_len > 0:
            print(
                f"[kangaroo] Received past_key_values with cache_len={cached_len}. "
                "This function expects delta-only inputs when KV cache is reused; "
                "passing a full prompt with a non-empty cache will duplicate context."
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model
    device = inputs['input_ids'].device

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
        token_eos = token_eos[0]
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape
    # print(f"batch_size: {batch_size}, context_length: {context_length}")
    assert batch_size == 1, "Speculative decoding only supports batch_size=1"

    max_length = context_length + max_new_tokens

    global_tokens = torch.full((batch_size, max_length), token_eos, dtype=torch.long, device=device)
    global_tokens[:, :context_length] = input_ids

    accept_length_list = []
    start_index = context_length

    # ========== STEP 0: Prefill ==========
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()

    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'output_hidden_states': True,
        'return_dict': True,
        'past_key_values': past_key_values,
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        'drop_method': 'none',
        'drop_threshold': 1.0,
        'drop_absolute': True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    output = base_model.model(**forward_kwargs)
    base_model.past_key_values = output.past_key_values

    first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
    global_tokens[:, start_index] = first_token.item()

    hidden_state_early = output.hidden_states[early_exit_layer]

    # Build TRUE 3D mRoPE position_ids for the adapter's prefill by reusing
    # the base model's get_rope_index. This is the only way to get correct
    # positions for image / video tokens (their T, H, W differ from each
    # other and from text-style arange). The earlier "arange + rope_deltas"
    # shortcut was wrong for text-before-image and image tokens themselves.
    prefill_position_ids, _ = base_model.model.get_rope_index(
        inputs['input_ids'],
        inputs.get('image_grid_thw'),
        inputs.get('video_grid_thw'),
        inputs.get('second_per_grid_ts'),
        inputs.get('attention_mask'),
    )
    prefill_position_ids = prefill_position_ids.to(hidden_state_early.device)
    # Shape is (3, batch, seq_len) already, matching adapter's expectation.

    # ---- One-time runtime sanity check: confirm position_ids look right ----
    _verify_adapter_prefill_positions(prefill_position_ids)

    _, adapter_past_key_values = adapter_model.forward_early_stop(
        inputs_embeds=hidden_state_early,
        position_ids=prefill_position_ids,
        use_cache=True,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    draft_times = []
    verify_times = []

    print(f" first token :{tokenizer.decode(first_token)}")
    if first_token.item() in token_eos_set:
        output_ids = global_tokens[:, :start_index + 1]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t_start
        stats = _build_stats([], prefill_time, [], [], total_time, 1)
        return output_ids, base_model.past_key_values, stats

    # ========== Draft-Verify Loop ==========
    max_infer_steps = min(max_length, start_index + max_new_tokens)
    stop = False
    round_idx = 0

    while start_index < max_infer_steps - 1:
        round_idx += 1
        start_index_copy = start_index
        end_index = start_index + 1
        remaining_budget = max_infer_steps - 1 - start_index
        round_speculative_steps = min(speculative_steps, remaining_budget)

        # ---- STEP 1: Draft ----
        # print("=====================** STEP 1: Draft **============================")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_draft_start = time.perf_counter()
        exited_hidden_states = None
        draft_token_ids = []

        for step in range(1 + round_speculative_steps):
            in_token = global_tokens[:, end_index - 1:end_index]
            print(f"\nDraft step {step}: in_token={in_token}, in_token_decoded={tokenizer.decode(in_token[0])}, end_index: {end_index}")

            adapter_cache_len = adapter_past_key_values[0][0].shape[2] if adapter_past_key_values else 0
            if adapter_cache_len < end_index - 1:
                hidden_state_early_last = exited_hidden_states[:, -1:, :] if exited_hidden_states is not None else None
            else:
                hidden_state_early_last = None

            hidden_state_early = base_model.forward_draft_or_large_model(
                in_tokens_small=in_token,
            )

            if step == 0:
                exited_hidden_states = None

            exited_hidden_states = hidden_state_early if exited_hidden_states is None \
                else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

            adapter_input = hidden_state_early
            if hidden_state_early_last is not None:
                adapter_input = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

            if step == round_speculative_steps:
                print(f"Draft step {step} reached round speculative step limit")
                break
            if step > 0 and predict_score < threshold:
                print(f"Draft step {step}, token {tokenizer.decode(predicted_token)}, predict_score {predict_score} < threshold {threshold}, stopping draft")
                break

            # Build mRoPE position_ids for the adapter explicitly, matching
            # the base model's verify-layer computation (earlyexit_qwen.py).
            # The adapter's own fallback uses a plain 1D arange that omits
            # rope_deltas, which is wrong for multimodal contexts (image /
            # video tokens push the effective position forward by thousands
            # of slots).
            adapter_seq_len = adapter_input.shape[1]
            adapter_past_len = (
                adapter_past_key_values[0][0].shape[2]
                if adapter_past_key_values is not None and len(adapter_past_key_values) > 0
                else 0
            )
            rope_deltas = base_model.model.rope_deltas
            if rope_deltas is not None:
                delta = (adapter_past_len + rope_deltas).to(adapter_input.device)
            else:
                delta = adapter_past_len
            adapter_position_ids = torch.arange(
                adapter_seq_len, device=adapter_input.device,
            )
            adapter_position_ids = adapter_position_ids.view(1, -1).expand(adapter_input.shape[0], -1) + delta
            adapter_position_ids = adapter_position_ids.unsqueeze(0).expand(3, -1, -1)

            hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=adapter_input,
                position_ids=adapter_position_ids,
                past_key_values=adapter_past_key_values,
                use_cache=True,
            )

            predict_logits = head_model(hidden_state[:, -1:, :]).float()
            predicted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)

            predict_score = predict_logits.softmax(dim=-1).max().item()
            # =====================================
            adapter_confidences.append(predict_score)
            #=======================================
            print(f"predicted_token: {predicted_token.item()}, predict_score: {predict_score}, token : {tokenizer.decode(predicted_token)}")

            global_tokens[:, end_index] = predicted_token
            draft_token_ids.append(predicted_token.item())

            if predicted_token.item() in token_eos_set:
                end_index += 1
                # print(f"Drafted token is EOS, stopping draft.")
                break

            end_index += 1

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        draft_times.append(time.perf_counter() - t_draft_start)

        # ---- STEP 2+3: Verify and Accept ----
        # print("======================** STEP 2+3: Verify and Accept **==============================")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_verify_start = time.perf_counter()

        output_length = end_index - start_index

        # base_model.past_key_values._seen_tokens = start_index
        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        # print(f"Verifying {output_length} drafted tokens...")
        # print(f"Current global tokens: {tokenizer.batch_decode(global_tokens[:, start_index:end_index])}")

        assert verify_cache_len == start_index, \
            f"Verify cache mismatch: {verify_cache_len} != {start_index}"

        # print(f"Sending drafted tokens to base_model for verification... {exited_hidden_states.shape}")

        _, hidden_state_normed = base_model.forward_draft_or_large_model(
            in_features_large=exited_hidden_states,
        )
        verify_logits = head_model(hidden_state_normed).float()
        verify_ids = torch.argmax(verify_logits, dim=-1)[0].tolist()

        for i, verify_id in enumerate(verify_ids):
            write_index = start_index + 1 + i
            if write_index >= max_length:
                start_index = max_length - 1
                stop = True
                break

            is_last = (i == output_length - 1)
            is_eos = (verify_id in token_eos_set)
            draft_id = global_tokens[0, start_index + 1 + i].item() if i < len(draft_token_ids) else None
            print(f"Verifying token {i}: verify_id={verify_id} ({tokenizer.decode(verify_id)}), draft_id={draft_id} ({tokenizer.decode(draft_id) if draft_id is not None else None}), is_last={is_last}, is_eos={is_eos}")
            is_mismatch = (not is_last and draft_id is not None and verify_id != draft_id)
            # print(f"is_mismatch: {is_mismatch}")

            # ========================================
            if draft_id is not None and not is_last:
                adapter_total += 1
                if verify_id == draft_id:
                    adapter_correct += 1
                if i == 0:
                    adapter_first_total += 1
                    if verify_id == draft_id:
                        adapter_first_correct += 1
            # =========================================

            if is_last or is_eos or is_mismatch:
                global_tokens[0, start_index + 1 + i] = verify_id
                print(f"Token {i} verification failed, accepting up to this token. is_last: {is_last}, is_eos: {is_eos}, is_mismatch: {is_mismatch}")
                start_index = start_index + 1 + i
                if is_eos:
                    stop = True
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        verify_times.append(time.perf_counter() - t_verify_start)

        accept_len = start_index - start_index_copy
        accept_length_list.append(accept_len)
        # print(f"Round {round_idx} accepted length: {accept_len}, total accepted length: {accept_length_list}")

        # ---- STEP 4: Trim caches ----
        draft_cache_len = base_model._get_layer_cache_length(0)
        if draft_cache_len > start_index:
            base_model.trim_draft_layers_cache(start_index)

        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        if verify_cache_len > start_index:
            base_model.trim_verify_layers_cache(start_index)

        if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
            adapter_past_key_values = [
                (k[:, :, :start_index, :], v[:, :, :start_index, :])
                for k, v in adapter_past_key_values
            ]

        base_model.past_key_values._seen_tokens = start_index
        assert base_model._get_layer_cache_length(0) == start_index, \
            f"Draft cache after trim: {base_model._get_layer_cache_length(0)} != {start_index}"
        assert base_model._get_layer_cache_length(early_exit_layer) == start_index, \
            f"Verify cache after trim: {base_model._get_layer_cache_length(early_exit_layer)} != {start_index}"

        if stop:
            break

    # Final output
    output_ids = global_tokens[:, :start_index + 1]
    num_new_tokens = start_index + 1 - context_length
    print(f"New tokens: {tokenizer.batch_decode(output_ids[:, context_length:])}")
    print(f"Total new tokens generated: {num_new_tokens}")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start

    stats = _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens)
    # =============================================================================================================
    adapter_acc = adapter_correct / adapter_total if adapter_total > 0 else 0
    adapter_first_acc = adapter_first_correct / adapter_first_total if adapter_first_total > 0 else 0
    avg_confidence = sum(adapter_confidences) / len(adapter_confidences) if adapter_confidences else 0
    draft_accept_per_round = adapter_correct / max(stats['total_rounds'], 1)
    print(
        f"[Spec] progress/round={stats['avg_accept_length']:.2f} "
        f"| draft_accept/round={draft_accept_per_round:.2f} "
        f"(counts every verified draft token, including EOS) "
        f"| rounds={stats['total_rounds']} | tokens={num_new_tokens} | context_len={context_length}"
    )
    print(
        f"[Adapter] top1={adapter_correct}/{adapter_total} ({adapter_acc:.1%}) "
        f"| first_top1={adapter_first_correct}/{adapter_first_total} ({adapter_first_acc:.1%}) "
        f"| avg_confidence={avg_confidence:.3f} | context_len={context_length}"
    )

    stats['adapter_accuracy'] = adapter_acc
    stats['adapter_correct'] = adapter_correct
    stats['adapter_total'] = adapter_total
    stats['adapter_first_accuracy'] = adapter_first_acc
    stats['adapter_first_correct'] = adapter_first_correct
    stats['adapter_first_total'] = adapter_first_total
    stats['adapter_avg_confidence'] = avg_confidence
    stats['progress_per_round'] = stats['avg_accept_length']
    stats['draft_accept_per_round'] = draft_accept_per_round
    #==============================================================================================================

    return output_ids, base_model.past_key_values, stats


def speculative_generate_for_streaming(
    model,
    inputs,
    processor,
    past_key_values=None,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
):
    output_ids, past_key_values, stats = kangaroo_speculative_generate(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=max_new_tokens,
        early_exit_layer=early_exit_layer,
        speculative_steps=speculative_steps,
        threshold=threshold,
        do_sample=False,
        past_key_values=past_key_values,
    )
    # print(f"Speculative generation completed. Stats: {stats}")

    input_length = inputs['input_ids'].shape[1]
    # print(f"Input length: {input_length}, Output length: {output_ids.shape[1]}, New tokens generated: {output_ids.shape[1] - input_length}")
    new_token_ids = output_ids[:, input_length:]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    return reply_text, past_key_values, stats
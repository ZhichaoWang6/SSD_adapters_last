"""
Sanity-check inference for the kangaroo speculative-decoding loop.

This is a near-copy of `inference_kangaroo.py`, except the lightweight
`adapter_model` is replaced with the FULL base-model upper layers
(layers [early_exit_layer:]).  In other words, the draft path now runs
the exact same computation as the verify path, just one token at a time.

Expected behaviour:
    - acceptance rate per round  ≈ speculative_steps + 1
    - adapter top1 accuracy      ≈ 100% (only numerical noise can break it)

If the numbers come out clearly below 100%, the bug is NOT in the adapter
- it is somewhere in this inference loop (verify, KV-cache trim, position
ids, rope deltas, etc.).

Usage: swap the import in `inference.py`:

    from inference_kangaroo_full import speculative_generate_for_streaming
"""

import time

import torch
from transformers.cache_utils import DynamicCache


_PRINTED_DRAFT_STRUCTURE = False


def _print_draft_structure_once(base_model):
    """Print the layers being used as the draft head (full base-model upper
    layers, [early_exit_layer:]). Only prints on the first call per process.
    """
    global _PRINTED_DRAFT_STRUCTURE
    if _PRINTED_DRAFT_STRUCTURE:
        return
    _PRINTED_DRAFT_STRUCTURE = True

    qwen_model = base_model.model.model
    early_exit_layer = base_model.early_exit_layer
    total_layers = len(qwen_model.layers)
    upper_layers = qwen_model.layers[early_exit_layer:]
    upper_norm = qwen_model.norm
    lm_head = base_model.model.lm_head

    n_params = sum(p.numel() for p in upper_layers.parameters())
    n_params += sum(p.numel() for p in upper_norm.parameters())

    print("=" * 80)
    print("[inference_kangaroo_full] DRAFT HEAD = full base-model upper layers")
    print(f"  early_exit_layer = {early_exit_layer}")
    print(f"  total decoder layers in base model = {total_layers}")
    print(f"  draft uses layers [{early_exit_layer}:{total_layers}]  "
          f"({total_layers - early_exit_layer} layers)")
    print(f"  + final RMSNorm + lm_head (shared with verify)")
    print(f"  draft-head trainable-equivalent params: {n_params/1e6:.2f}M "
          "(only used during draft; verify reuses the same weights)")
    print(f"  dtype = {next(upper_layers.parameters()).dtype}, "
          f"device = {next(upper_layers.parameters()).device}")
    print("-" * 80)
    print("Draft-head module (layers[early_exit_layer:]):")
    print(upper_layers)
    print("-" * 80)
    print("Final norm + lm_head:")
    print(upper_norm)
    print(lm_head)
    print("=" * 80)


def _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens):
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


def _clone_upper_cache(base_kv, early_exit_layer, max_length):
    """Clone the upper-layer KV (layers >= early_exit_layer) of `base_kv`
    into a fresh DynamicCache, trimmed to `max_length` tokens. Lower-layer
    slots are left as empty placeholders so the layer indices stay aligned
    with the model's own decoder-layer indices.
    """
    new_cache = DynamicCache()
    n_layers = len(base_kv.key_cache)
    for i in range(n_layers):
        if i < early_exit_layer:
            new_cache.key_cache.append([])
            new_cache.value_cache.append([])
            continue
        k = base_kv.key_cache[i]
        v = base_kv.value_cache[i]
        if isinstance(k, list) or (hasattr(k, 'numel') and k.numel() == 0):
            new_cache.key_cache.append([])
            new_cache.value_cache.append([])
        else:
            new_cache.key_cache.append(k[:, :, :max_length, :].clone().contiguous())
            new_cache.value_cache.append(v[:, :, :max_length, :].clone().contiguous())
    new_cache._seen_tokens = max_length
    return new_cache


@torch.no_grad()
def _run_upper_layers(base_model, hidden_states, upper_kv, return_normed=True):
    """Run the base model's upper layers (layers [early_exit_layer:]) on
    `hidden_states`, writing new KV entries into `upper_kv`.
    Mirrors the position-id / cache-position / mask logic used by
    `forward_draft_or_large_model(in_features_large=...)`.
    """
    qwen_model = base_model.model.model
    early_exit_layer = base_model.early_exit_layer
    layers = qwen_model.layers[early_exit_layer:]
    batch_size, seq_length, _ = hidden_states.shape

    # Current past-length for the upper layers (consistent across them).
    if (len(upper_kv.key_cache) > early_exit_layer
            and not isinstance(upper_kv.key_cache[early_exit_layer], list)
            and upper_kv.key_cache[early_exit_layer].numel() > 0):
        layer_past_length = upper_kv.key_cache[early_exit_layer].shape[2]
    else:
        layer_past_length = 0

    # Same hack as in earlyexit_qwen.py: keep _seen_tokens in sync so that
    # the model's mask / cache-position machinery agrees with the upper-layer
    # cache length (DynamicCache only auto-bumps _seen_tokens for layer 0).
    upper_kv._seen_tokens = layer_past_length

    cache_position = torch.arange(
        layer_past_length, layer_past_length + seq_length,
        device=hidden_states.device,
    )

    rope_deltas = base_model.model.rope_deltas
    if rope_deltas is not None:
        delta = (layer_past_length + rope_deltas).to(hidden_states.device)
    else:
        delta = layer_past_length
    position_ids = torch.arange(seq_length, device=hidden_states.device)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1) + delta
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)

    attention_mask = torch.ones(
        (batch_size, layer_past_length + seq_length),
        dtype=torch.bool, device=hidden_states.device,
    )
    causal_mask = qwen_model._update_causal_mask(
        attention_mask, hidden_states, cache_position,
        upper_kv, output_attentions=False,
    )

    for decoder_layer in layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=upper_kv,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

    if return_normed:
        return qwen_model.norm(hidden_states)
    return hidden_states


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
    # -------- accuracy / confidence tracking --------
    adapter_correct = 0
    adapter_total = 0
    adapter_first_correct = 0
    adapter_first_total = 0
    adapter_confidences = []
    # ------------------------------------------------

    assert not do_sample, "Only greedy decoding is supported for speculative decoding"

    if past_key_values is not None:
        cached_len = 0
        if len(past_key_values.key_cache) > 0 and len(past_key_values.key_cache[0]) > 0:
            cached_len = past_key_values.key_cache[0].shape[2]
        if cached_len > 0:
            print(
                f"[kangaroo-full] Received past_key_values with cache_len={cached_len}. "
                "This function expects delta-only inputs when KV cache is reused; "
                "passing a full prompt with a non-empty cache will duplicate context."
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    base_model = model.base_model
    head_model = model.head_model
    device = inputs['input_ids'].device

    _print_draft_structure_once(base_model)

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
        token_eos = token_eos[0]
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape
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

        # Fresh upper-layer KV scratch for this round. Initialised to the
        # current verify-layer prefix (length == start_index) so that the
        # draft path sees exactly the same K/V history as verify would.
        draft_upper_cache = _clone_upper_cache(
            base_model.past_key_values, early_exit_layer, start_index,
        )

        # ---- STEP 1: Draft (FULL upper layers, not adapter) ----
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_draft_start = time.perf_counter()
        exited_hidden_states = None
        draft_token_ids = []

        for step in range(1 + round_speculative_steps):
            in_token = global_tokens[:, end_index - 1:end_index]
            print(f"\nDraft step {step}: in_token={in_token.item()}, decoded={tokenizer.decode(in_token[0])}, end_index: {end_index}")

            # Lower layers: writes layers [0, early_exit_layer) of
            # base_model.past_key_values, identical to original code.
            hidden_state_early = base_model.forward_draft_or_large_model(
                in_tokens_small=in_token,
            )

            if step == 0:
                exited_hidden_states = None
            exited_hidden_states = hidden_state_early if exited_hidden_states is None \
                else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

            if step == round_speculative_steps:
                print(f"Draft step {step} reached round speculative step limit")
                break

            # Upper layers (this round's draft scratch cache).
            normed = _run_upper_layers(
                base_model, hidden_state_early, draft_upper_cache, return_normed=True,
            )
            predict_logits = head_model(normed[:, -1:, :]).float()
            predicted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)
            predict_score = predict_logits.softmax(dim=-1).max().item()
            adapter_confidences.append(predict_score)
            print(f"predicted_token: {predicted_token.item()}, predict_score: {predict_score}, token : {tokenizer.decode(predicted_token)}")

            # Mirror original threshold-based draft termination, although with
            # the full upper layers the score is virtually always >= threshold.
            if step > 0 and predict_score < threshold:
                print(f"Draft step {step}, predict_score {predict_score} < threshold {threshold}, stopping draft")
                break

            global_tokens[:, end_index] = predicted_token
            draft_token_ids.append(predicted_token.item())

            if predicted_token.item() in token_eos_set:
                end_index += 1
                break

            end_index += 1

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        draft_times.append(time.perf_counter() - t_draft_start)

        # ---- STEP 2+3: Verify and Accept ----
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_verify_start = time.perf_counter()

        output_length = end_index - start_index

        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        assert verify_cache_len == start_index, \
            f"Verify cache mismatch: {verify_cache_len} != {start_index}"

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

            if draft_id is not None and not is_last:
                adapter_total += 1
                if verify_id == draft_id:
                    adapter_correct += 1
                if i == 0:
                    adapter_first_total += 1
                    if verify_id == draft_id:
                        adapter_first_correct += 1

            if is_last or is_eos or is_mismatch:
                global_tokens[0, start_index + 1 + i] = verify_id
                print(f"Token {i} verification stopped here. is_last: {is_last}, is_eos: {is_eos}, is_mismatch: {is_mismatch}")
                start_index = start_index + 1 + i
                if is_eos:
                    stop = True
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        verify_times.append(time.perf_counter() - t_verify_start)

        accept_len = start_index - start_index_copy
        accept_length_list.append(accept_len)

        # ---- STEP 4: Trim caches (identical to original) ----
        draft_cache_len = base_model._get_layer_cache_length(0)
        if draft_cache_len > start_index:
            base_model.trim_draft_layers_cache(start_index)

        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        if verify_cache_len > start_index:
            base_model.trim_verify_layers_cache(start_index)

        base_model.past_key_values._seen_tokens = start_index
        assert base_model._get_layer_cache_length(0) == start_index, \
            f"Draft cache after trim: {base_model._get_layer_cache_length(0)} != {start_index}"
        assert base_model._get_layer_cache_length(early_exit_layer) == start_index, \
            f"Verify cache after trim: {base_model._get_layer_cache_length(early_exit_layer)} != {start_index}"

        # The per-round upper-layer scratch cache is rebuilt fresh next round.
        del draft_upper_cache

        if stop:
            break

    # ---- Final output ----
    output_ids = global_tokens[:, :start_index + 1]
    num_new_tokens = start_index + 1 - context_length
    print(f"New tokens: {tokenizer.batch_decode(output_ids[:, context_length:])}")
    print(f"Total new tokens generated: {num_new_tokens}")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start

    stats = _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens)

    adapter_acc = adapter_correct / adapter_total if adapter_total > 0 else 0
    adapter_first_acc = adapter_first_correct / adapter_first_total if adapter_first_total > 0 else 0
    avg_confidence = sum(adapter_confidences) / len(adapter_confidences) if adapter_confidences else 0
    draft_accept_per_round = adapter_correct / max(stats['total_rounds'], 1)

    print(
        f"[Spec-FULL] progress/round={stats['avg_accept_length']:.2f} "
        f"| draft_accept/round={draft_accept_per_round:.2f} "
        f"(counts every verified draft token, including EOS) "
        f"| rounds={stats['total_rounds']} | tokens={num_new_tokens} | context_len={context_length}"
    )
    print(
        f"[FullAdapter] top1={adapter_correct}/{adapter_total} ({adapter_acc:.1%}) "
        f"| first_top1={adapter_first_correct}/{adapter_first_total} ({adapter_first_acc:.1%}) "
        f"| avg_confidence={avg_confidence:.3f} | context_len={context_length}"
    )
    print(
        "[FullAdapter] Note: with the full upper layers as the draft head, top1 "
        "should be ~100%. Anything noticeably lower indicates a bug in the "
        "verify/trim/position-id logic, NOT in the adapter."
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

    input_length = inputs['input_ids'].shape[1]
    new_token_ids = output_ids[:, input_length:]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    return reply_text, past_key_values, stats
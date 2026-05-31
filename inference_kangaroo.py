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



# Per-step debug prints (token decode + stdout) are expensive CPU work. They
# run inside the timed draft/verify regions while the AR baseline has none, so
# leaving them on inflates spec time and understates the speedup. Keep OFF for
# any timing/benchmark run.
DEBUG = False


def apply_repetition_penalty(logits, prefix_ids, penalty, eos_ids_set=None):
    """Apply repetition penalty to a (1, vocab) logits tensor.

    - logits: (1, vocab) float tensor.
    - prefix_ids: 1D tensor of token ids generated in the current turn so far.
    - penalty > 1.0 lowers the probability of tokens already in prefix_ids.
    - eos_ids_set: ids that must NOT be penalized (so the model can still stop).

    Matches HF's RepetitionPenaltyLogitsProcessor semantics: positive logits are
    divided by penalty, negative logits are multiplied. Idempotent across
    duplicates (scatter writes once per unique id).
    """
    if penalty == 1.0 or prefix_ids.numel() == 0:
        return logits
    prefix_2d = prefix_ids.view(1, -1).to(device=logits.device, dtype=torch.long)
    score = torch.gather(logits, 1, prefix_2d)
    score = torch.where(score < 0, score * penalty, score / penalty)
    modified = logits.clone()
    modified.scatter_(1, prefix_2d, score)
    if eos_ids_set:
        for eos_id in eos_ids_set:
            modified[..., eos_id] = logits[..., eos_id]
    return modified




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
    repetition_penalty: float = 1.0,
    eos_token_ids=None,
    adapter_first_token: bool = False,
    use_gate: bool = False,
    gate_threshold: float = 0.5,
    no_reply_token_ids=None,
    must_reply_token_ids=None,
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
    # Prefer the caller's explicit eos_token_ids (typically from
    # model.generation_config.eos_token_id, which for Qwen2.5-VL is
    # [<|im_end|>=151645, <|endoftext|>=151643]). Fall back to the tokenizer's
    # single eos_token_id otherwise (often just one of them).
    if eos_token_ids is not None:
        token_eos_set = set(eos_token_ids) if isinstance(eos_token_ids, (list, tuple, set)) else {eos_token_ids}
        token_eos = next(iter(token_eos_set))
    else:
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

    # Base's first-token argmax (always computed; used as baseline for
    # disagreement logging, and as the actual first token when
    # adapter_first_token=False).
    base_first_token = torch.argmax(output.logits[:, -1, :], dim=-1)

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



    adapter_hidden_prefill, adapter_past_key_values = adapter_model.forward_early_stop(
        inputs_embeds=hidden_state_early,
        position_ids=prefill_position_ids,
        use_cache=True,
    )

    # Choose source of the very first generated token:
    #   - adapter_first_token=False (default): use base's prefill argmax (lossless probe)
    #   - adapter_first_token=True : use adapter's prediction at the last prefill
    #     position. This is for testing the adapter's standalone "should I
    #     respond now?" capability — when adapter disagrees with base, the
    #     output is no longer lossless to base, but you can read off the
    #     adapter's trigger behavior directly (e.g. how often it picks NO).
    if adapter_first_token:
        adapter_first_logits = head_model(adapter_hidden_prefill[:, -1:, :]).float()
        first_pos_logits = adapter_first_logits[:, 0, :]
        if repetition_penalty != 1.0:
            adapter_first_prefix = inputs['input_ids'][0]
            first_pos_logits = apply_repetition_penalty(
                first_pos_logits, adapter_first_prefix, repetition_penalty, None,
            )
        first_token = torch.argmax(first_pos_logits, dim=-1)
    else:
        first_token = base_first_token

    global_tokens[:, start_index] = first_token.item()
    first_token_disagree = (first_token.item() != base_first_token.item())

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    draft_times = []
    verify_times = []

    # ========== Gate decision (proactive-response trigger) ==========
    # If --use_gate is enabled AND the loaded adapter has a gate head,
    # ask it whether to respond at all. If p(respond) < gate_threshold,
    # short-circuit: emit "NO REPLY" tokens directly and skip spec.
    # If p(respond) >= threshold, inject "I must reply.\n" into the prompt
    # to override the adapter's lm_head NO-REPLY bias before the spec loop.
    gate_used = False
    gate_prob = None
    gate_skipped_spec = False
    gate_must_reply_meta = {
        'gate_must_reply_injected': False,
        'gate_must_reply_len': 0,
        'pre_must_reply_first_token_str': None,
        'post_must_reply_first_token_str': None,
        'blocked_no_reply_first_token_id': None,
    }
    if use_gate and getattr(adapter_model, 'use_gate_head', False):
        # Last prefill position == position right before generation
        gate_logit = adapter_model.gate_logits(adapter_hidden_prefill[:, -1:, :]).float()
        gate_prob = torch.sigmoid(gate_logit).item()
        gate_used = True
        if gate_prob < gate_threshold:
            # Gate says "do not respond" → write NO REPLY tokens and return
            no_reply_ids = list(no_reply_token_ids) if no_reply_token_ids else []
            if not no_reply_ids:
                # Fallback: tokenize the string on the fly
                no_reply_ids = tokenizer.encode("NO REPLY", add_special_tokens=False)
            # Ensure trailing EOS so downstream stops cleanly
            eos_for_close = next(iter(token_eos_set))
            if not no_reply_ids or no_reply_ids[-1] not in token_eos_set:
                no_reply_ids = no_reply_ids + [int(eos_for_close)]
            # Bound by max_length
            write_end = min(start_index + len(no_reply_ids), max_length)
            for i, tid in enumerate(no_reply_ids[: write_end - start_index]):
                global_tokens[0, start_index + i] = int(tid)
            start_index = write_end - 1   # last written position
            output_ids = global_tokens[:, : start_index + 1]
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.perf_counter() - t_start
            stats = _build_stats([], prefill_time, [], [], total_time,
                                 max(1, len(no_reply_ids)))
            stats['first_token_used_adapter'] = adapter_first_token
            stats['first_token_disagree_with_base'] = first_token_disagree
            stats['first_token_id'] = int(first_token.item())
            stats['base_first_token_id'] = int(base_first_token.item())
            stats['first_token_str'] = tokenizer.decode([int(first_token.item())])
            stats['base_first_token_str'] = tokenizer.decode([int(base_first_token.item())])
            stats['gate_used'] = True
            stats['gate_prob'] = gate_prob
            stats['gate_threshold'] = gate_threshold
            stats['gate_skipped_spec'] = True
            stats['gate_must_reply_injected'] = False
            return output_ids, base_model.past_key_values, stats
        else:
            # Gate says "respond" → inject "I must reply.\n" to override the
            # adapter's lm_head NO-REPLY bias, then take base's argmax at the
            # new last position as first_token. Subsequent spec runs on the
            # extended prompt (lossless to "base + must_reply prompt").
            mr_ids_list = list(must_reply_token_ids) if must_reply_token_ids else \
                tokenizer.encode("I must reply.\n", add_special_tokens=False)
            mr_ids = torch.tensor(mr_ids_list, dtype=torch.long, device=device).unsqueeze(0)
            N_mr = int(mr_ids.shape[1])

            # Re-prefill base + adapter on extended input. Simplest correct
            # path: rebuild the full forward (cheap relative to a full turn).
            new_input_ids = torch.cat([inputs['input_ids'], mr_ids], dim=1)
            new_attn_mask = None
            if inputs.get('attention_mask') is not None:
                new_attn_mask = torch.cat([
                    inputs['attention_mask'],
                    torch.ones((1, N_mr), dtype=inputs['attention_mask'].dtype, device=device),
                ], dim=1)

            mr_forward_kwargs = {
                'input_ids': new_input_ids,
                'attention_mask': new_attn_mask,
                'use_cache': True,
                'output_hidden_states': True,
                'return_dict': True,
                'past_key_values': None,
                'pixel_values': inputs.get('pixel_values'),
                'pixel_values_videos': inputs.get('pixel_values_videos'),
                'image_grid_thw': inputs.get('image_grid_thw'),
                'video_grid_thw': inputs.get('video_grid_thw'),
                'second_per_grid_ts': inputs.get('second_per_grid_ts'),
                'drop_method': 'none',
                'drop_threshold': 1.0,
                'drop_absolute': True,
            }
            mr_forward_kwargs = {k: v for k, v in mr_forward_kwargs.items() if v is not None}

            mr_output = base_model.model(**mr_forward_kwargs)
            base_model.past_key_values = mr_output.past_key_values
            mr_hidden_early = mr_output.hidden_states[early_exit_layer]

            mr_prefill_position_ids, _ = base_model.model.get_rope_index(
                new_input_ids,
                inputs.get('image_grid_thw'),
                inputs.get('video_grid_thw'),
                inputs.get('second_per_grid_ts'),
                new_attn_mask,
            )
            mr_prefill_position_ids = mr_prefill_position_ids.to(mr_hidden_early.device)

            _, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=mr_hidden_early,
                position_ids=mr_prefill_position_ids,
                use_cache=True,
            )

            # New first token = base's argmax AFTER must_reply.
            #
            # We do NOT hard-mask the NO-REPLY first token here. Masking forced
            # the base to "speak" on frames it wanted to stay silent on, turning
            # gate false-alarms into garbage output and breaking losslessness.
            # Instead we let the base pick freely: if it still chooses NO REPLY
            # after the must_reply nudge, that is its honest decision and stays
            # lossless.
            new_first_logits = mr_output.logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                new_first_logits = apply_repetition_penalty(
                    new_first_logits, new_input_ids[0], repetition_penalty, None,
                )
            # No first-token filtering: after the must_reply nudge the base picks
            # its first token freely (it may still choose NO REPLY / EOS — that is
            # its honest decision).
            blocked_first_token_id = None
            new_first_token = torch.argmax(new_first_logits, dim=-1)

            # Replace global_tokens with a bigger buffer that includes must_reply
            new_context_length = int(new_input_ids.shape[1])
            new_max_length = new_context_length + max_new_tokens
            new_global_tokens = torch.full(
                (batch_size, new_max_length), token_eos, dtype=torch.long, device=device,
            )
            new_global_tokens[:, :new_context_length] = new_input_ids
            global_tokens = new_global_tokens
            context_length = new_context_length
            max_length = new_max_length

            # Update first_token / start_index to point past must_reply
            pre_must_reply_first_token_str = tokenizer.decode([int(first_token.item())])
            first_token = new_first_token
            start_index = context_length
            global_tokens[:, start_index] = first_token.item()
            prefill_position_ids = mr_prefill_position_ids

            gate_must_reply_meta['gate_must_reply_injected'] = True
            gate_must_reply_meta['gate_must_reply_len'] = N_mr
            gate_must_reply_meta['pre_must_reply_first_token_str'] = pre_must_reply_first_token_str
            gate_must_reply_meta['post_must_reply_first_token_str'] = tokenizer.decode([int(first_token.item())])
            gate_must_reply_meta['blocked_no_reply_first_token_id'] = blocked_first_token_id

    if DEBUG:
        print(f" first token :{tokenizer.decode(first_token)}")
    if first_token.item() in token_eos_set:
        output_ids = global_tokens[:, :start_index + 1]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t_start
        stats = _build_stats([], prefill_time, [], [], total_time, 1)
        stats['first_token_used_adapter'] = adapter_first_token
        stats['first_token_disagree_with_base'] = first_token_disagree
        stats['first_token_id'] = int(first_token.item())
        stats['base_first_token_id'] = int(base_first_token.item())
        stats['first_token_str'] = tokenizer.decode([int(first_token.item())])
        stats['base_first_token_str'] = tokenizer.decode([int(base_first_token.item())])
        stats['gate_used'] = gate_used
        stats['gate_prob'] = gate_prob
        stats['gate_threshold'] = gate_threshold if use_gate else None
        stats['gate_skipped_spec'] = False
        stats.update(gate_must_reply_meta)
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
            if DEBUG:
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
                if DEBUG:
                    print(f"Draft step {step} reached round speculative step limit")
                break
            if step > 0 and predict_score < threshold:
                if DEBUG:
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
            # Apply repetition penalty over the FULL sequence so far (prompt +
            # history + this turn's generated tokens), matching HF's
            # RepetitionPenaltyLogitsProcessor exactly. Do NOT protect EOS —
            # HF doesn't either; protecting it makes spec output longer than
            # MMDuet2's base output.
            draft_pos_logits = predict_logits[:, -1, :]
            if repetition_penalty != 1.0:
                draft_prefix = global_tokens[0, :end_index]
                draft_pos_logits = apply_repetition_penalty(
                    draft_pos_logits, draft_prefix, repetition_penalty, None,
                )
            predicted_token = torch.argmax(draft_pos_logits, dim=-1)

            predict_score = predict_logits.softmax(dim=-1).max().item()
            # =====================================
            adapter_confidences.append(predict_score)
            #=======================================
            if DEBUG:
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
        if repetition_penalty != 1.0:
            # Each verify position i predicts token at start_index+1+i. Its
            # prefix is the FULL sequence so far global_tokens[: start_index+1+i]
            # (prompt + history + generated). Matches HF's per-step advancing
            # prefix exactly; do NOT protect EOS.
            verify_ids = []
            for j in range(verify_logits.shape[1]):
                pos_logits = verify_logits[:, j, :]
                pos_prefix = global_tokens[0, :start_index + 1 + j]
                pos_logits = apply_repetition_penalty(
                    pos_logits, pos_prefix, repetition_penalty, None,
                )
                verify_ids.append(int(torch.argmax(pos_logits, dim=-1).item()))
        else:
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
            if DEBUG:
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
                if DEBUG:
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
    if DEBUG:
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
    stats['first_token_used_adapter'] = adapter_first_token
    stats['first_token_disagree_with_base'] = first_token_disagree
    stats['first_token_id'] = int(first_token.item())
    stats['base_first_token_id'] = int(base_first_token.item())
    stats['first_token_str'] = tokenizer.decode([int(first_token.item())])
    stats['base_first_token_str'] = tokenizer.decode([int(base_first_token.item())])
    stats['gate_used'] = gate_used
    stats['gate_prob'] = gate_prob
    stats['gate_threshold'] = gate_threshold if use_gate else None
    stats['gate_skipped_spec'] = False
    stats.update(gate_must_reply_meta)
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
    repetition_penalty: float = 1.0,
    eos_token_ids=None,
    adapter_first_token: bool = False,
    use_gate: bool = False,
    gate_threshold: float = 0.5,
    no_reply_token_ids=None,
    must_reply_token_ids=None,
    must_reply_text: str = "I must reply.\n",
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
        repetition_penalty=repetition_penalty,
        eos_token_ids=eos_token_ids,
        adapter_first_token=adapter_first_token,
        use_gate=use_gate,
        gate_threshold=gate_threshold,
        no_reply_token_ids=no_reply_token_ids,
        must_reply_token_ids=must_reply_token_ids,
    )
    # print(f"Speculative generation completed. Stats: {stats}")

    # When gate injected "I must reply." into the prompt, output_ids now
    # contains [original prompt | must_reply tokens | generated tokens].
    # Skip both the original prompt and the must_reply chunk so the visible
    # reply_text only contains the actual generated content.
    input_length = inputs['input_ids'].shape[1]
    mr_len = int(stats.get('gate_must_reply_len', 0) or 0)
    new_token_ids = output_ids[:, input_length + mr_len:]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    # Safety net: if for any reason must_reply text leaks into the decoded
    # string (e.g. tokenization boundary differences), strip it.
    if stats.get('gate_must_reply_injected') and must_reply_text:
        if reply_text.startswith(must_reply_text):
            reply_text = reply_text[len(must_reply_text):]

    return reply_text, past_key_values, stats
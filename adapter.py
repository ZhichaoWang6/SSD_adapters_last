"""
Adapter model for self-speculative decoding on Qwen2.5-VL.
A lightweight single-layer transformer decoder that bridges early-exit hidden states
to the full model's output distribution.

Adapted from Kangaroo (https://github.com/Equationliu/Kangaroo) for Qwen2.5 architecture.
"""

import json
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MultimodalRotaryEmbedding(nn.Module):
    """3D mRoPE for Qwen2.5-VL style multimodal position encoding.

    Accepts position_ids of shape (3, batch, seq_len) where the 3 channels
    are (T, H, W). For pure text tokens T=H=W=sequence position.
    Returns cos/sin of shape (3, batch, seq_len, head_dim).
    """

    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        # position_ids: (3, batch, seq_len) for 3D mRoPE
        # Or (batch, seq_len) for 1D — auto-broadcast to 3 channels (text-style T=H=W).
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # position_ids: (3, batch, seq_len)
        position_ids_f = position_ids.float()
        # inv_freq: (dim/2,)  →  broadcast to (1, 1, 1, dim/2)
        # position_ids: (3, batch, seq_len)  →  unsqueeze to (3, batch, seq_len, 1)
        # freqs: (3, batch, seq_len, dim/2)
        freqs = position_ids_f.unsqueeze(-1) * self.inv_freq[None, None, None, :].float()
        emb = torch.cat([freqs, freqs], dim=-1)  # (3, batch, seq_len, dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


# Kept for backward compatibility (unused after switching to mRoPE).
class RotaryEmbedding(nn.Module):
    """Standard 1D rotary position embeddings (legacy)."""

    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).float()
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=position_ids.dtype if position_ids.is_floating_point() else torch.float32), \
               sin.to(dtype=position_ids.dtype if position_ids.is_floating_point() else torch.float32)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section):
    """
    cos, sin: (3, batch, seq_len, head_dim)
    mrope_section: list of 3 ints summing to head_dim/2
                   e.g. [16, 24, 24] for head_dim=128.
    """
    # mrope_section halves are doubled because cos/sin span full head_dim.
    section = list(mrope_section) * 2  # e.g. [16, 24, 24, 16, 24, 24]
    # cos.split(section, dim=-1) → 6 chunks of (3, batch, seq_len, ...)
    # For chunk i, take channel i % 3 (T/H/W repeating).
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(section, dim=-1))], dim=-1
    ).unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(section, dim=-1))], dim=-1
    ).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin):
    """Legacy 1D RoPE (unused)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class AdapterAttention(nn.Module):
    """Multi-head attention with GQA support and tuple-based KV cache."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        rope_theta = getattr(config, 'rope_theta', 1000000.0)
        max_pos = getattr(config, 'max_position_embeddings', 32768)
        self.rotary_emb = MultimodalRotaryEmbedding(self.head_dim, max_position_embeddings=max_pos, base=rope_theta)
        # mrope_section: how head_dim/2 is split into (T, H, W). Read from config; default for Qwen2.5-VL.
        self.mrope_section = getattr(config, 'mrope_section', [16, 24, 24])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        target_dtype = hidden_states.dtype  # 记录输入dtype，后续统一用这个

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section
        )

        # 统一dtype，防止RoPE引入的float32和其他张量不一致
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        # PyTorch SDPA picks the fastest backend automatically (flash attention
        # for long seqs, mem-efficient for borderline shapes, math for short
        # seqs). Equivalent to the previous manual matmul + softmax + matmul
        # path but avoids materialising the [B, H, L, L] attention scores
        # tensor (~1 GB at L=4000) and runs ~2-3x faster on long sequences.
        # The additive `attention_mask` (causal + padding bias) is passed
        # straight through; mRoPE and the (K, V) cache are unchanged.
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states_expanded,
            value_states_expanded,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class AdapterDecoderLayer(nn.Module):
    """A single transformer decoder layer for the adapter."""

    def __init__(self, config):
        super().__init__()
        self.use_mlp = getattr(config, "use_mlp", True)
        self.self_attn = AdapterAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        if self.use_mlp:
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        if self.use_mlp:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
            hidden_states = self.down_proj(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value


def _make_causal_mask(input_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class AdapterModel(nn.Module):
    """
    Lightweight adapter that maps early-exit hidden states to full model output space.
    Architecture: N decoder layers (configurable via config.num_adapter_layers) + final RMSNorm.
    Uses tuple-based KV cache for compatibility with Kangaroo-style speculative decoding.
    """

    def __init__(self, config):
        super().__init__()
        self.gradient_checkpointing = False
        self.padding_idx = getattr(config, 'pad_token_id', 0)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_adapter_layers = getattr(config, 'num_adapter_layers', 1)

        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.layers = nn.ModuleList(
            [AdapterDecoderLayer(config) for _ in range(self.num_adapter_layers)]
        )  # N 层解码层 (TwigVLM-style stack)

        # Optional binary gate head for the proactive-response trigger. When
        # enabled, it reads the adapter's normed output at a position and emits a
        # single logit = "should respond here?" (sigmoid -> p(respond)). This is
        # a dedicated 2-class objective (BCE + pos_weight), separate from the
        # lm_head next-token path used for lossless speculative drafting, so the
        # trigger decision is not drowned in the 150k-way vocab CE.
        self.use_gate_head = getattr(config, 'gate_head', False)
        if self.use_gate_head:
            self.gate_head = nn.Linear(config.hidden_size, 1, bias=True)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        dtype = inputs_embeds.dtype  # 跟随输入dtype，不写死float32
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        return self.forward_early_stop(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def forward_early_stop(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_length, _ = inputs_embeds.shape
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            # Default: 1D arange. mRoPE will auto-broadcast to 3 channels (text-style T=H=W).
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            # Accept 3D mRoPE position_ids (3, batch, seq_len) or 2D (batch, seq_len).
            # mRoPE forward handles the dim==2 case by auto-broadcasting.
            if position_ids.dim() == 3:
                # (3, batch, seq_len)
                assert position_ids.shape[0] == 3, (
                    f"Expected 3 channels (T,H,W), got {position_ids.shape[0]}"
                )
                position_ids = position_ids.long()
            else:
                position_ids = position_ids.view(batch_size, seq_length).long()

        seq_length_with_past = seq_length + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
        )

        hidden_states = inputs_embeds
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, None, False)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states, attention_mask, position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, next_decoder_cache
        return hidden_states

    def gate_logits(self, hidden_states):
        """Per-position gate logit from already-normed adapter output.

        hidden_states: (B, L, D) = the return of forward_early_stop / forward.
        Returns (B, L) logits. sigmoid(logit) = p(respond at this position).
        """
        if not self.use_gate_head:
            raise RuntimeError("gate_head is disabled; set config.gate_head=True")
        return self.gate_head(hidden_states).squeeze(-1)


def create_adapter_config(base_model_path, num_adapter_layers: int = 1, gate_head: bool = False):
    """Create adapter config from base model config.

    Args:
        base_model_path: Path to the base Qwen2.5-VL checkpoint.
        num_adapter_layers: Number of stacked decoder layers in the adapter.
        gate_head: Add a binary proactive-response gate head (Linear(D,1)).
    """
    base_config = AutoConfig.from_pretrained(base_model_path)
    # Read mrope_section from base config's rope_scaling.
    rope_scaling = getattr(base_config, 'rope_scaling', None) or {}
    mrope_section = rope_scaling.get('mrope_section', [16, 24, 24])
    adapter_config = type(base_config)(
        hidden_size=base_config.hidden_size,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        rms_norm_eps=base_config.rms_norm_eps,
        vocab_size=base_config.vocab_size,
        max_position_embeddings=base_config.max_position_embeddings,
        rope_theta=getattr(base_config, 'rope_theta', 1000000.0),
        pad_token_id=getattr(base_config, 'pad_token_id', 0),
        use_mlp=True,
    )
    # Qwen2_5_VLConfig doesn't accept mrope_section as a kwarg; set it as
    # a custom attribute after construction so AdapterAttention can read it.
    adapter_config.mrope_section = mrope_section
    adapter_config.num_adapter_layers = num_adapter_layers
    adapter_config.gate_head = gate_head
    return adapter_config


def init_adapter_from_base_layer(adapter_model, base_model_path, source_layer):
    """Initialize adapter's N decoder layers from consecutive base model layers
    starting at ``source_layer``. Also copies ``base.norm`` into ``adapter.norm``.
    EAGLE/TwigVLM-style init that gives the adapter a strong prior.

    For an adapter with N layers and source_layer S, copies:
        base.layers[S]   -> adapter.layers[0]
        base.layers[S+1] -> adapter.layers[1]
        ...
        base.layers[S+N-1] -> adapter.layers[N-1]
        base.norm        -> adapter.norm

    Returns the number of parameters successfully copied.
    """
    from safetensors import safe_open

    num_layers = len(adapter_model.layers)
    layer_indices = list(range(source_layer, source_layer + num_layers))
    layer_prefixes = [f"model.layers.{idx}." for idx in layer_indices]
    norm_key = "model.norm.weight"

    def _wanted(key):
        if key == norm_key:
            return True
        return any(key.startswith(p) for p in layer_prefixes)

    index_path = os.path.join(base_model_path, "model.safetensors.index.json")
    weights = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            weight_map = json.loads(f.read())["weight_map"]
        files_needed = set()
        for key, fname in weight_map.items():
            if _wanted(key):
                files_needed.add(fname)
        for fname in files_needed:
            with safe_open(os.path.join(base_model_path, fname), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if _wanted(key):
                        weights[key] = f.get_tensor(key)
    else:
        single = os.path.join(base_model_path, "model.safetensors")
        if os.path.exists(single):
            with safe_open(single, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if _wanted(key):
                        weights[key] = f.get_tensor(key)
        else:
            raise FileNotFoundError(f"No safetensors index or single file in {base_model_path}")

    if not weights:
        raise RuntimeError(
            f"No weights found for layers {layer_indices} in {base_model_path}."
        )

    mappings = {}
    for i, base_idx in enumerate(layer_indices):
        layer_prefix = f"model.layers.{base_idx}."
        adapter_prefix = f"layers.{i}."
        mappings.update({
            f"{layer_prefix}self_attn.q_proj.weight":         f"{adapter_prefix}self_attn.q_proj.weight",
            f"{layer_prefix}self_attn.q_proj.bias":           f"{adapter_prefix}self_attn.q_proj.bias",
            f"{layer_prefix}self_attn.k_proj.weight":         f"{adapter_prefix}self_attn.k_proj.weight",
            f"{layer_prefix}self_attn.k_proj.bias":           f"{adapter_prefix}self_attn.k_proj.bias",
            f"{layer_prefix}self_attn.v_proj.weight":         f"{adapter_prefix}self_attn.v_proj.weight",
            f"{layer_prefix}self_attn.v_proj.bias":           f"{adapter_prefix}self_attn.v_proj.bias",
            f"{layer_prefix}self_attn.o_proj.weight":         f"{adapter_prefix}self_attn.o_proj.weight",
            f"{layer_prefix}input_layernorm.weight":          f"{adapter_prefix}input_layernorm.weight",
            f"{layer_prefix}post_attention_layernorm.weight": f"{adapter_prefix}post_attention_layernorm.weight",
            f"{layer_prefix}mlp.gate_proj.weight":            f"{adapter_prefix}gate_proj.weight",
            f"{layer_prefix}mlp.up_proj.weight":              f"{adapter_prefix}up_proj.weight",
            f"{layer_prefix}mlp.down_proj.weight":            f"{adapter_prefix}down_proj.weight",
        })
    mappings[norm_key] = "norm.weight"

    adapter_state = adapter_model.state_dict()
    new_state = {}
    matched, skipped = 0, []
    n_params = 0
    for src_key, dst_key in mappings.items():
        if src_key in weights and dst_key in adapter_state:
            tensor = weights[src_key]
            if tensor.shape != adapter_state[dst_key].shape:
                skipped.append((dst_key, f"shape mismatch {tuple(tensor.shape)} vs {tuple(adapter_state[dst_key].shape)}"))
                continue
            new_state[dst_key] = tensor.to(adapter_state[dst_key].dtype)
            matched += 1
            n_params += tensor.numel()
        else:
            skipped.append((dst_key, f"src key '{src_key}' not in base weights" if src_key not in weights else "dst not in adapter"))

    adapter_state.update(new_state)
    adapter_model.load_state_dict(adapter_state, strict=False)
    last_idx = layer_indices[-1] if layer_indices else source_layer
    print(f"[init_adapter_from_base] copied {matched}/{len(mappings)} keys "
          f"from base layers {source_layer}..{last_idx} ({n_params/1e6:.2f}M params)")
    for dst, reason in skipped:
        print(f"  [skip] {dst}: {reason}")
    return matched


if __name__ == "__main__":
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    import torch

    DEVICE     = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "/data/wangzhichao/models/Qwen2.5-VL-3B-Instruct"
    CKPT_PATH  = "/data/wangzhichao/projects/SSD/SSD2/kangaroo/training_data/data_0.ckpt"
    EXIT_LAYER = 2

    def section(title): print(f"\n{'='*60}\n  {title}\n{'='*60}")
    def ok(msg):        print(f"  ✓  {msg}")
    def info(msg):      print(f"  →  {msg}")

    section("1. 读取 ckpt")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ok(f"加载成功: {CKPT_PATH}")
    info(f"包含的键: {list(ckpt.keys())}")

    input_ids    = ckpt["input_ids"]
    loss_mask    = ckpt["loss_mask"]
    early_hidden = ckpt[f"hidden_state_layer{EXIT_LAYER}"]
    final_hidden = ckpt["hidden_state"]

    seq_len, hidden_size = early_hidden.shape
    info(f"seq_len: {seq_len},  hidden_size: {hidden_size}")
    info(f"loss_mask 有效 token 数: {int(loss_mask.sum().item())}")

    section("2. 加载 Qwen 模型")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map=DEVICE
    )
    qwen.eval()
    lm_head = qwen.lm_head
    ok(f"lm_head: {lm_head.weight.shape}")

    section("3. 创建 AdapterModel")
    adapter_config = create_adapter_config(MODEL_PATH, num_adapter_layers=1)
    adapter = AdapterModel(adapter_config).to(DEVICE).to(torch.float32)
    adapter.eval()
    n_params = sum(p.numel() for p in adapter.parameters())
    ok(f"参数量: {n_params/1e6:.2f}M")

    section("4. Early hidden → AdapterModel")
    inputs_embeds = early_hidden.unsqueeze(0).to(DEVICE).to(torch.float32)
    with torch.no_grad():
        adapter_out = adapter(inputs_embeds=inputs_embeds)
    ok(f"adapter 输出: {adapter_out.shape}  dtype={adapter_out.dtype}")
    info(f"含 NaN: {torch.isnan(adapter_out).any().item()}  (期望 False)")

    section("5. AdapterModel → lm_head → logits")
    with torch.no_grad():
        adapter_logits = lm_head(adapter_out)
        final_logits   = lm_head(final_hidden.unsqueeze(0).to(DEVICE).to(torch.float32))
    ok(f"adapter logits: {adapter_logits.shape}")
    ok(f"full    logits: {final_logits.shape}")

    section("6. 对比预测（assistant token 位置）")
    assistant_pos = loss_mask.nonzero(as_tuple=True)[0].tolist()
    adapter_pred  = adapter_logits[0].argmax(dim=-1).cpu()
    full_pred     = final_logits[0].argmax(dim=-1).cpu()

    print(f"\n  {'位置':>6}  {'真实token':>12}  {'完整模型':>12}  {'adapter':>12}  {'一致':>4}")
    print(f"  {'-'*56}")
    match = 0
    for pos in assistant_pos[:20]:
        t  = repr(tokenizer.decode([input_ids[pos].item()]))
        fp = repr(tokenizer.decode([full_pred[pos].item()]))
        ap = repr(tokenizer.decode([adapter_pred[pos].item()]))
        eq = "✓" if full_pred[pos] == adapter_pred[pos] else "✗"
        if full_pred[pos] == adapter_pred[pos]:
            match += 1
        print(f"  {pos:>6}  {t:>12}  {fp:>12}  {ap:>12}  {eq:>4}")

    section("7. MSE Loss（adapter_out vs final_hidden）")
    target = final_hidden.unsqueeze(0).to(DEVICE).to(torch.float32)
    mse_all = F.mse_loss(adapter_out, target).item()
    info(f"全序列 MSE: {mse_all:.6f}  （未训练时偏大，训练目标是让此值降低）")

    if assistant_pos:
        pos_t = torch.tensor(assistant_pos)
        mse_asst = F.mse_loss(adapter_out[0, pos_t], target[0, pos_t]).item()
        info(f"assistant 位置 MSE: {mse_asst:.6f}")

    print(f"\n{'='*60}")
    print("  ALL STEPS PASSED ✓")
    print(f"  seq_len:      {seq_len}")
    print(f"  hidden_size:  {hidden_size}")
    print(f"  adapter 参数: {n_params/1e6:.2f}M")
    print(f"  token 一致率: {match}/{min(20, len(assistant_pos))}  （未训练，低正常）")
    print(f"  全序列 MSE:   {mse_all:.6f}")
    print('='*60)
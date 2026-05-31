import collections, math, json, copy, re, os, time
from dataclasses import asdict, dataclass, field
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoProcessor
from torchvision.io import read_video

from qwen_vl_utils import process_vision_info
from model import Qwen2_5_VLForConditionalGeneration
import logging

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import speculative_generate_for_streaming
from ar_generate import autoregressive_manual_baseline

logger = transformers.logging.get_logger('inference')
logger.setLevel(logging.INFO)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--llm_pretrained", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames. Your responses should include information from the video since your last reply (if any). If the information in this segment of the video cannot answer the question, output \"NO REPLY\".")
    parser.add_argument("--input_assistant_turns", action="store_true")
    parser.add_argument("--test_fname", type=str, default="./data/annotations/2fps/ego_dataset.json")
    parser.add_argument("--output_fname", type=str, default="./outputs/2fps/preds_auto_full_10_no_reply.jsonl")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None, help="默认跑完整个数据集")
    parser.add_argument("--max_turns", type=int, default=None, help="最多跑几轮对话，None表示跑完")
    parser.add_argument("--warmup_turns", type=int, default=2,
                        help="正式记录前空跑几轮做 GPU 预热（kernel 编译/显存分配），不计入统计。0 表示不预热。")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="logits 上的重复惩罚，spec/AR/draft 三处统一套，仍 lossless。"
                             "不传则从 model.generation_config 读，读不到则 1.0（关闭）。")
    parser.add_argument("--adapter_first_token", action="store_true",
                        help="Use adapter's prediction for the first token of each turn "
                             "(i.e. the 'should I respond' decision) instead of base "
                             "prefill's argmax. Spec output is no longer lossless to base "
                             "when enabled; probe of adapter's standalone trigger capability.")
    parser.add_argument("--use_gate", action="store_true",
                        help="Use adapter's binary gate head (trained on the dreamy-ride "
                             "branch) as the proactive-response trigger. When gate sigmoid "
                             "p(respond) < --gate_threshold, the turn emits 'NO REPLY' "
                             "directly and skips speculative decoding entirely. Requires "
                             "an adapter checkpoint whose adapter_config.json has "
                             "gate_head=true.")
    parser.add_argument("--gate_threshold", type=float, default=0.5,
                        help="Sigmoid threshold for the gate head; respond iff p >= this.")
    parser.add_argument("--device", type=str, default="cuda:4")

    # generation args
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)

    # speculative decoding args
    parser.add_argument("--use_speculative_decoding", action="store_true")
    parser.add_argument("--compare_AR_SSD", action="store_true")
    parser.add_argument("--adapter_path", type=str, default="/data/wangzhichao/projects/SSD_full_history/adapter_checkpoints/10_no_reply/epochs/epoch025_acc0.9144_accept0.9070_loss0.8976")
    parser.add_argument("--exit_layer", type=int, default=2)
    parser.add_argument("--num_adapter_layers", type=int, default=1,
                        help="Number of stacked transformer layers in the adapter. "
                             "Must match the saved adapter checkpoint (kangaroo_model.py "
                             "also auto-detects from adapter_config.json and will override).")
    parser.add_argument("--disable_adapter_mlp", action="store_true")
    parser.add_argument("--speculative_threshold", type=float, default=0.6)
    parser.add_argument("--speculative_steps", type=int, default=6)

    args = parser.parse_args()
    return args


# tailored for timechat-online (or, say, Qwen-2.5 VL)
class ProactiveInferenceClient:
    def __init__(self, args=None, model=None, processor=None) -> None:
        self.args = args
        self.device = args.device
        self.use_speculative_decoding = getattr(args, 'use_speculative_decoding', False)
        self.compare_AR_SSD = getattr(args, 'compare_AR_SSD', False)

        if self.use_speculative_decoding and model is None:
            logger.info("Loading model with speculative decoding (Kangaroo adapter)")
            self.kangaroo_model = KangarooQwenModel(
                base_model_path=args.llm_pretrained,
                adapter_model_path=args.adapter_path,
                early_exit_layer=args.exit_layer,
                use_adapter_mlp=None if not args.disable_adapter_mlp else False,
                num_adapter_layers=args.num_adapter_layers,
                dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
            ).to(args.device)
            self.model = self.kangaroo_model.base_model.model  # raw Qwen2.5-VL model for compatibility
            self.speculative_threshold = args.speculative_threshold
            self.speculative_steps = args.speculative_steps
            self.exit_layer = args.exit_layer
        else:
            self.kangaroo_model = None
            self.model = model if model is not None else Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.llm_pretrained, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation,
            ).eval().to(args.device)

        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(
            args.llm_pretrained
        )
        self.system_prompt = args.system_prompt
        logger.info("using system prompt:" + self.system_prompt)
        self.input_assistant_turns = args.input_assistant_turns
        logger.info(f"using assistant turns in input: {self.input_assistant_turns}")

        self.do_sample = args.do_sample
        self.temperature = args.temperature
        self.top_k = args.top_k

        # Resolve repetition_penalty: CLI > model.generation_config > 1.0.
        # Resolve EOS ids:           model.generation_config > tokenizer.eos.
        # We pass these into spec / AR / draft three places so the greedy
        # logits path matches what model.generate() would apply, and the eos
        # set covers BOTH <|im_end|> and <|endoftext|> as Qwen2.5-VL ships.
        gen_cfg = getattr(self.model, 'generation_config', None)
        if args.repetition_penalty is not None:
            self.repetition_penalty = float(args.repetition_penalty)
        else:
            self.repetition_penalty = float(getattr(gen_cfg, 'repetition_penalty', 1.0) or 1.0)
        cfg_eos = getattr(gen_cfg, 'eos_token_id', None) if gen_cfg is not None else None
        if cfg_eos is not None:
            self.eos_token_ids = list(cfg_eos) if isinstance(cfg_eos, (list, tuple, set)) else [int(cfg_eos)]
        else:
            tok_eos = self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
            self.eos_token_ids = list(tok_eos) if isinstance(tok_eos, (list, tuple, set)) else (
                [int(tok_eos)] if tok_eos is not None else None
            )
        self.adapter_first_token = bool(getattr(args, 'adapter_first_token', False))
        self.use_gate = bool(getattr(args, 'use_gate', False))
        self.gate_threshold = float(getattr(args, 'gate_threshold', 0.5))
        # Pre-tokenize the two literal strings the gate path uses, so we don't
        # re-encode them every turn:
        #   "NO REPLY"        — what to emit when gate says skip
        #   "I must reply.\n" — what to inject into the prompt when gate says
        #                       respond (overrides the adapter lm_head's
        #                       NO REPLY bias so base actually starts content)
        tok = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        self.no_reply_token_ids = tok.encode("NO REPLY", add_special_tokens=False)
        self.must_reply_text = "I must reply.\n"
        self.must_reply_token_ids = tok.encode(self.must_reply_text, add_special_tokens=False)
        logger.info(f"repetition_penalty={self.repetition_penalty}, eos_token_ids={self.eos_token_ids}, "
                    f"adapter_first_token={self.adapter_first_token}, "
                    f"use_gate={self.use_gate}, gate_threshold={self.gate_threshold}, "
                    f"no_reply_token_ids={self.no_reply_token_ids}, "
                    f"must_reply_token_ids={self.must_reply_token_ids}")

        self.history = list()
        self.prev_frame_before_token_drop = None    # for dynamic token drop
        self.prev_image_inputs = list()
        self.prev_video_inputs = list()
        self.all_keep_masks = list()
        # Generation speed tracking
        self.generation_stats = []
        self.reset()

    def set_fps(self, fps=None, frame_interval=None):
        assert fps is not None or frame_interval is not None
        assert not (fps is not None and frame_interval is not None)
        if fps is not None:
            self.frame_fps = fps
            self.frame_interval = 1 / self.frame_fps
        else:
            self.frame_interval = frame_interval
            self.frame_fps = 1 / self.frame_interval

    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.frame_idx = 0
        self.video_tensor = None
        self.past_key_values = None
        self.past_key_values_ar = None  # Separate KV cache for AR baseline comparison
        self.history = list()
        self.prev_frame_before_token_drop = None

        self.prev_image_inputs = list()
        self.prev_video_inputs = list()
        self.all_keep_masks = list()
        self.generation_stats = []
        if hasattr(self.model, 'reset_status'):
            self.model.reset_status()
        if self.kangaroo_model is not None:
            self.kangaroo_model.reset_status()

    def input_query_stream(self, conversation):
        if conversation[0]['role'] != 'system':
            self.query_queue.append({'role': 'system', 'content': self.system_prompt})
        else:
            logger.info(f"using system prompt in data instead of default system prompt: {conversation[0]['content']=}")
            self.query_queue.append(conversation[0])
            del conversation[0]
        for turn in conversation:
            if self.input_assistant_turns or turn['role'] == 'user':
                self.query_queue.append(turn)

    def _recursive_stat_num_frames(self, inputs):
        num_frames = 0
        if isinstance(inputs, (list, tuple)):
            for input in inputs:
                if isinstance(input, (torch.Tensor, Image.Image, np.ndarray)):
                    num_frames += 1
                elif isinstance(input, (list, tuple)):
                    num_frames += self._recursive_stat_num_frames(input)
        return num_frames

    def _encode_query(self):
        newly_added_turns = list()
        while True:
            query = self.query_queue.popleft()
            self.history.append(query)
            newly_added_turns.append(query)
            if query['role'] in ['system', 'assistant'] or query.get('skip_inference', False):
                pass
            else:
                break

        text = self.processor.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True,
        )

        prompt_text = text

        new_image_inputs, new_video_inputs = process_vision_info(newly_added_turns)
        if new_image_inputs is not None:
            self.prev_image_inputs.extend(new_image_inputs)
        if new_video_inputs is not None:
            self.prev_video_inputs.extend(new_video_inputs)
        image_inputs = copy.deepcopy(self.prev_image_inputs) if self.prev_image_inputs else None
        video_inputs = copy.deepcopy(self.prev_video_inputs) if self.prev_video_inputs else None

        num_frames = self._recursive_stat_num_frames(new_image_inputs) + self._recursive_stat_num_frames(new_video_inputs)
        self.video_time += num_frames * self.frame_interval
        self.history[-1]['time'] = self.video_time

        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        context_len = int(inputs.input_ids.shape[1])
        inputs = inputs.to(self.device)
        # print("==============================================================")

        if self.model.model.all_keep_masks and any(not m.all() for m in self.model.model.all_keep_masks):
            assert inputs.input_ids.size(0) == 1, "token drop in inference only support batch size 1 now"
            keep_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
            old_keep_mask = torch.cat(self.model.model.all_keep_masks, dim=1)
            copy_len = min(old_keep_mask.size(1), keep_mask.size(1))
            keep_mask[:, :copy_len] = old_keep_mask[:, :copy_len]
            inputs['input_ids'] = inputs.input_ids[keep_mask].unsqueeze(0)
            inputs['attention_mask'] = inputs.attention_mask[keep_mask].unsqueeze(0)

        if self.use_speculative_decoding and self.kangaroo_model is not None:
            # ---- Run speculative decoding ----
            reply_text, self.past_key_values, spec_stats = speculative_generate_for_streaming(
                model=self.kangaroo_model,
                inputs=inputs,
                processor=self.processor,
                # This pipeline rebuilds the full conversation prompt each turn.
                # Reusing the previous KV cache here would append the whole prompt
                # on top of already-cached history and corrupt cache lengths.
                past_key_values=None,
                max_new_tokens=512,
                early_exit_layer=self.exit_layer,
                speculative_steps=self.speculative_steps,
                threshold=self.speculative_threshold,
                repetition_penalty=self.repetition_penalty,
                eos_token_ids=self.eos_token_ids,
                adapter_first_token=self.adapter_first_token,
                use_gate=self.use_gate,
                gate_threshold=self.gate_threshold,
                no_reply_token_ids=self.no_reply_token_ids,
                must_reply_token_ids=self.must_reply_token_ids,
                must_reply_text=self.must_reply_text,
            )
            spec_stats["context_len"] = context_len
            combined_stats = {'speculative': spec_stats}

            # 重置模型状态，确保 AR baseline 和 speculative decoding 起点一致
            if hasattr(self.model, 'reset_status'):
                self.model.reset_status()
            if self.kangaroo_model is not None:
                self.kangaroo_model.reset_status()
 
            # print("======================== AR (manual) ==================================")
            # print("Running manual AR baseline with same forward path...")
 
            ar_text, _, ar_stats = autoregressive_manual_baseline(
                model=self.kangaroo_model,
                inputs=inputs,
                processor=self.processor,
                max_new_tokens=512,
                early_exit_layer=self.exit_layer,
                repetition_penalty=self.repetition_penalty,
                eos_token_ids=self.eos_token_ids,
            )
            ar_stats["context_len"] = context_len
 
            # print(f"AR (manual) generated text: {ar_text}")
            # print(f"AR (manual) stats: {ar_stats}")
 
            # 对比
            print(f"\\n===== Lossless Check =====")
            print(f"Speculative output: {reply_text}")
            print(f"AR manual output:   {ar_text}")
            output_match = reply_text == ar_text
            history_fallback_to_ar = False
            print(f"Exact match: {output_match}")
 
            if not output_match:
                print("WARNING: Outputs differ! Speculative decoding is NOT lossless.")
                print("Using AR output for conversation history to avoid contaminating later turns.")
                history_fallback_to_ar = True
            else:
                print("OK: Outputs match. Speculative decoding is lossless.")
 
            if ar_stats['total_time'] > 0 and spec_stats['total_time'] > 0:
                speedup_decode = spec_stats['decode_tokens_per_second'] / ar_stats['decode_tokens_per_second']
                turn_record = {
                    'speculative': spec_stats,
                    'autoregressive': ar_stats,
                    'output_match': output_match,
                    'history_fallback_to_ar': history_fallback_to_ar,
                    'speedup_decode': speedup_decode,
                }
                self.generation_stats.append(turn_record)
                print_generation_summary("Turn Summary", summarize_generation_stats([turn_record]))

        history_reply_text = ar_text if self.compare_AR_SSD and 'ar_text' in locals() and reply_text != ar_text else reply_text
        self.history.append({'role': 'assistant', 'content': history_reply_text, 'time': self.video_time})

    def inference(self, max_turns=None):
        turn_count = 0
        while self.query_queue:
            self._encode_query()
            turn_count += 1
            if max_turns is not None and turn_count >= max_turns:
                break
        return {
            'conversation': copy.deepcopy(self.history),
            'drop_ratio': copy.deepcopy(self.model.model.all_drop_ratios),
            'generation_stats': copy.deepcopy(self.generation_stats),
        }


class DoNothingDataCollator:
    def __call__(self, batch):
        return batch[0]


def round_numbers(data, n):
    if isinstance(data, list):
        return [round_numbers(d, n) for d in data]
    elif isinstance(data, dict):
        return {k: round_numbers(v, n) for k, v in data.items()}
    elif isinstance(data, float):
        return round(data, n)
    return data


def post_process_conversation_for_print(conversation):
    no_reply_text= "NO REPLY"
    new_conversation = list()
    for turn in conversation:
        if isinstance(turn['content'], list):
            res = ''
            for content in turn['content']:
                if 'text' in content:
                    res += content['text'].strip()
            turn['content'] = res
        if turn['role'] == 'assistant':
            if turn['content'] != no_reply_text:
                new_conversation.append(turn)
        elif turn['role'] == 'user':
            if turn['content']:
                new_conversation.append(turn)
    return new_conversation


SHORT_REPLY_MAX_TOKENS = 5


def _safe_div(num, den):
    return num / den if den else 0


def _avg(values):
    return sum(values) / len(values) if values else 0


def _metrics(turn_stats):
    """Accept-length and speedup metrics for a set of turns.

    accept_length_with_bonus    = sum(accept_lengths) / rounds
        Mean tokens advanced per round, INCLUDING the free token the big
        model emits every round. Same definition as Kangaroo's mean
        accepted tokens.
    accept_length_without_bonus = (sum - rounds) / rounds
        Pure drafted tokens accepted per round (mean of a-1), with the big
        model's free token removed.
    """
    spec = [s['speculative'] for s in turn_stats if 'speculative' in s]
    ar = [s['autoregressive'] for s in turn_stats if 'autoregressive' in s]

    accept_lengths = [a for s in spec for a in s.get('accept_lengths', [])]
    rounds = len(accept_lengths)
    total = sum(accept_lengths)

    spec_tokens = sum(s.get('total_tokens', 0) for s in spec)
    ar_tokens = sum(s.get('total_tokens', 0) for s in ar)
    spec_decode = sum(s.get('decode_time', 0) for s in spec)
    ar_decode = sum(s.get('decode_time', 0) for s in ar)
    spec_tps = _safe_div(spec_tokens, spec_decode)
    ar_tps = _safe_div(ar_tokens, ar_decode)

    # Adapter quality (for monitoring training): top1 = drafted token matches
    # the big model; first = first drafted token of each round.
    adapter_correct = sum(s.get('adapter_correct', 0) for s in spec)
    adapter_total = sum(s.get('adapter_total', 0) for s in spec)
    adapter_first_correct = sum(s.get('adapter_first_correct', 0) for s in spec)
    adapter_first_total = sum(s.get('adapter_first_total', 0) for s in spec)
    conf_weight = sum(s.get('adapter_total', 0) for s in spec)
    avg_confidence = _safe_div(
        sum(s.get('adapter_avg_confidence', 0) * s.get('adapter_total', 0) for s in spec),
        conf_weight,
    )

    # Per-round detail. spec_time is measured (draft+verify); the AR time is
    # estimated from the turn's average AR per-token cost (AR has no rounds),
    # so per-round speedup is approximate and noisy on very short turns.
    per_turn = []
    for rec in turn_stats:
        s = rec.get('speculative')
        if not s:
            continue
        ar_rec = rec.get('autoregressive', {})
        al = s.get('accept_lengths', [])
        dts = s.get('draft_times', [])
        vts = s.get('verify_times', [])
        ar_per_tok = _safe_div(ar_rec.get('decode_time', 0), ar_rec.get('total_tokens', 0))
        rounds_detail = []
        for i, a in enumerate(al):
            spec_time = (dts[i] if i < len(dts) else 0.0) + (vts[i] if i < len(vts) else 0.0)
            ar_time = a * ar_per_tok
            rounds_detail.append({
                'accept_length': a,
                'spec_time': spec_time,
                'ar_time_est': ar_time,
                'speedup': round(_safe_div(ar_time, spec_time), 4),
            })
        per_turn.append({
            'steps': len(al),
            'accept_lengths': al,
            'rounds': rounds_detail,
            # First-token probe (only meaningful when --adapter_first_token):
            'first_token_used_adapter': s.get('first_token_used_adapter', False),
            'first_token_disagree': s.get('first_token_disagree_with_base', False),
            'first_token_str': s.get('first_token_str', ''),
            'base_first_token_str': s.get('base_first_token_str', ''),
            # Gate probe (only meaningful when --use_gate):
            'gate_used': s.get('gate_used', False),
            'gate_prob': s.get('gate_prob', None),
            'gate_skipped_spec': s.get('gate_skipped_spec', False),
        })

    # First-token + gate aggregates
    first_token_disagree_n = sum(
        1 for s in spec if s.get('first_token_disagree_with_base', False)
    )
    used_adapter_first = any(s.get('first_token_used_adapter', False) for s in spec)
    gate_probs = [s.get('gate_prob') for s in spec if s.get('gate_prob') is not None]
    gate_skipped_n = sum(1 for s in spec if s.get('gate_skipped_spec', False))
    gate_was_used = any(s.get('gate_used', False) for s in spec)

    return {
        'turns': len(spec),
        'rounds': rounds,
        'accept_lengths': accept_lengths,
        'accept_lengths_sum': total,
        'accept_length_with_bonus': _safe_div(total, rounds),
        # Drafted tokens accepted per round = adapter_correct/rounds. NOT
        # (sum-rounds)/rounds: a round that ends on a correctly-drafted EOS
        # has no big-model bonus token, so subtracting a fixed 1 would
        # undercount the adapter there.
        'accept_length_without_bonus': _safe_div(adapter_correct, rounds),
        'spec_tokens': spec_tokens,
        'ar_tokens': ar_tokens,
        'spec_decode_time': spec_decode,
        'ar_decode_time': ar_decode,
        'spec_decode_tokens_per_second': round(spec_tps, 2),
        'ar_decode_tokens_per_second': round(ar_tps, 2),
        'speedup': round(_safe_div(spec_tps, ar_tps), 4) if ar_tps > 0 else None,
        'adapter_correct': adapter_correct,
        'adapter_total': adapter_total,
        'adapter_accuracy': _safe_div(adapter_correct, adapter_total),
        'adapter_first_correct': adapter_first_correct,
        'adapter_first_total': adapter_first_total,
        'adapter_first_accuracy': _safe_div(adapter_first_correct, adapter_first_total),
        'avg_confidence': avg_confidence,
        'per_turn': per_turn,
        'first_token_used_adapter': used_adapter_first,
        'first_token_disagree_count': first_token_disagree_n,
        'first_token_disagree_rate': _safe_div(first_token_disagree_n, len(spec)),
        'gate_used': gate_was_used,
        'gate_skipped_count': gate_skipped_n,
        'gate_skip_rate': _safe_div(gate_skipped_n, len(spec)),
        'gate_prob_mean': (sum(gate_probs) / len(gate_probs)) if gate_probs else None,
        'gate_prob_min': min(gate_probs) if gate_probs else None,
        'gate_prob_max': max(gate_probs) if gate_probs else None,
        'gate_probs': gate_probs,
    }


def summarize_generation_stats(turn_stats):
    turns = [s for s in turn_stats if 'speculative' in s]
    if not turns:
        return {}

    short = [s for s in turns
             if s['speculative'].get('total_tokens', 0) <= SHORT_REPLY_MAX_TOKENS]
    long_ = [s for s in turns
             if s['speculative'].get('total_tokens', 0) > SHORT_REPLY_MAX_TOKENS]

    summary = _metrics(turns)
    summary['short_reply'] = _metrics(short)
    summary['long_reply'] = _metrics(long_)
    return summary


def _print_metrics(label, m):
    print(
        f"[{label}] turns={m['turns']} rounds={m['rounds']} sum={m['accept_lengths_sum']} | "
        f"accept_len with_bonus={m['accept_length_with_bonus']:.3f} "
        f"no_bonus={m['accept_length_without_bonus']:.3f} | "
        f"spec {m['spec_decode_tokens_per_second']:.1f} tok/s | "
        f"AR {m['ar_decode_tokens_per_second']:.1f} tok/s | "
        f"speedup {m['speedup'] or 0:.2f}x"
    )
    print(
        f"    adapter top1={m['adapter_correct']}/{m['adapter_total']} "
        f"({m['adapter_accuracy']:.1%}) | "
        f"first={m['adapter_first_correct']}/{m['adapter_first_total']} "
        f"({m['adapter_first_accuracy']:.1%}) | conf={m['avg_confidence']:.3f}"
    )
    if m.get('first_token_used_adapter'):
        n = m['turns']
        d = m['first_token_disagree_count']
        print(
            f"    first-token probe (adapter): disagree {d}/{n} "
            f"({m['first_token_disagree_rate']:.1%}) vs base"
        )
    if m.get('gate_used'):
        n = m['turns']
        s = m['gate_skipped_count']
        p_mean = m.get('gate_prob_mean')
        p_min = m.get('gate_prob_min')
        p_max = m.get('gate_prob_max')
        print(
            f"    gate: skipped_spec {s}/{n} ({m['gate_skip_rate']:.1%}) | "
            f"prob mean={p_mean:.3f} min={p_min:.3f} max={p_max:.3f}"
        )


def print_generation_summary(title, summary):
    if not summary:
        return

    print(f"\n--- {title} ({summary['turns']} turns) ---")
    _print_metrics("Overall", summary)
    for turn_idx, t in enumerate(summary.get('per_turn', [])):
        round_speedup = [round(r['speedup'], 2) for r in t.get('rounds', [])]
        first_info = ""
        if t.get('first_token_used_adapter'):
            ft = t.get('first_token_str', '').replace('\n', '\\n')
            bt = t.get('base_first_token_str', '').replace('\n', '\\n')
            tag = "DISAGREE" if t.get('first_token_disagree') else "agree"
            first_info = f" | first: adapter='{ft}' base='{bt}' ({tag})"
        gate_info = ""
        if t.get('gate_used'):
            p = t.get('gate_prob')
            if p is not None:
                skip = "SKIP" if t.get('gate_skipped_spec') else "pass"
                gate_info = f" | gate: p={p:.3f} ({skip})"
        print(f"  turn {turn_idx}: steps={t['steps']} | accept={t['accept_lengths']} | round_speedup={round_speedup}{first_info}{gate_info}")
    for label, key in (('Short/NO_REPLY <=5 tok', 'short_reply'), ('Long >5 tok', 'long_reply')):
        bucket = summary.get(key)
        if bucket and bucket['turns']:
            _print_metrics(label, bucket)


def load_examples(path):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_example(example, default_system_prompt, example_idx=0):
    """Return a dict guaranteed to have 'question_id' and 'conversation' fields.

    Accepted input formats (the first key found wins):
      - {'conversation': [{role, content}, ...]}   already proper
      - {'messages':     [{role, content}, ...]}   SFT format
      - {'turns':        [str, str, ...]}           MT-Bench format
                                                    (each str becomes a user msg)
    A question_id is auto-generated when missing.
    A leading system message is added when missing.
    """
    if 'conversation' in example:
        conv = list(example['conversation'])
    elif 'messages' in example:
        conv = list(example['messages'])
    elif 'turns' in example:
        # MT-Bench style: list of user-turn strings.
        conv = [{'role': 'user', 'content': str(t)} for t in example['turns']]
    else:
        raise KeyError(
            f"example has no 'conversation' / 'messages' / 'turns' field. "
            f"Keys found: {sorted(example.keys())}"
        )

    if not conv or conv[0].get('role') != 'system':
        conv = [{'role': 'system', 'content': default_system_prompt}] + conv

    qid = (
        example.get('question_id')
        or (example.get('metadata') or {}).get('question_id')
        or example.get('id')
        or f'item_{example_idx}'
    )

    return {
        'question_id': str(qid),
        'conversation': conv,
    }


def main():
    all_stats = []
    args = parse_args()
    print(args)
    data_list = load_examples(args.test_fname)
    args.end_idx = len(data_list) if args.end_idx is None else args.end_idx

    existing_question_ids = set()
    if os.path.exists(args.output_fname):
        for line in open(args.output_fname):
            existing_question_ids.add(json.loads(line)['question_id'])
        print(f"found {len(existing_question_ids)} existing question ids in {args.output_fname}")

    f_out = open(args.output_fname, 'a')
    wrapper = ProactiveInferenceClient(args)

    frame_interval = 1.0
    print(f"setting {frame_interval=} for testing on {args.test_fname=}")
    wrapper.set_fps(frame_interval=frame_interval)

    # GPU warmup: run a few turns through the full spec+AR path so CUDA kernel
    # compilation / cuBLAS algo selection / allocator costs don't land inside
    # the timed regions of the first recorded turn. Stats are discarded.
    if args.warmup_turns and data_list:
        warmup_example = normalize_example(data_list[0], args.system_prompt, example_idx=0)
        print(f"[warmup] running {args.warmup_turns} turn(s), stats discarded")
        wrapper.reset()
        wrapper.input_query_stream(warmup_example['conversation'])
        wrapper.inference(max_turns=args.warmup_turns)
        wrapper.reset()  # clears generation_stats so warmup never enters the results
        print("[warmup] done")

    for example_i, example in enumerate(tqdm(data_list)):
        # Normalize input format: accepts 'conversation' / 'messages' / 'turns'
        # so the same script works on multimodal streaming data AND on
        # text-only benchmarks like MT-Bench or ShareGPT.
        example = normalize_example(example, args.system_prompt, example_idx=example_i)
        if example['question_id'] in existing_question_ids:
            print(f"question {example['question_id']} already exists in {args.output_fname}, skip")
            continue
        if example_i < args.start_idx: continue
        if example_i >= args.end_idx: break
        wrapper.reset()
        wrapper.input_query_stream(example['conversation'])
        conversation_start_time = time.perf_counter()
        model_outputs = wrapper.inference(max_turns=args.max_turns)
        conversation_elapsed_time = time.perf_counter() - conversation_start_time
        
        turn_stats = model_outputs['generation_stats']
        id_summary = summarize_generation_stats(turn_stats)

        res = {
            'question_id': example['question_id'],
            'model_response_list': post_process_conversation_for_print(model_outputs['conversation']),
            'drop_ratio_list': model_outputs['drop_ratio'],
            'summary': id_summary,
        }


        f_out.write(json.dumps(res) + '\n')
        f_out.flush()

        if turn_stats:
            print_generation_summary(f"Question {example['question_id']} Summary", id_summary)
            all_stats.extend(turn_stats)

    f_out.close()

    if all_stats:
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print_generation_summary("All Turns", summarize_generation_stats(all_stats))


if __name__ == '__main__':
    main()
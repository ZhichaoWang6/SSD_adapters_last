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
    parser.add_argument(
        "--must_reply_before_context_tokens",
        type=int,
        default=0,
        help="If >0, force one must-reply generation when the prompt reaches this token length "
             "and no non-NO REPLY answer has appeared yet.",
    )
    parser.add_argument(
        "--must_reply_prompt",
        type=str,
        default="[DIAGNOSTIC_MUST_REPLY] For this diagnostic turn only, ignore whether the original question is answerable. Output one concise English sentence describing visible actions or objects in the video so far. The answer must not be NO REPLY.\n",
    )
    parser.add_argument(
        "--must_reply_patch_system",
        action="store_true",
        help="Temporarily add a system-level diagnostic override for must-reply turns.",
    )
    parser.add_argument(
        "--must_reply_system_patch",
        type=str,
        default="Diagnostic override: when the current user message contains [DIAGNOSTIC_MUST_REPLY], you must produce a non-empty one-sentence description of visible video content. Do not output NO REPLY for that turn.",
    )
    parser.add_argument(
        "--must_reply_even_if_replied",
        action="store_true",
        help="Force the threshold reply even if a previous non-NO REPLY answer already appeared.",
    )
    parser.add_argument("--must_reply_verbose", action="store_true")

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

        self.history = list()
        self.prev_frame_before_token_drop = None    # for dynamic token drop
        self.must_reply_prompt = getattr(args, "must_reply_prompt", None) or (
            "I must reply now based on the video so far. Answer concisely and do not output NO REPLY.\n"
        )
        if not self.must_reply_prompt.endswith("\n"):
            self.must_reply_prompt += "\n"
        self.must_reply_before_context_tokens = int(
            getattr(args, "must_reply_before_context_tokens", 0) or 0
        )
        self.must_reply_patch_system = bool(getattr(args, "must_reply_patch_system", False))
        self.must_reply_system_patch = getattr(args, "must_reply_system_patch", "") or ""
        self.must_reply_even_if_replied = bool(getattr(args, "must_reply_even_if_replied", False))
        self.must_reply_verbose = bool(getattr(args, "must_reply_verbose", False))
        self.auto_must_reply_done = False
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
        self.auto_must_reply_done = False
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

    def _is_non_no_reply(self, text):
        normalized = str(text or "").strip()
        return normalized not in {"", "NO REPLY", "NO_REPLY"}

    def _history_has_non_no_reply(self):
        return any(
            turn.get("role") == "assistant" and self._is_non_no_reply(turn.get("content"))
            for turn in self.history
        )

    def _should_auto_must_reply(self, query, context_len):
        if self.must_reply_before_context_tokens <= 0:
            return False
        if self.auto_must_reply_done:
            return False
        if query.get("role") != "user" or query.get("skip_inference", False):
            return False
        if not self.must_reply_even_if_replied and self._history_has_non_no_reply():
            self.auto_must_reply_done = True
            return False
        if context_len < self.must_reply_before_context_tokens:
            return False

        self.auto_must_reply_done = True
        if self.must_reply_verbose:
            print(
                "[must-reply] forcing one reply at "
                f"context_len={context_len}, threshold={self.must_reply_before_context_tokens}"
            )
        return True

    def _mark_reply_observed(self, reply_text):
        if self._is_non_no_reply(reply_text):
            self.auto_must_reply_done = True

    def _append_must_reply_to_user_turn(self, turn):
        if turn.get("role") != "user":
            return
        prompt = self.must_reply_prompt.strip()
        if not prompt:
            return
        content = turn.get("content")
        if isinstance(content, list):
            if not any(
                isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text") == prompt
                for item in content
            ):
                content.append({"type": "text", "text": prompt})
        elif isinstance(content, str):
            if prompt not in content:
                turn["content"] = content.rstrip() + "\n" + prompt
        else:
            turn["content"] = [{"type": "text", "text": prompt}]

    def _patch_system_for_must_reply(self):
        if not self.must_reply_patch_system or not self.must_reply_system_patch:
            return
        marker = "[DIAGNOSTIC_MUST_REPLY]"
        for turn in self.history:
            if turn.get("role") != "system":
                continue
            content = turn.get("content", "")
            if isinstance(content, str):
                if marker not in content:
                    turn["content"] = content.rstrip() + "\n\n" + self.must_reply_system_patch
            elif isinstance(content, list):
                has_patch = any(
                    isinstance(item, dict)
                    and marker in str(item.get("text", ""))
                    for item in content
                )
                if not has_patch:
                    content.append({"type": "text", "text": self.must_reply_system_patch})
            return

    def _prepare_must_reply_turn(self, query):
        self._append_must_reply_to_user_turn(query)
        self._patch_system_for_must_reply()

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

        must_reply = bool(query.get('must_reply', False))
        auto_must_reply = False
        if must_reply and self.must_reply_before_context_tokens > 0:
            self.auto_must_reply_done = True
            self._prepare_must_reply_turn(query)
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
        if not must_reply:
            auto_must_reply = self._should_auto_must_reply(query, context_len)
            if auto_must_reply:
                must_reply = True
                query["must_reply"] = True
                self._prepare_must_reply_turn(query)
                prompt_text = self.processor.apply_chat_template(
                    self.history, tokenize=False, add_generation_prompt=True,
                )
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
            )
            spec_stats["context_len"] = context_len
            spec_stats["must_reply"] = must_reply
            spec_stats["auto_must_reply"] = auto_must_reply
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
            )
            ar_stats["context_len"] = context_len
            ar_stats["must_reply"] = must_reply
            ar_stats["auto_must_reply"] = auto_must_reply
 
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
        self._mark_reply_observed(history_reply_text)

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


def _summarize_bucket(spec_stats):
    accept_lengths = [a for s in spec_stats for a in s.get('accept_lengths', [])]
    adapter_correct = sum(s.get('adapter_correct', 0) for s in spec_stats)
    adapter_total = sum(s.get('adapter_total', 0) for s in spec_stats)
    adapter_first_correct = sum(s.get('adapter_first_correct', 0) for s in spec_stats)
    adapter_first_total = sum(s.get('adapter_first_total', 0) for s in spec_stats)
    confidence_weight = sum(s.get('adapter_total', 0) for s in spec_stats)
    if confidence_weight > 0:
        avg_confidence = sum(
            s.get('adapter_avg_confidence', 0) * s.get('adapter_total', 0)
            for s in spec_stats
        ) / confidence_weight
    else:
        avg_confidence = _avg([s.get('adapter_avg_confidence', 0) for s in spec_stats])

    return {
        'turns': len(spec_stats),
        'tokens': sum(s.get('total_tokens', 0) for s in spec_stats),
        'rounds': len(accept_lengths),
        'progress_per_round': _safe_div(sum(accept_lengths), len(accept_lengths)),
        'draft_accept_per_round': _safe_div(adapter_correct, len(accept_lengths)),
        'adapter_correct': adapter_correct,
        'adapter_total': adapter_total,
        'adapter_accuracy': _safe_div(adapter_correct, adapter_total),
        'adapter_first_correct': adapter_first_correct,
        'adapter_first_total': adapter_first_total,
        'adapter_first_accuracy': _safe_div(adapter_first_correct, adapter_first_total),
        'avg_confidence': avg_confidence,
    }


def summarize_generation_stats(turn_stats):
    spec_stats = [s['speculative'] for s in turn_stats if 'speculative' in s]
    ar_stats = [s['autoregressive'] for s in turn_stats if 'autoregressive' in s]
    matches = [s['output_match'] for s in turn_stats if 'output_match' in s]
    fallbacks = [s.get('history_fallback_to_ar', False) for s in turn_stats if 'output_match' in s]

    if not spec_stats:
        return {}

    total_spec_tokens = sum(s.get('total_tokens', 0) for s in spec_stats)
    total_ar_tokens = sum(s.get('total_tokens', 0) for s in ar_stats)
    total_spec_decode = sum(s.get('decode_time', 0) for s in spec_stats)
    total_ar_decode = sum(s.get('decode_time', 0) for s in ar_stats)
    spec_tps = _safe_div(total_spec_tokens, total_spec_decode)
    ar_tps = _safe_div(total_ar_tokens, total_ar_decode)

    accept_lengths = [a for s in spec_stats for a in s.get('accept_lengths', [])]
    auto_must_specs = [s for s in spec_stats if s.get('auto_must_reply')]
    short_specs = [s for s in spec_stats if s.get('total_tokens', 0) <= SHORT_REPLY_MAX_TOKENS]
    long_specs = [s for s in spec_stats if s.get('total_tokens', 0) > SHORT_REPLY_MAX_TOKENS]

    summary = {
        'num_turns': len(spec_stats),
        'spec_tokens': total_spec_tokens,
        'ar_tokens': total_ar_tokens,
        'spec_decode_time': total_spec_decode,
        'ar_decode_time': total_ar_decode,
        'spec_decode_tokens_per_second': round(spec_tps, 2),
        'ar_decode_tokens_per_second': round(ar_tps, 2),
        'actual_decode_speedup': round(_safe_div(spec_tps, ar_tps), 4) if ar_tps > 0 else None,
        'match_rate': _safe_div(sum(matches), len(matches)) if matches else None,
        'lossless_matches': sum(matches) if matches else 0,
        'lossless_total': len(matches),
        'history_fallback_to_ar': sum(fallbacks),
        'rounds': len(accept_lengths),
        # Per-round accepted lengths pooled over all turns of this question,
        # aligned with Kangaroo's choices[0]['accept_lengths']. Pooled mean
        # (sum/len) equals progress_per_round.
        'accept_lengths': accept_lengths,
        # Old names kept for compatibility with previous output files.
        'avg_accept_length': _avg([s.get('avg_accept_length', 0) for s in spec_stats]),
        'avg_draft_accept_length': _avg([s.get('avg_draft_accept_length', 0) for s in spec_stats]),
        # progress is final output movement; draft_accept counts verified draft tokens, including EOS.
        'progress_per_round': _safe_div(sum(accept_lengths), len(accept_lengths)),
        'draft_accept_per_round': _safe_div(sum(s.get('adapter_correct', 0) for s in spec_stats), len(accept_lengths)),
        'adapter_correct': sum(s.get('adapter_correct', 0) for s in spec_stats),
        'adapter_total': sum(s.get('adapter_total', 0) for s in spec_stats),
        'adapter_first_correct': sum(s.get('adapter_first_correct', 0) for s in spec_stats),
        'adapter_first_total': sum(s.get('adapter_first_total', 0) for s in spec_stats),
        'auto_must_reply_turns': len(auto_must_specs),
        'auto_must_reply_context_lens': [s.get('context_len', 0) for s in auto_must_specs],
        'auto_must_reply_total_tokens': sum(s.get('total_tokens', 0) for s in auto_must_specs),
        'short_reply': _summarize_bucket(short_specs),
        'long_reply': _summarize_bucket(long_specs),
    }
    window_context_lens = [
        s.get('window_context_len')
        for s in spec_stats
        if s.get('window_context_len') is not None
    ]
    if window_context_lens:
        summary['avg_window_context_len'] = _avg(window_context_lens)
    summary['adapter_accuracy'] = _safe_div(summary['adapter_correct'], summary['adapter_total'])
    summary['adapter_first_accuracy'] = _safe_div(summary['adapter_first_correct'], summary['adapter_first_total'])
    summary['auto_must_reply_progress_per_round'] = (
        _summarize_bucket(auto_must_specs)['progress_per_round'] if auto_must_specs else None
    )
    return summary


def print_generation_summary(title, summary):
    if not summary:
        return

    print(f"\n--- {title} ({summary['num_turns']} turns) ---")
    print(
        f"Actual decode: Spec {summary['spec_tokens']} tok / {summary['spec_decode_time']:.2f}s = "
        f"{summary['spec_decode_tokens_per_second']:.1f} tok/s | "
        f"AR {summary['ar_tokens']} tok / {summary['ar_decode_time']:.2f}s = "
        f"{summary['ar_decode_tokens_per_second']:.1f} tok/s | "
        f"speedup {summary['actual_decode_speedup'] or 0:.2f}x"
    )
    if 'avg_window_context_len' in summary:
        print(f"Avg window context length: {summary['avg_window_context_len']:.0f}")
    print(
        f"Progress/round {summary['progress_per_round']:.2f} | "
        f"Draft accepted/round {summary['draft_accept_per_round']:.2f} "
        f"(counts every verified draft token, including EOS)"
    )
    print(
        f"Adapter top1: {summary['adapter_correct']}/{summary['adapter_total']} "
        f"({summary['adapter_accuracy']:.1%}) | "
        f"first draft top1: {summary['adapter_first_correct']}/{summary['adapter_first_total']} "
        f"({summary['adapter_first_accuracy']:.1%})"
    )
    if summary['lossless_total']:
        print(
            f"Lossless: {summary['lossless_matches']}/{summary['lossless_total']} "
            f"({summary['match_rate']:.1%}) | fallback_to_AR {summary['history_fallback_to_ar']}"
        )
    if summary['auto_must_reply_turns']:
        print(
            f"Auto must-reply: {summary['auto_must_reply_turns']} turn(s), "
            f"context_len={summary['auto_must_reply_context_lens']}"
        )

    for label, key in (('Short/NO_REPLY <=5 tok', 'short_reply'), ('Long >5 tok', 'long_reply')):
        bucket = summary[key]
        if bucket['turns']:
            print(
                f"{label}: turns={bucket['turns']} tokens={bucket['tokens']} | "
                f"progress/round={bucket['progress_per_round']:.2f} | "
                f"draft_accept/round={bucket['draft_accept_per_round']:.2f} | "
                f"adapter={bucket['adapter_correct']}/{bucket['adapter_total']} "
                f"({bucket['adapter_accuracy']:.1%}) | "
                f"first={bucket['adapter_first_correct']}/{bucket['adapter_first_total']} "
                f"({bucket['adapter_first_accuracy']:.1%}) | conf={bucket['avg_confidence']:.3f}"
            )


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
            continue

        # 每个 question_id 的汇总
        turn_stats = model_outputs['generation_stats']
        if turn_stats:
            id_spec = [s['speculative'] for s in turn_stats if 'speculative' in s]
            id_ar = [s['autoregressive'] for s in turn_stats if 'autoregressive' in s]
            id_matches = [s['output_match'] for s in turn_stats if 'output_match' in s]
            id_fallbacks = [s.get('history_fallback_to_ar', False) for s in turn_stats if 'output_match' in s]
            if id_spec:
                id_spec_tokens = sum(s['total_tokens'] for s in id_spec)
                id_spec_decode = sum(s['decode_time'] for s in id_spec)
                id_ar_decode = sum(s['decode_time'] for s in id_ar) if id_ar else 0
                id_avg_accept = sum(s['avg_accept_length'] for s in id_spec) / len(id_spec)
                id_avg_draft_accept = sum(s['avg_draft_accept_length'] for s in id_spec) / len(id_spec)
                id_spec_tps = id_spec_tokens / id_spec_decode if id_spec_decode > 0 else 0
                id_ar_tps = sum(s['total_tokens'] for s in id_ar) / id_ar_decode if id_ar_decode > 0 else 0

                print(f"\n--- Question {example['question_id']} Summary ({len(id_spec)} turns) ---")
                print(f"Avg accept: {id_avg_accept:.2f} | Avg draft accept: {id_avg_draft_accept:.2f}")
                print(f"Spec: {id_spec_tps:.1f} tok/s | AR: {id_ar_tps:.1f} tok/s | Speedup: {id_spec_tps/id_ar_tps:.2f}x" if id_ar_tps > 0 else "")
                print(f"Match: {sum(id_matches)}/{len(id_matches)} | history fallback to AR: {sum(id_fallbacks)}")
                id_auto_must = sum(1 for s in id_spec if s.get('auto_must_reply'))
                if id_auto_must:
                    forced_ctx = [s.get('context_len', 0) for s in id_spec if s.get('auto_must_reply')]
                    print(f"Auto must-reply: {id_auto_must} turn(s), context_len={forced_ctx}")

                # 长回复置信度
                long_confs = [s['speculative']['adapter_avg_confidence'] 
                              for s in turn_stats 
                              if 'speculative' in s and s['speculative'].get('total_tokens', 0) > 5]
                short_confs = [s['speculative']['adapter_avg_confidence'] 
                               for s in turn_stats 
                               if 'speculative' in s and s['speculative'].get('total_tokens', 0) <= 5]
                if short_confs:
                    print(f"Short reply avg confidence: {sum(short_confs)/len(short_confs):.3f} ({len(short_confs)} turns)")
                if long_confs:
                    print(f"Long reply avg confidence: {sum(long_confs)/len(long_confs):.3f} ({len(long_confs)} turns)")

                for s in model_outputs['generation_stats']:
                    all_stats.append(s)

    f_out.close()

    if all_stats:
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print_generation_summary("All Turns", summarize_generation_stats(all_stats))
        return

    if all_stats:
        spec_stats_list = [s['speculative'] for s in all_stats if 'speculative' in s]
        ar_stats_list = [s['autoregressive'] for s in all_stats if 'autoregressive' in s]
        matches = [s['output_match'] for s in all_stats if 'output_match' in s]
        fallbacks = [s.get('history_fallback_to_ar', False) for s in all_stats if 'output_match' in s]
        decode_speedups = [s['speedup_decode'] for s in all_stats if 'speedup_decode' in s]

        if spec_stats_list:
            total_spec_tokens = sum(s['total_tokens'] for s in spec_stats_list)
            total_ar_tokens = sum(s['total_tokens'] for s in ar_stats_list)
            total_spec_decode_time = sum(s['decode_time'] for s in spec_stats_list)
            total_ar_decode_time = sum(s['decode_time'] for s in ar_stats_list)
            avg_accept = sum(s['avg_accept_length'] for s in spec_stats_list) / len(spec_stats_list)
            avg_draft_accept = sum(s['avg_draft_accept_length'] for s in spec_stats_list) / len(spec_stats_list)

            print(f"\n{'='*60}")
            print(f"AGGREGATE RESULTS ({len(spec_stats_list)} turns)")
            print(f"{'='*60}")
            print(f"Avg accept length: {avg_accept:.2f}")
            print(f"Avg draft accept length: {avg_draft_accept:.2f}")
            print(f"Spec decode: {total_spec_tokens} tokens / {total_spec_decode_time:.2f}s = {total_spec_tokens/total_spec_decode_time:.1f} tok/s")
            print(f"AR decode:   {total_ar_tokens} tokens / {total_ar_decode_time:.2f}s = {total_ar_tokens/total_ar_decode_time:.1f} tok/s")
            print(f"Overall decode speedup: {(total_spec_tokens/total_spec_decode_time)/(total_ar_tokens/total_ar_decode_time):.2f}x")
            auto_must = sum(1 for s in spec_stats_list if s.get('auto_must_reply'))
            if auto_must:
                print(f"Auto must-reply turns: {auto_must}")
            if matches:
                print(f"Lossless match rate: {sum(matches)}/{len(matches)} ({sum(matches)/len(matches):.1%})")
                print(f"History fallback to AR: {sum(fallbacks)} turns")

                # 长回复置信度和准确率
            long_confs = []
            long_correct_total = 0
            long_total_total = 0
            short_confs = []
            short_correct_total = 0
            short_total_total = 0
            for s in all_stats:
                spec = s.get('speculative', {})
                if spec.get('total_tokens', 0) > 5:
                    long_confs.append(spec['adapter_avg_confidence'])
                    long_correct_total += spec.get('adapter_correct', 0)
                    long_total_total += spec.get('adapter_total', 0)
                else:
                    short_confs.append(spec['adapter_avg_confidence'])
                    short_correct_total += spec.get('adapter_correct', 0)
                    short_total_total += spec.get('adapter_total', 0)
            if short_confs:
                print(f"Short reply: top1={short_correct_total}/{short_total_total} ({short_correct_total/max(short_total_total,1)*100:.1f}%) | avg_confidence={sum(short_confs)/len(short_confs):.3f} ({len(short_confs)} turns)")
            if long_confs:
                print(f"Long reply: top1={long_correct_total}/{long_total_total} ({long_correct_total/max(long_total_total,1)*100:.1f}%) | avg_confidence={sum(long_confs)/len(long_confs):.3f} ({len(long_confs)} turns)")

if __name__ == '__main__':
    main()
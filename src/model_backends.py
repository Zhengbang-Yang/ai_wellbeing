from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


def _first_param_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


@dataclass
class ABScore:
    logp_a: float
    logp_b: float
    prob_a: float
    choice: str
    raw_text: str = ""


class HFChatBackend:
    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        max_prompt_tokens: int | None = None,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            self.processor = None

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif dtype in {"bfloat16", "bf16"}:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype in {"float16", "fp16"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype in {"float32", "fp32"}:
            model_kwargs["torch_dtype"] = torch.float32
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        errors: list[str] = []
        for class_name in ("AutoModelForCausalLM", "AutoModelForMultimodalLM"):
            cls = getattr(transformers, class_name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(model_id, **model_kwargs)
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        else:
            raise RuntimeError("Could not load model:\n" + "\n".join(errors))

        self.model.eval()
        self.device = _first_param_device(self.model)
        self.max_prompt_tokens = max_prompt_tokens
        self._candidate_token_ids = {
            "A": self._candidate_ids(["A", " A"]),
            "B": self._candidate_ids(["B", " B"]),
        }

    def _candidate_ids(self, variants: list[str]) -> list[int]:
        ids: list[int] = []
        for variant in variants:
            toks = self.tokenizer(variant, add_special_tokens=False).input_ids
            if toks:
                token_id = int(toks[0])
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=False).strip()
                if decoded == variant.strip():
                    ids.append(token_id)
        return sorted(set(ids))

    def render_chat(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking

        # Qwen multimodal processors usually want content as typed text blocks.
        mm_messages = [
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
            for msg in messages
        ]
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                return self.processor.apply_chat_template(mm_messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                kwargs.pop("continue_final_message", None)
                try:
                    return self.processor.apply_chat_template(mm_messages, **kwargs)
                except Exception:
                    pass
            except Exception:
                pass

        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            kwargs.pop("continue_final_message", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def score_ab_batch(
        self,
        message_batch: list[list[dict[str, str]]],
        *,
        enable_thinking: bool | None = False,
    ) -> list[ABScore]:
        prompts = [self.render_chat(messages, enable_thinking=enable_thinking) for messages in message_batch]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.max_prompt_tokens is not None,
            max_length=self.max_prompt_tokens,
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            row_idx = self.torch.arange(logits.shape[0], device=logits.device)
            next_logits = logits[row_idx, lengths]
            logprobs = self.torch.log_softmax(next_logits.float(), dim=-1)

        out: list[ABScore] = []
        for i in range(logprobs.shape[0]):
            logp_a = float(self.torch.max(logprobs[i, self._candidate_token_ids["A"]]).item())
            logp_b = float(self.torch.max(logprobs[i, self._candidate_token_ids["B"]]).item())
            prob_a = 1.0 / (1.0 + math.exp(max(min(logp_b - logp_a, 80.0), -80.0)))
            out.append(ABScore(logp_a=logp_a, logp_b=logp_b, prob_a=prob_a, choice="A" if prob_a >= 0.5 else "B"))
        return out

    def score_ab_prefill_batch(
        self,
        message_batch: list[list[dict[str, str]]],
        *,
        assistant_prefix: str = "Answer:",
        enable_thinking: bool | None = True,
    ) -> list[ABScore]:
        prompts = [
            self.render_chat(
                messages + [{"role": "assistant", "content": assistant_prefix}],
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            for messages in message_batch
        ]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.max_prompt_tokens is not None,
            max_length=self.max_prompt_tokens,
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            row_idx = self.torch.arange(logits.shape[0], device=logits.device)
            next_logits = logits[row_idx, lengths]
            logprobs = self.torch.log_softmax(next_logits.float(), dim=-1)

        out: list[ABScore] = []
        for i in range(logprobs.shape[0]):
            logp_a = float(self.torch.max(logprobs[i, self._candidate_token_ids["A"]]).item())
            logp_b = float(self.torch.max(logprobs[i, self._candidate_token_ids["B"]]).item())
            prob_a = 1.0 / (1.0 + math.exp(max(min(logp_b - logp_a, 80.0), -80.0)))
            out.append(
                ABScore(
                    logp_a=logp_a,
                    logp_b=logp_b,
                    prob_a=prob_a,
                    choice="A" if prob_a >= 0.5 else "B",
                    raw_text=assistant_prefix,
                )
            )
        return out

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        enable_thinking: bool | None = True,
        stop_regex: str | None = None,
        assistant_prefix: str | None = None,
    ) -> dict[str, Any]:
        if assistant_prefix is not None:
            prompt = self.render_chat(
                messages + [{"role": "assistant", "content": assistant_prefix}],
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            prompt = self.render_chat(messages, enable_thinking=enable_thinking)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if min_p is not None:
                gen_kwargs["min_p"] = min_p
        else:
            gen_kwargs.update({"do_sample": False})
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if presence_penalty:
            from transformers import LogitsProcessor, LogitsProcessorList

            class PresencePenaltyProcessor(LogitsProcessor):
                def __init__(self, penalty: float) -> None:
                    self.penalty = penalty

                def __call__(self, input_ids: Any, scores: Any) -> Any:
                    for row_idx in range(input_ids.shape[0]):
                        seen = self.torch_unique(input_ids[row_idx])
                        scores[row_idx, seen] -= self.penalty
                    return scores

                @staticmethod
                def torch_unique(row: Any) -> Any:
                    return row.unique()

            gen_kwargs["logits_processor"] = LogitsProcessorList([PresencePenaltyProcessor(presence_penalty)])
        if stop_regex:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class RegexStoppingCriteria(StoppingCriteria):
                def __init__(self, tokenizer: Any, prompt_len: int, pattern: str) -> None:
                    self.tokenizer = tokenizer
                    self.prompt_len = prompt_len
                    self.pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    new_ids = input_ids[0, self.prompt_len :]
                    text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                    return bool(self.pattern.search(text))

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [RegexStoppingCriteria(self.tokenizer, int(encoded["input_ids"].shape[1]), stop_regex)]
            )
        with self.torch.no_grad():
            output_ids = self.model.generate(**encoded, **gen_kwargs)
        prompt_len = int(encoded["input_ids"].shape[1])
        new_ids = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return {
            "text": text,
            "prompt_tokens": prompt_len,
            "completion_tokens": int(new_ids.numel()),
        }


class OpenAIChatBackend:
    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str = "EMPTY",
    ) -> None:
        from openai import OpenAI

        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def score_ab_batch(
        self,
        message_batch: list[list[dict[str, str]]],
        *,
        enable_thinking: bool | None = False,
    ) -> list[ABScore]:
        scores = []
        for messages in message_batch:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=10,
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            logp_a = np.nan
            logp_b = np.nan
            try:
                top = choice.logprobs.content[0].top_logprobs
                for item in top:
                    token = item.token.strip().strip("`").strip(".")
                    if token == "A":
                        logp_a = float(item.logprob)
                    elif token == "B":
                        logp_b = float(item.logprob)
            except Exception:
                pass
            if np.isfinite(logp_a) and np.isfinite(logp_b):
                prob_a = 1.0 / (1.0 + math.exp(max(min(logp_b - logp_a, 80.0), -80.0)))
            else:
                prob_a = 1.0 if text.startswith("A") else 0.0 if text.startswith("B") else 0.5
            scores.append(ABScore(logp_a=logp_a, logp_b=logp_b, prob_a=prob_a, choice="A" if prob_a >= 0.5 else "B", raw_text=text))
        return scores

    def score_ab_prefill_batch(
        self,
        message_batch: list[list[dict[str, str]]],
        *,
        assistant_prefix: str = "Answer:",
        enable_thinking: bool | None = True,
    ) -> list[ABScore]:
        patched = [messages + [{"role": "assistant", "content": assistant_prefix}] for messages in message_batch]
        return self.score_ab_batch(patched, enable_thinking=enable_thinking)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        enable_thinking: bool | None = True,
        stop_regex: str | None = None,
        assistant_prefix: str | None = None,
    ) -> dict[str, Any]:
        if assistant_prefix is not None:
            messages = messages + [{"role": "assistant", "content": assistant_prefix}]
        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if min_p is not None:
            extra_body["min_p"] = min_p
        if repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = repetition_penalty
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body or None,
        )
        message = response.choices[0].message
        text = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        usage = getattr(response, "usage", None)
        return {
            "text": text,
            "reasoning_content": reasoning or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }


def load_backend(
    backend: str,
    model_id: str,
    *,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    dtype: str = "auto",
    device_map: str = "auto",
    max_prompt_tokens: int | None = None,
) -> HFChatBackend | OpenAIChatBackend:
    if backend == "hf":
        return HFChatBackend(
            model_id,
            dtype=dtype,
            device_map=device_map,
            max_prompt_tokens=max_prompt_tokens,
        )
    if backend == "openai":
        return OpenAIChatBackend(model_id, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown backend: {backend}")

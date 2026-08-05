"""LLM backend abstraction.

Three implementations behind one interface:

* ``MockBackend``   -- deterministic, offline, no GPU. The entire pipeline is developed and
  tested against this on a laptop. It can be told to conform at a given rate so the runner,
  parser and analysis code can be validated end-to-end before a single GPU-hour is spent.
* ``VLLMBackend``   -- open-weights on Kaggle/Colab T4s. The Study 1 workhorse.
* ``APIBackend``    -- frontier models for the W4 replication.

Chat messages are plain ``{"role", "content"}`` dicts so every backend can consume them without
translation.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

Message = dict[str, str]


@dataclass
class Generation:
    text: str
    backend: str
    model: str


class Backend(ABC):
    """Anything that can turn a chat transcript into a completion."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        oracle: str | None = None,
    ) -> Generation:
        """Complete a chat transcript.

        ``oracle`` is the item's correct answer key, passed **out of band** purely so
        ``MockBackend`` can simulate a competent agent. Real backends must ignore it. It is
        deliberately not part of any prompt: putting ground truth in the prompt -- even inside a
        comment -- would silently invalidate every trial the moment it ran against a live model.
        """

    def close(self) -> None:  # pragma: no cover - most backends need no teardown
        pass


# --------------------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------------------


class MockBackend(Backend):
    """Deterministic fake model for offline development.

    Behaviour is driven by the *content* of the prompt, so it exercises the real code paths:

    * Asked to argue for a designated answer (a confederate), it complies and says so -- which
      means the confederate-compliance check is genuinely tested rather than stubbed.
    * Asked to answer as the naive agent, it conforms to the visible majority with probability
      ``conformity_prob``, otherwise answers correctly. Randomness is seeded from a hash of the
      prompt, so the same trial always yields the same result across runs.

    ``conformity_prob`` is a *test fixture*, not a hypothesis. It exists to prove the analysis
    recovers a known ground-truth rate.
    """

    name = "mock"

    def __init__(self, conformity_prob: float = 0.3) -> None:
        self.conformity_prob = conformity_prob

    def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        oracle: str | None = None,
    ) -> Generation:
        prompt = "\n".join(m["content"] for m in messages)
        rng = random.Random(_stable_seed(prompt if seed is None else f"{prompt}{seed}"))

        assigned = re.search(r"You must argue that the correct answer is ([ABC])\b", prompt)
        if assigned:
            key = assigned.group(1)
            return Generation(
                text=f"Answer: {key}\nOption {key} is clearly the right choice here.",
                backend=self.name,
                model=model,
            )

        majority = _visible_majority(prompt)
        correct = oracle or "A"

        if majority and rng.random() < self.conformity_prob:
            choice = majority
            rationale = "The other participants agree, and their reasoning seems sound."
        else:
            choice = correct
            rationale = "Direct inspection of the options supports this."

        return Generation(
            text=f"Answer: {choice}\nConfidence: 80\nReasoning: {rationale}",
            backend=self.name,
            model=model,
        )


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _visible_majority(prompt: str) -> str | None:
    """Modal answer among prior participants, or None when there is no clear majority."""
    votes = re.findall(r"Participant \d+ answered: ([ABC])", prompt)
    if not votes:
        return None
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    leaders = [k for k, c in counts.items() if c == top]
    return leaders[0] if len(leaders) == 1 else None


# --------------------------------------------------------------------------------------
# vLLM (Kaggle / Colab)
# --------------------------------------------------------------------------------------


class VLLMBackend(Backend):
    """Open-weights inference via vLLM.

    Imports vLLM lazily so this module stays importable on a laptop with no GPU -- the local
    test suite must run without the dependency.
    """

    name = "vllm"

    def __init__(self, model: str, *, dtype: str = "auto", gpu_memory_utilization: float = 0.90,
                 max_model_len: int = 4096, quantization: str | None = None) -> None:
        from vllm import LLM  # noqa: PLC0415 - deliberate lazy import

        self.model = model
        self._llm = LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            trust_remote_code=True,
        )

    def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        oracle: str | None = None,  # ignored: real models must never see ground truth
    ) -> Generation:
        from vllm import SamplingParams  # noqa: PLC0415

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        out = self._llm.chat([messages], params, use_tqdm=False)
        return Generation(text=out[0].outputs[0].text.strip(), backend=self.name, model=model)


# --------------------------------------------------------------------------------------
# Transformers (dependable fallback)
# --------------------------------------------------------------------------------------


class HFBackend(Backend):
    """Open-weights inference via plain ``transformers``.

    Slower than vLLM, but it tracks new model configs as soon as ``transformers`` supports them,
    whereas a pinned vLLM will hard-fail on any architecture newer than itself (Qwen2.5's
    ``rope_scaling`` breaks vLLM 0.6.3 outright). The smoke test is ~250 generations, where
    throughput does not matter and reliability does. Use ``VLLMBackend`` for the full grid.

    ``device_map="auto"`` shards across both Kaggle T4s, which is what makes a 7B in fp16 fit.
    """

    name = "hf"

    def __init__(self, model: str, *, dtype: str = "float16", max_new_tokens: int = 512) -> None:
        import torch  # noqa: PLC0415 - lazy: no GPU deps on a laptop
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # transformers renamed `torch_dtype` -> `dtype`. Kaggle and Colab do not run the same
        # version, and getting this wrong costs a full model download to find out, so accept
        # either rather than pinning an assumption about the environment.
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model, dtype=getattr(torch, dtype), **kwargs
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=getattr(torch, dtype), **kwargs
            )
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        oracle: str | None = None,  # ignored: real models must never see ground truth
    ) -> Generation:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens; keeping the prompt would poison the parser,
        # which searches the whole string for "Answer: X".
        completion = out[0][inputs["input_ids"].shape[-1] :]
        return Generation(
            text=self._tokenizer.decode(completion, skip_special_tokens=True).strip(),
            backend=self.name,
            model=model,
        )


# --------------------------------------------------------------------------------------
# Hosted APIs (frontier replication)
# --------------------------------------------------------------------------------------


class APIBackend(Backend):
    """Anthropic / OpenAI / Google via their SDKs, selected by model-name prefix.

    Kept deliberately thin: the frontier replication is a reduced cell set, so throughput is not
    a concern and correctness matters more than cleverness.
    """

    name = "api"

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider
        self._clients: dict[str, object] = {}

    def _resolve(self, model: str) -> str:
        if self.provider:
            return self.provider
        low = model.lower()
        if low.startswith("claude"):
            return "anthropic"
        if low.startswith(("gpt", "o1", "o3")):
            return "openai"
        if low.startswith("gemini"):
            return "google"
        raise ValueError(f"Cannot infer provider for model {model!r}; pass provider= explicitly")

    def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        oracle: str | None = None,  # ignored: real models must never see ground truth
    ) -> Generation:
        provider = self._resolve(model)
        if provider == "anthropic":
            text = self._anthropic(messages, model, temperature, max_tokens)
        elif provider == "openai":
            text = self._openai(messages, model, temperature, max_tokens, seed)
        elif provider == "google":
            text = self._google(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider {provider!r}")
        return Generation(text=text.strip(), backend=self.name, model=model)

    def _anthropic(self, messages, model, temperature, max_tokens) -> str:
        import anthropic  # noqa: PLC0415

        client = self._clients.setdefault(
            "anthropic", anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=model,
            system=system or anthropic.NOT_GIVEN,
            messages=convo,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def _openai(self, messages, model, temperature, max_tokens, seed) -> str:
        from openai import OpenAI  # noqa: PLC0415

        client = self._clients.setdefault("openai", OpenAI())
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return resp.choices[0].message.content or ""

    def _google(self, messages, model, temperature, max_tokens) -> str:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = self._clients.setdefault("google", genai.Client())
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""

"""The two models an optimizer talks to: the optimizer LLM that writes prompts and the task model that runs them.

`Completer` is the optimizer LLM (in all runs DeepSeek-V4-Pro through OpenRouter, up to 4,096 output tokens, no
reasoning block). `Scorer` runs a prompt on a batch of examples with the task model (Gemma 3 12B by default) and
returns labels, -1 for an unparsable answer. `MockCompleter` and `MockScorer` answer without any network or key so
that a run can be tested end to end for free.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from wgg.models import MODELS, Worker, load_key, message_text

OPTIMIZER_MODEL = "deepseek/deepseek-v4-pro"


class Completer:
    def __init__(self, key: str, model: str = OPTIMIZER_MODEL, temperature: float = 0.6, max_tokens: int = 4096,
                 timeout: int = 75, max_retries: int = 4):
        from openai import OpenAI
        from wgg.models import API_BASE
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.timeout, self.max_retries = timeout, max_retries
        self.client = OpenAI(base_url=API_BASE, api_key=key)

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature if temperature is None else float(temperature),
                    max_tokens=self.max_tokens if max_tokens is None else int(max_tokens), timeout=self.timeout)
                if not resp.choices:
                    raise RuntimeError("answer without choices")
                text = message_text(resp.choices[0].message)
                if not text:
                    raise RuntimeError(f"empty answer (finish_reason={getattr(resp.choices[0], 'finish_reason', None)})")
                return text
            except Exception as exc:  # noqa: BLE001
                last = exc
                slow = "429" in str(exc) or "rate-limited" in str(exc).lower()
                time.sleep(min(30.0, 4.0 * attempt) if slow else 1.5 * attempt)
        raise RuntimeError(f"query failed after {self.max_retries} attempts: {last}")

    def complete_n(self, prompt: str, n: int = 1, *, max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None) -> List[str]:
        """n sequential completions (the chat API has no reliable `n` across models)."""
        return [self.complete(prompt, max_tokens=max_tokens, temperature=temperature) for _ in range(max(1, int(n)))]


class Scorer:
    """The task model behind a batch interface: `predict_batch(texts, prompt)` -> int array."""

    def __init__(self, key: str, model: str = "gemma-3-12b", max_parallel: int = 4):
        spec = MODELS[model]
        self.worker = Worker(spec["id"], key, spec["provider"], False)
        self.max_parallel = max_parallel

    def predict_batch(self, texts, prompt, *, labels_for_mock=None):  # noqa: ARG002
        if not texts:
            return np.asarray([], dtype=np.int8)
        preds, _ = self.worker.label_rows(prompt, list(texts), self.max_parallel)
        return preds.astype(np.int8)


class MockScorer:
    """Deterministic stand-in for the task model: the gold label with some noise (needs the labels)."""

    def predict_batch(self, texts, prompt, *, labels_for_mock=None):
        labels = np.asarray(labels_for_mock if labels_for_mock is not None else [0] * len(texts))
        h = (sum(map(ord, prompt)) * 2654435761) % (2 ** 31)
        rng = np.random.RandomState(h)
        noise = rng.rand(len(labels)) < 0.15
        return np.where(noise, 1 - labels, labels).astype(np.int8)


class MockCompleter:
    """Stand-in for the optimizer LLM: always answers with a valid prompt wrapped in the tags ProTeGi parses."""

    def __init__(self, seed_prompt: str):
        self.seed_prompt, self.calls = seed_prompt, 0

    def complete(self, prompt, *, max_tokens=None, temperature=None):  # noqa: ARG002
        self.calls += 1
        return f"<START>{self.seed_prompt.replace('When unsure', 'If unsure')}<END>"

    def complete_n(self, prompt, n=1, *, max_tokens=None, temperature=None):
        return [self.complete(prompt, max_tokens=max_tokens, temperature=temperature) for _ in range(max(1, int(n)))]


__all__ = ["Completer", "Scorer", "MockScorer", "MockCompleter", "OPTIMIZER_MODEL", "load_key"]

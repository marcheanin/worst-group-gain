"""Querying a task model through OpenRouter, exactly as the paper's predictions were collected.

A prompt is a template with a `{review}` slot; the model answers with temperature 0 and at most 16
tokens, and the answer is parsed into a label. An answer the parser cannot read is marked -1
("unparsable"), never read as the negative class; analyses drop such examples for every prompt.

The key is read from the environment variable `OPENROUTER_API_KEY` or from a `.env` file in the working
directory, and is never printed.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

INVALID = -1
API_BASE = "https://openrouter.ai/api/v1"
# the log-probability of a label token that is absent from the top alternatives
LOGPROB_FLOOR = -60.0

MODELS = {
    "gemma-3-12b": {"id": "google/gemma-3-12b-it", "provider": None, "logprobs": False},
    "gpt-4o-mini": {"id": "openai/gpt-4o-mini", "provider": "OpenAI", "logprobs": True},
    "qwen3-235b": {"id": "qwen/qwen3-235b-a22b-2507", "provider": "DeepInfra", "logprobs": False},
    "qwen3-30b": {"id": "qwen/qwen3-30b-a3b-instruct-2507", "provider": "Alibaba", "logprobs": False},
}


def parse_binary(response: str) -> int:
    """Label 0/1 from a model answer: an explicit "label: 0/1", then the label words, then a bare digit.
    -1 when none of them matches."""
    text = (response or "").strip()
    patterns = [
        r"^(?:label|toxicity|answer|class)?\s*[:\-]?\s*([01])\s*$",
        r"\b(?:label|toxicity|answer|class)\s*[:\-]?\s*([01])\b",
        r"\b(non[\s\-]?toxic|not toxic|benign|safe)\b",
        r"\b(toxic|hate|abusive)\b",
        r"^([01])$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match or match.group(1) is None:
            continue
        low = match.group(1).lower()
        if low == "0" or "non" in low or low in ("benign", "safe"):
            return 0
        if low == "1" or low in ("toxic", "hate", "abusive"):
            return 1
    digits = re.findall(r"\b([01])\b", text)
    return int(digits[-1]) if digits else INVALID


def load_key(var: Optional[str] = None) -> str:
    """The API key from the environment or a `.env` file; `var` names the variable to use."""
    names = [var] if var else ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
    env = Path(".env")
    if env.is_file():
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise SystemExit(f"no API key: set {names[0]} or put it in a .env file")


def key_usage(key: str, retries: int = 3) -> Optional[float]:
    """Spend so far on this key in USD, or None when the endpoint cannot be reached (never raises: a
    failed balance check must not stop a long job)."""
    req = urllib.request.Request(f"{API_BASE}/auth/key", headers={"Authorization": f"Bearer {key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return float(json.load(r)["data"]["usage"])
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def label_logodds(choice) -> float:
    """log P('1') - log P('0') at the position of the label digit; NaN when no log-probabilities."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    for tok in content or []:
        if (getattr(tok, "token", "") or "").strip() in ("0", "1"):
            best = {"0": LOGPROB_FLOOR, "1": LOGPROB_FLOOR}
            for alt in [tok, *(getattr(tok, "top_logprobs", None) or [])]:
                s = (getattr(alt, "token", "") or "").strip()
                if s in best:
                    best[s] = max(best[s], float(alt.logprob))
            return best["1"] - best["0"]
    return float("nan")


def message_text(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else (p.get("text") if isinstance(p, dict) else getattr(p, "text", None))
                 for p in content]
        joined = "\n".join(str(p) for p in parts if p)
        if joined.strip():
            return joined.strip()
    return ""


class Worker:
    """One task model behind OpenRouter. `provider` pins the serving provider (without it OpenRouter may
    alternate between providers inside one run)."""

    def __init__(self, model: str, key: str, provider: Optional[str] = None, logprobs: bool = False,
                 timeout: int = 75, max_retries: int = 4):
        from openai import OpenAI
        self.model, self.provider, self.logprobs = model, provider, logprobs
        self.timeout, self.max_retries = timeout, max_retries
        self.client = OpenAI(base_url=API_BASE, api_key=key)
        self.retries = 0
        self.providers: Counter = Counter()

    def ask(self, prompt: str):
        """(answer text, log-odds of the label) for one prompt; retries with back-off, raises when they run out."""
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                extra: dict = {}
                if self.logprobs:
                    extra.update(logprobs=True, top_logprobs=20)
                if self.provider:
                    extra["extra_body"] = {"provider": {"order": [self.provider], "allow_fallbacks": False}}
                resp = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0,
                    max_tokens=16, timeout=self.timeout, **extra)
                if not resp.choices:
                    raise RuntimeError("answer without choices")
                choice = resp.choices[0]
                text = message_text(choice.message)
                if not text:
                    raise RuntimeError(f"empty answer (finish_reason={getattr(choice, 'finish_reason', None)})")
                if getattr(resp, "provider", None):
                    self.providers[str(resp.provider)] += 1
                return text, label_logodds(choice)
            except Exception as exc:  # noqa: BLE001
                last = exc
                self.retries += 1
                slow = "429" in str(exc) or "rate-limited" in str(exc).lower()
                time.sleep(min(30.0, 4.0 * attempt) if slow else 1.5 * attempt)
        raise RuntimeError(f"query failed after {self.max_retries} attempts: {last}")

    def label(self, template: str, text: str):
        """(label, log-odds): -1 when the template does not format, the query fails or the answer is unparsable."""
        try:
            prompt = template.format(review=text)
        except (KeyError, ValueError, IndexError):
            return INVALID, float("nan")
        try:
            answer, lo = self.ask(prompt)
            return parse_binary(answer), lo
        except Exception:  # noqa: BLE001
            return INVALID, float("nan")

    def label_rows(self, template: str, texts: list, max_parallel: int):
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            res = list(pool.map(lambda t: self.label(template, t), texts))
        return (np.asarray([r[0] for r in res], dtype=np.int16), np.asarray([r[1] for r in res], dtype=np.float32))

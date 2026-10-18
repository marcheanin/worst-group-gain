"""The prompt contract: what a candidate must satisfy before any query is spent on it.

A prompt must contain the slot `{review}`, format without error, be at most 8,000 characters, mention both
labels 0 and 1 and say what to output. A free-form instruction written by the optimizer LLM is wrapped into the
seed prompt (it replaces the `<DynamicRules>` block) so that it meets the contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from optimizers import profile


@dataclass
class ContractResult:
    ok: bool
    reason: Optional[str] = None


def check_prompt_contract(prompt: str, *, prompt_len_limit: int = 8000) -> ContractResult:
    text = prompt or ""
    if not text.strip():
        return ContractResult(False, "empty")
    if len(text) > prompt_len_limit:
        return ContractResult(False, "too_long")
    if "{review}" not in text and "{text}" not in text:
        return ContractResult(False, "missing_placeholder")
    try:
        text.format(review="sample comment") if "{review}" in text else text.format(text="sample comment")
    except (KeyError, ValueError, IndexError) as exc:
        return ContractResult(False, f"format_error:{type(exc).__name__}")
    if not (re.search(r"\b0\b", text) and re.search(r"\b1\b", text)):
        return ContractResult(False, "missing_binary_labels")
    if not re.search(r"(label|toxic|output|class)", text, re.IGNORECASE):
        return ContractResult(False, "missing_task_language")
    return ContractResult(True, None)


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def inject_instruction_into_seed(instruction: str, seed_prompt: str) -> str:
    """Place a free-form instruction into the seed's `<DynamicRules>` block, or build a prompt around it."""
    instr = strip_code_fence(instruction)
    instr = instr.replace("{text}", "{review}").replace("{review}", "the review")
    seed = seed_prompt or ""
    if re.search(r"<DynamicRules>.*?</DynamicRules>", seed, flags=re.DOTALL | re.IGNORECASE):
        return re.sub(r"<DynamicRules>.*?</DynamicRules>",
                      "<DynamicRules>\n" f"{instr}\n" f"{profile.active().out_line}\n" "    </DynamicRules>",
                      seed, count=1, flags=re.DOTALL | re.IGNORECASE)
    return f"{instr}\n\n{profile.active().answer_tail}"


def ensure_prompt_contract(instruction: str, *, seed_fallback: Optional[str] = None) -> str:
    """Wrap a free-form instruction into a prompt that meets the contract; the seed (or a last-resort prompt)
    when it cannot."""
    text = strip_code_fence(instruction).replace("{text}", "{review}")
    if seed_fallback and ("<System>" in seed_fallback or "<DynamicRules>" in seed_fallback):
        text = inject_instruction_into_seed(text, seed_fallback)
    elif "{review}" not in text:
        text = f"{text.rstrip()}\n\n{profile.active().answer_tail}"
    if check_prompt_contract(text).ok:
        return text
    return seed_fallback if seed_fallback else profile.active().last_resort

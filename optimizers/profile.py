"""The task words the methods insert into the instructions they give the optimizer LLM.

`CIVIL` is the profile of CivilComments and of multilingual toxicity (the same task and seed prompt); `MNLI` is
the contradiction task of MultiNLI, where `{review}` is filled with the pair "Premise: ...\\nHypothesis: ...".
The profile is chosen once per process with `set_active`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    name: str
    ape_fallback_instruction: str  # APE: instruction used when the optimizer LLM does not answer
    out_line: str                  # output line inside the <DynamicRules> block
    answer_tail: str               # tail of a prompt built around a free-form instruction
    last_resort: str               # prompt of last resort when the contract cannot be met
    mutate_head: str               # EvoPrompt: instruction for a mutation (ends with a blank line)
    crossover_head: str            # EvoPrompt: instruction for a crossover (ends with a blank line)
    default_seed: str              # seed prompt when the task has none
    reflect_head: str              # GEPA: head of the reflection prompt (ends with a blank line)
    reflect_item: str              # GEPA: one failure, fields {text} {gold} {pred}
    reflect_max_chars: int         # GEPA: characters of the text shown per failure


CIVIL = TaskProfile(
    name="civilcomments",
    ape_fallback_instruction=(
        "classify the comment as toxic (1) or non-toxic (0); "
        "mentioning identity alone is not toxic; output Label: 0 or Label: 1"
    ),
    out_line="        Output exactly Label: 0 (non-toxic) or Label: 1 (toxic).",
    answer_tail=("Output exactly 'Label: 0' (non-toxic) or 'Label: 1' (toxic).\n"
                 "Comment: {review}"),
    last_resort=("Classify the comment as toxic (1) or not (0). "
                 "Identity mentions alone are not toxic. "
                 "Output Label: 0 or Label: 1.\nComment: {review}"),
    mutate_head=("Mutate the prompt to improve group-robust toxicity classification. "
                 "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"),
    crossover_head=("Combine two parent prompts into one stronger toxicity classifier prompt. "
                    "Preserve {review} and Label: 0/1. Return only the full child prompt.\n\n"),
    default_seed="Classify toxicity. Output Label: 0 or Label: 1.\nComment: {review}",
    reflect_head=("You are reflecting on toxicity-classification failures. "
                  "Rewrite the prompt to fix these errors without breaking correct cases. "
                  "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"),
    reflect_item="Comment: {text}\nGold: {gold} Pred: {pred}",
    reflect_max_chars=240,
)

MNLI = TaskProfile(
    name="mnli",
    ape_fallback_instruction=(
        "decide whether the hypothesis contradicts the premise (1) or not (0); "
        "a negation word alone does not imply contradiction; output Label: 0 or Label: 1"
    ),
    out_line="        Output exactly Label: 0 (does not contradict) or Label: 1 (contradicts).",
    answer_tail=("Output exactly 'Label: 0' (does not contradict) or 'Label: 1' (contradicts).\n"
                 "{review}"),
    last_resort=("Decide whether the hypothesis contradicts the premise (1) or not (0). "
                 "A negation word alone does not imply contradiction. "
                 "Output Label: 0 or Label: 1.\n{review}"),
    mutate_head=("Mutate the prompt to improve group-robust natural language inference "
                 "(does the hypothesis contradict the premise). "
                 "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"),
    crossover_head=("Combine two parent prompts into one stronger contradiction-detection prompt "
                    "for premise/hypothesis pairs. "
                    "Preserve {review} and Label: 0/1. Return only the full child prompt.\n\n"),
    default_seed="Decide whether the hypothesis contradicts the premise. Output Label: 0 or Label: 1.\n{review}",
    reflect_head=("You are reflecting on failures of a contradiction classifier for premise/hypothesis pairs. "
                  "Rewrite the prompt to fix these errors without breaking correct cases. "
                  "Preserve {review} and Label: 0/1. Return only the full prompt.\n\n"),
    reflect_item="Pair: {text}\nGold: {gold} Pred: {pred}",
    reflect_max_chars=480,
)

_ACTIVE = CIVIL


def active() -> TaskProfile:
    return _ACTIVE


def set_active(profile: TaskProfile) -> None:
    global _ACTIVE
    _ACTIVE = profile

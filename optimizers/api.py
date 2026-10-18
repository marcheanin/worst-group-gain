"""What every optimizer receives and returns, and the budget counter.

All methods go through the same scorer (the task model, with fail-closed parsing), the same prompt contract and the
same selection statistic. A `Task` holds the labelled training examples the search learns from (`train`), the
labelled validation examples it selects on (`dev`), unlabelled texts (GPO), a scorer, the optimizer LLM and the
seed prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from optimizers import fitness


@dataclass
class LabeledSet:
    texts: List[str]
    labels: List[int]
    group_ids: Optional[List[int]] = None
    example_ids: Optional[List[int]] = None


@dataclass
class UnlabeledSet:
    texts: List[str]
    group_ids: Optional[List[int]] = None
    example_ids: Optional[List[int]] = None


@dataclass
class Budget:
    """Counts the calls made by a run (scorer = task model, optimizer = optimizer LLM)."""

    scorer_calls: int = 0
    optimizer_calls: int = 0

    def charge(self, level: str, n_calls: int = 1, *, kind: str = "scorer", note: str = "") -> None:  # noqa: ARG002
        if kind == "scorer":
            self.scorer_calls += int(n_calls)
        elif kind == "optimizer":
            self.optimizer_calls += int(n_calls)


@dataclass
class Task:
    train: LabeledSet
    dev: LabeledSet
    unlabeled: UnlabeledSet
    scorer: Any            # predict_batch(texts, prompt) -> int array, -1 = unparsable
    optimizer_llm: Any     # complete(prompt, max_tokens=, temperature=), complete_n(...)
    budget: Budget
    seed_prompt: Optional[str] = None
    label_budget: int = 240
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerResult:
    best_prompt: str
    all_candidates: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    scorer_calls: int = 0
    optimizer_calls: int = 0


def score_prompt_on_set(scorer, prompt: str, labeled: LabeledSet) -> Dict[str, Any]:
    """Score a prompt on a labelled set and return its fitness record (see `fitness.evaluate`)."""
    preds = scorer.predict_batch(labeled.texts, prompt, labels_for_mock=labeled.labels)
    # without group ids no group has a balanced accuracy and the prompt is rejected ("gba_empty")
    groups = labeled.group_ids if labeled.group_ids else [0] * len(labeled.labels)
    return fitness.evaluate(preds, labeled.labels, groups, prompt)

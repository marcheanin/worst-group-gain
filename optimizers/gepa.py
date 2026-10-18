"""GEPA (Agrawal et al., 2026), reimplemented as a lightweight adapter, not the authors' package.

The core loop is kept: a reflective mutation from error feedback, and a Pareto archive over per-example
correctness, capped by the number of metric calls. The archive holds at most 8 prompts; each step picks a parent
at random, runs it on a subsample of 400 validation examples, shows the optimizer LLM up to six of its errors, and
scores the rewritten prompt; the child enters the archive if no prompt dominates it (more correct examples, as a
set, or the same set and a higher score), and the prompts it dominates leave. The best prompt of the archive is
scored on the whole validation set.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from optimizers import profile
from optimizers.api import LabeledSet, OptimizerResult, Task
from optimizers.contract import ensure_prompt_contract
from optimizers import fitness


@dataclass
class _Cand:
    prompt: str
    score: float
    correct: Set[int] = field(default_factory=set)


def subsample(dev: LabeledSet, n: int, seed: int) -> LabeledSet:
    if len(dev.texts) <= n:
        return dev
    idxs = list(range(len(dev.texts)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]
    return LabeledSet(texts=[dev.texts[i] for i in idxs], labels=[dev.labels[i] for i in idxs],
                      group_ids=[dev.group_ids[i] for i in idxs] if dev.group_ids else None, example_ids=list(idxs))


def evaluate_candidate(task: Task, prompt: str, dev: LabeledSet, note: str) -> _Cand:
    preds = [int(x) for x in task.scorer.predict_batch(list(dev.texts), prompt, labels_for_mock=dev.labels)]
    task.budget.charge("val", n_calls=len(dev.texts), kind="scorer", note=note)
    correct = {i for i, (p, y) in enumerate(zip(preds, dev.labels)) if p == int(y)}
    groups = dev.group_ids if dev.group_ids else [0] * len(dev.labels)
    fit = fitness.evaluate(np.asarray(preds), np.asarray(dev.labels), np.asarray(groups), prompt)
    return _Cand(prompt, float(fit.get("R_soft_min_gba", fit.get("fitness", 0.0))), correct)


def pareto_keep(cands: Sequence[_Cand], max_keep: int) -> List[_Cand]:
    """Non-dominated candidates by inclusion of the sets of correctly answered examples, then by score."""
    kept: List[_Cand] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        dominated = any(c.correct < o.correct or (c.correct <= o.correct and c.score < o.score - 1e-9) for o in kept)
        if not dominated:
            kept = [o for o in kept if not (o.correct < c.correct or (o.correct <= c.correct and o.score < c.score - 1e-9))]
            kept.append(c)
        if len(kept) >= max_keep:
            break
    return kept[:max_keep]


def reflect(llm, prompt: str, errors: Sequence[Tuple[str, int, int]], seed_prompt, budget) -> str:
    p = profile.active()
    lines = [p.reflect_item.format(text=text[: p.reflect_max_chars], gold=gold, pred=pred) for text, gold, pred in errors[:6]]
    meta = p.reflect_head + f"Current prompt:\n{prompt}\n\nFailures:\n" + "\n\n".join(lines)
    try:
        raw = llm.complete(meta, max_tokens=900, temperature=0.7)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note="gepa_reflect")
        return ensure_prompt_contract(raw, seed_fallback=seed_prompt)
    except Exception:  # noqa: BLE001
        return prompt


class GEPA:
    def __init__(self, *, max_metric_calls: int = 60_000, population: int = 8, reflections: int = 12,
                 dev_subsample: int = 400):
        self.max_metric_calls, self.population = max_metric_calls, population
        self.reflections, self.dev_subsample = reflections, dev_subsample

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        rng = random.Random(seed)
        seed_prompt = task.seed_prompt or profile.active().default_seed
        dev = subsample(task.dev, self.dev_subsample, seed)
        calls0 = int(task.budget.scorer_calls)
        print("[gepa] evaluate the seed", flush=True)
        front = [evaluate_candidate(task, seed_prompt, dev, "gepa_init")]
        history: List[Dict] = [{"step": 0, "best": front[0].score, "front": 1}]
        for step in range(1, self.reflections + 1):
            if int(task.budget.scorer_calls) - calls0 >= self.max_metric_calls:
                print(f"[gepa] budget exhausted at step {step}", flush=True)
                break
            parent = rng.choice(front)
            preds = task.scorer.predict_batch(list(dev.texts), parent.prompt, labels_for_mock=dev.labels)
            task.budget.charge("val", n_calls=len(dev.texts), kind="scorer", note="gepa_err")
            errors = [(dev.texts[i], int(dev.labels[i]), int(preds[i])) for i in range(len(dev.texts))
                      if int(preds[i]) != int(dev.labels[i])]
            rng.shuffle(errors)
            child = evaluate_candidate(task, reflect(task.optimizer_llm, parent.prompt, errors, seed_prompt, task.budget),
                                       dev, f"gepa_s{step}")
            front = pareto_keep(front + [child], self.population)
            best = max(front, key=lambda c: c.score)
            history.append({"step": step, "best": best.score, "front": len(front)})
            print(f"[gepa] step {step}/{self.reflections} best={best.score:.4f} front={len(front)}", flush=True)
        best = max(front, key=lambda c: c.score)
        full = evaluate_candidate(task, best.prompt, task.dev, "gepa_full_dev")
        return OptimizerResult(
            best_prompt=best.prompt, all_candidates=[c.prompt for c in front],
            trace={"method": "gepa", "max_metric_calls": self.max_metric_calls, "reflections": self.reflections,
                   "dev_subsample": self.dev_subsample, "history": history, "best_softmin_sub": best.score,
                   "best_softmin_full_dev": full.score},
            scorer_calls=task.budget.scorer_calls, optimizer_calls=task.budget.optimizer_calls)

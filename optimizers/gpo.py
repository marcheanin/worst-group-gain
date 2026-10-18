"""GPO (Li et al., 2023): robust prompt optimization in three steps, following `run_gpo.py` of the authors' code.

1. APE generates K = 6 prompts from N = 36 labelled source examples.
2. The K prompts label an unlabelled pool from the target groups; the examples whose majority label has agreement
   above T = 0.83 (more than five of six) are kept and upsampled to the size of the pool.
3. APE runs again on the source examples mixed with the pseudo-labelled target examples; the final is the candidate
   with the best selection statistic on the validation set.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from optimizers.ape import generate_candidates, select_best_on_dev
from optimizers.api import LabeledSet, OptimizerResult, Task


def ensemble_label(prompts: Sequence[str], texts: Sequence[str], scorer, budget, *,
                   conf_threshold: float = 0.83) -> Tuple[LabeledSet, dict]:
    """Majority label of the K prompts per text; keep those with confidence above the threshold, upsample to |U|."""
    if not prompts or not texts:
        return LabeledSet(texts=[], labels=[]), {"kept": 0, "total": 0}
    votes: List[List[Optional[int]]] = []
    for pi, p in enumerate(prompts):
        print(f"[gpo] ensemble-label prompt {pi + 1}/{len(prompts)} on |U|={len(texts)}", flush=True)
        preds = scorer.predict_batch(list(texts), p, labels_for_mock=None)
        budget.charge("val", n_calls=len(texts), kind="scorer", note="gpo_pseudo")
        votes.append([int(x) if int(x) in (0, 1) else None for x in preds])
    k = len(prompts)
    kept_texts: List[str] = []
    kept_labels: List[int] = []
    confidences: List[float] = []
    for i, text in enumerate(texts):
        preds_i = [votes[j][i] for j in range(k) if votes[j][i] is not None]
        if not preds_i:
            confidences.append(0.0)
            continue
        label, count = Counter(preds_i).most_common(1)[0]
        conf = count / float(k)
        confidences.append(conf)
        if conf > conf_threshold:
            kept_texts.append(text)
            kept_labels.append(int(label))
    total, kept = len(texts), len(kept_texts)
    if 0 < kept < total:  # upsample to |U| as in the authors' `label_data`
        rng = random.Random(0)
        pairs = list(zip(kept_texts, kept_labels))
        rep = total // kept
        up = pairs * rep + rng.sample(pairs, total - kept * rep)
        rng.shuffle(up)
        kept_texts, kept_labels = [t for t, _ in up], [y for _, y in up]
    elif kept == 0:
        kept_texts, kept_labels = [], []
    return LabeledSet(texts=kept_texts, labels=kept_labels), {
        "kept_before_upsample": kept, "total": total, "conf_threshold": conf_threshold,
        "mean_confidence": float(sum(confidences) / max(1, len(confidences))),
        "frac_above_T": float(kept / max(1, total)), "labeled_after_upsample": len(kept_texts)}


def mix_source_target(source: LabeledSet, target: LabeledSet, *, seed: int) -> LabeledSet:
    texts, labels = list(source.texts) + list(target.texts), list(source.labels) + list(target.labels)
    gids = None
    if source.group_ids is not None or target.group_ids is not None:
        gids = list(source.group_ids or [0] * len(source.texts)) + list(target.group_ids or [0] * len(target.texts))
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    return LabeledSet(texts=[texts[i] for i in order], labels=[labels[i] for i in order],
                      group_ids=[gids[i] for i in order] if gids is not None else None)


class GPO:
    def __init__(self, *, k_prompts: int = 6, n_shots: int = 36, conf_threshold: float = 0.83, max_tokens: int = 100,
                 temperature: float = 0.0):
        self.k_prompts, self.n_shots, self.conf_threshold = k_prompts, n_shots, conf_threshold
        self.max_tokens, self.temperature = max_tokens, temperature

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        kw = dict(k_prompts=self.k_prompts, n_shots=self.n_shots, max_tokens=self.max_tokens,
                  temperature=self.temperature, seed_prompt=task.seed_prompt, budget=task.budget)
        print("[gpo] step 1: APE on the source examples", flush=True)
        stage1 = generate_candidates(task.train, task.optimizer_llm, seed=seed, **kw)
        ensemble = stage1[: self.k_prompts] if len(stage1) >= self.k_prompts else stage1
        u_texts = list(task.unlabeled.texts)
        if not u_texts:
            raise ValueError("GPO needs an unlabelled target pool")
        pseudo, label_trace = ensemble_label(ensemble, u_texts, task.scorer, task.budget,
                                             conf_threshold=self.conf_threshold)
        print(f"[gpo] step 2 done: {label_trace}", flush=True)
        mixed = mix_source_target(task.train, pseudo, seed=seed + 1) if pseudo.texts else task.train
        print(f"[gpo] step 3: APE on {len(mixed.texts)} mixed examples", flush=True)
        stage3 = generate_candidates(mixed, task.optimizer_llm, seed=seed + 2, **kw)
        best, best_score, scored = select_best_on_dev(stage3, task, note="gpo_dev")
        return OptimizerResult(
            best_prompt=best, all_candidates=list(stage3),
            trace={"method": "gpo", "k": self.k_prompts, "n_shots": self.n_shots, "conf_threshold": self.conf_threshold,
                   "stage1_n": len(stage1), "label": label_trace, "mixed_n": len(mixed.texts), "scored": scored,
                   "best_softmin": best_score},
            scorer_calls=task.budget.scorer_calls, optimizer_calls=task.budget.optimizer_calls)

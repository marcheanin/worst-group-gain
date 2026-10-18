"""ProTeGi (Pryzant et al., 2023, also called APO): textual gradients and beam search.

A port of `ProTeGi` and the round loop of `prompt_optimization/main.py` in the authors' code (LMOps), with their
defaults: 6 rounds, beam 4, minibatches of 64, 4 gradients per prompt, 4 errors per gradient, 1 step per gradient,
2 Monte-Carlo samples per step, at most 8 expanded prompts per beam prompt. Differences: the optimizer LLM and the
scorer are ours, the beam is ranked on the labelled training set by the smoothed worst-group balanced accuracy,
the final beam is selected on the validation set by the selection statistic, and the seed prompt is rewritten into
ProTeGi's sections ("# Task", "# Output format", "# Prediction").

`RandomLabels` and `AllLabels` run the same search with extra labelled target examples: 240 random ones of the
unlabelled target pool, or the whole pool (capped at 800).
"""
from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List, Optional, Sequence

from optimizers.api import LabeledSet, OptimizerResult, Task, UnlabeledSet, score_prompt_on_set
from optimizers.contract import ensure_prompt_contract


def parse_sectioned_prompt(s: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    current = None
    for line in s.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            current = stripped[2:].strip().lower().split()[0]
            current = current.translate(str.maketrans("", "", string.punctuation))
            result[current] = ""
        elif current is not None:
            result[current] += line + "\n"
    return result


def seed_to_protegi(seed_prompt: str) -> str:
    """Our XML seed in ProTeGi's sectioned format."""
    m = re.search(r"<DynamicRules>(.*?)</DynamicRules>", seed_prompt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        task_body = m.group(1).strip()
    else:
        task_body = re.sub(r"</?System>|</?Role>|</?BaseGuidelines>|</?DynamicRules>|</?FewShotExamples>|</?Task>",
                           "", seed_prompt)
        task_body = task_body.replace("Comment: {review}", "").replace("Review: {review}", "").strip()
    return ("# Task\n" f"{task_body.strip()}\n\n" "# Output format\n" "Output exactly 'Label: 0' or 'Label: 1'.\n\n"
            "# Prediction\n" "Text: {review}\n" "Label:\n")


def parse_tagged_text(text: str, start_tag: str, end_tag: str) -> List[str]:
    texts: List[str] = []
    while True:
        i = text.find(start_tag)
        if i == -1:
            break
        j = text.find(end_tag, i)
        if j == -1:
            break
        i += len(start_tag)
        texts.append(text[i:j].strip())
        text = text[j + len(end_tag):]
    return texts


class Core:
    """Expansion of a beam: error samples, gradients, edits and Monte-Carlo rewrites."""

    def __init__(self, opt: Dict[str, Any], llm) -> None:
        self.opt, self.llm = opt, llm

    def _error_str(self, texts, labels, preds, n=4) -> str:
        errors = [i for i, (l, p) in enumerate(zip(labels, preds)) if int(l) != int(p)]
        out = ""
        for k, i in enumerate(random.sample(errors, min(len(errors), n))):
            out += f"## Example {k + 1}\n" f'Text: "{texts[i].strip()}"\n' f"Label: {int(labels[i])}\n" \
                   f"Prediction: {int(preds[i])}\n\n"
        return out.strip()

    def _gradients(self, prompt: str, error_string: str, num_feedbacks: int = 5, n: int = 1) -> List[str]:
        text = f"""
        I'm trying to write a zero-shot classifier prompt.

        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        text = "\n".join(line.lstrip() for line in text.split("\n"))
        feedbacks: List[str] = []
        for r in self.llm.complete_n(text, n=n, temperature=0.7, max_tokens=1024):
            feedbacks += parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt: str, error_str: str, feedback_str: str, steps: int, n: int = 1) -> List[str]:
        text = f"""
        I'm trying to write a zero-shot classifier.

        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps} new prompts are:
        """
        text = "\n".join(line.lstrip() for line in text.split("\n"))
        new: List[str] = []
        for r in self.llm.complete_n(text, n=n, temperature=0.7, max_tokens=1024):
            new += parse_tagged_text(r, "<START>", "<END>")
        return new

    def synonyms(self, section: str, n: int = 3) -> List[str]:
        text = ("Generate a variation of the following instruction while keeping the "
                f"semantic meaning.\n\nInput: {section}\n\nOutput:")
        return [x for x in self.llm.complete_n(text, n=n, temperature=0.7, max_tokens=512) if x]

    def expand(self, prompts: Sequence[str], texts, labels, scorer, budget) -> List[str]:
        n_mb = min(int(self.opt["minibatch_size"]), len(texts))
        idxs = random.sample(range(len(texts)), k=n_mb)
        mb_texts, mb_labels = [texts[i] for i in idxs], [labels[i] for i in idxs]
        new_prompts: List[str] = []
        for prompt in prompts:
            sections = parse_sectioned_prompt(prompt)
            if "task" not in sections:
                prompt = seed_to_protegi(prompt)
                sections = parse_sectioned_prompt(prompt)
            task_section = sections.get("task", "").strip()
            if not task_section:
                continue
            preds = [int(x) for x in scorer.predict_batch(mb_texts, prompt, labels_for_mock=mb_labels)]
            budget.charge("val", n_calls=len(mb_texts), kind="scorer", note="apo_minibatch")
            new_sections: List[str] = []
            for _ in range(int(self.opt["n_gradients"])):
                error_string = self._error_str(mb_texts, mb_labels, preds, n=int(self.opt["errors_per_gradient"]))
                if not error_string:
                    continue
                gradients = self._gradients(task_section, error_string, int(self.opt["gradients_per_error"]), n=1)
                budget.charge("mutator", n_calls=1, kind="optimizer", note="apo_grad")
                for feedback in gradients:
                    new_sections += self.apply_gradient(task_section, error_string, feedback,
                                                        int(self.opt["steps_per_gradient"]))
                    budget.charge("mutator", n_calls=1, kind="optimizer", note="apo_apply")
            sampled: List[str] = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in new_sections + [task_section]:
                    sampled += self.synonyms(sect, n=int(self.opt["mc_samples_per_step"]))
                    budget.charge("mutator", n_calls=int(self.opt["mc_samples_per_step"]), kind="optimizer", note="apo_mc")
            tmp = [prompt.replace(task_section, s) for s in list(set(new_sections + sampled))]
            if len(tmp) > int(self.opt["max_expansion_factor"]):
                tmp = random.sample(tmp, k=int(self.opt["max_expansion_factor"]))
            new_prompts += tmp
        new_prompts += list(prompts)
        seen, out = set(), []
        for p in new_prompts:
            p2 = p if "{review}" in p else ensure_prompt_contract(p)
            if p2 not in seen:
                seen.add(p2)
                out.append(p2)
        return out


def _score_all(prompts, task: Task, labeled, note: str) -> List[float]:
    scores = []
    for p in prompts:
        m = score_prompt_on_set(task.scorer, p, labeled)
        task.budget.charge("val", n_calls=len(labeled.texts), kind="scorer", note=note)
        scores.append(float(m.get("R_soft_min_gba", m.get("fitness", 0.0))))
    return scores


class ProTeGi:
    def __init__(self, *, rounds: int = 6, beam_size: int = 4, minibatch_size: int = 64, n_gradients: int = 4,
                 errors_per_gradient: int = 4, gradients_per_error: int = 1, steps_per_gradient: int = 1,
                 mc_samples_per_step: int = 2, max_expansion_factor: int = 8):
        self.rounds, self.beam_size = rounds, beam_size
        self.opt = {"minibatch_size": minibatch_size, "n_gradients": n_gradients,
                    "errors_per_gradient": errors_per_gradient, "gradients_per_error": gradients_per_error,
                    "steps_per_gradient": steps_per_gradient, "mc_samples_per_step": mc_samples_per_step,
                    "max_expansion_factor": max_expansion_factor, "reject_on_errors": False}

    def run(self, task: Task) -> OptimizerResult:
        random.seed(int(task.meta.get("seed", 42)))
        if not task.seed_prompt or task.optimizer_llm is None:
            raise ValueError("ProTeGi needs a seed prompt and an optimizer LLM")
        core = Core(self.opt, task.optimizer_llm)
        candidates = [seed_to_protegi(task.seed_prompt)]
        history: List[dict] = []
        for rnd in range(self.rounds + 1):
            print(f"[protegi] round {rnd}/{self.rounds} beam={len(candidates)}", flush=True)
            if rnd > 0:
                candidates = core.expand(candidates, task.train.texts, task.train.labels, task.scorer, task.budget)
                print(f"[protegi] expanded to {len(candidates)} candidates", flush=True)
            scores = _score_all(candidates, task, task.train, "apo_train_beam")
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            scores = [s for s, _ in ranked[: self.beam_size]]
            candidates = [p for _, p in ranked[: self.beam_size]]
            history.append({"round": rnd, "beam_train_softmin": scores, "beam_lens": [len(p) for p in candidates]})
        dev_scores = _score_all(candidates, task, task.dev, "apo_dev_final")
        best_i = max(range(len(candidates)), key=lambda i: dev_scores[i])
        return OptimizerResult(
            best_prompt=candidates[best_i], all_candidates=list(candidates),
            trace={"method": "apo_protegi", "rounds": self.rounds, "beam_size": self.beam_size, "opt": self.opt,
                   "history": history, "dev_softmin_beam": dev_scores, "best_softmin": dev_scores[best_i]},
            scorer_calls=task.budget.scorer_calls, optimizer_calls=task.budget.optimizer_calls)


def _merge(a: LabeledSet, b: LabeledSet, *, seed: int) -> LabeledSet:
    texts, labels = list(a.texts) + list(b.texts), list(a.labels) + list(b.labels)
    gids = None
    if a.group_ids is not None or b.group_ids is not None:
        gids = list(a.group_ids or [0] * len(a.texts)) + list(b.group_ids or [0] * len(b.texts))
    eids = None
    if a.example_ids is not None or b.example_ids is not None:
        eids = list(a.example_ids or list(range(len(a.texts)))) + list(b.example_ids or list(range(len(b.texts))))
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    return LabeledSet(texts=[texts[i] for i in order], labels=[labels[i] for i in order],
                      group_ids=[gids[i] for i in order] if gids is not None else None,
                      example_ids=[eids[i] for i in order] if eids is not None else None)


class WithTargetLabels:
    """ProTeGi on the training set plus labelled target examples. `reveal(ids)` returns the labelled set of
    the unlabelled pool's examples with those ids (the gold labels the pool hides)."""

    def __init__(self, reveal, *, mode: str, label_budget: int = 240, max_target_labels: Optional[int] = None,
                 rounds: int = 6, beam_size: int = 4):
        assert mode in ("random", "all")
        self.reveal, self.mode, self.label_budget = reveal, mode, label_budget
        self.max_target_labels, self.rounds, self.beam_size = max_target_labels, rounds, beam_size

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        ids = list(task.unlabeled.example_ids or [])
        if not ids:
            raise ValueError("the target pool needs example ids")
        if self.mode == "random":
            n = min(self.label_budget, len(ids))
            ids = sorted(random.Random(seed + 99).sample(ids, k=n))
        elif self.max_target_labels is not None and len(ids) > self.max_target_labels:
            ids = sorted(random.Random(seed).sample(ids, k=self.max_target_labels))
        extra = self.reveal(ids)
        mixed = _merge(task.train, extra, seed=seed)
        inner = Task(train=mixed, dev=task.dev, unlabeled=UnlabeledSet(texts=[]), scorer=task.scorer,
                     optimizer_llm=task.optimizer_llm, budget=task.budget, seed_prompt=task.seed_prompt,
                     label_budget=self.label_budget if self.mode == "random" else len(ids), meta=dict(task.meta))
        result = ProTeGi(rounds=self.rounds, beam_size=self.beam_size).run(inner)
        result.trace = {**result.trace, "method": "random_al" if self.mode == "random" else "oracle",
                        "extra_n": len(extra.texts), "mixed_n": len(mixed.texts), "base": "apo_protegi"}
        return result

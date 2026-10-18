"""EvoPrompt (Guo et al., 2024), genetic (GA) and differential-evolution (DE) variants, reimplemented from the paper.

The population starts from the seed prompt and paraphrases of it written by the optimizer LLM. Each generation
produces one child per population slot, scored on a subsample of 500 validation examples; the best `population`
prompts of parents and children survive, and the best survivor is scored on the whole validation set.

  GA  a child is the crossover of two parents chosen by roulette on their scores, or, with probability 0.2, a
      mutation of one parent;
  DE  three parents are drawn: the first two are crossed, the result is mutated, and with probability 0.5 it is
      crossed with the third.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from optimizers import profile
from optimizers.api import LabeledSet, OptimizerResult, Task, score_prompt_on_set
from optimizers.contract import ensure_prompt_contract


@dataclass
class _Cand:
    prompt: str
    score: float


def subsample(dev: LabeledSet, n: int, seed: int) -> LabeledSet:
    if len(dev.texts) <= n:
        return dev
    order = list(range(len(dev.texts)))
    random.Random(seed).shuffle(order)
    order = order[:n]
    return LabeledSet(texts=[dev.texts[i] for i in order], labels=[dev.labels[i] for i in order],
                      group_ids=[dev.group_ids[i] for i in order] if dev.group_ids else None,
                      example_ids=[dev.example_ids[i] for i in order] if dev.example_ids else None)


def _score(task: Task, prompt: str, dev: LabeledSet, note: str) -> float:
    m = score_prompt_on_set(task.scorer, prompt, dev)
    task.budget.charge("val", n_calls=len(dev.texts), kind="scorer", note=note)
    return float(m.get("R_soft_min_gba", m.get("R_global", m.get("fitness", 0.0))))


def _write(llm, meta: str, fallback: str, seed_prompt, budget, note: str) -> str:
    try:
        raw = llm.complete(meta, max_tokens=800, temperature=0.8)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note=note)
        return ensure_prompt_contract(raw, seed_fallback=seed_prompt)
    except Exception:  # noqa: BLE001
        return fallback


def paraphrase(llm, prompt, seed_prompt, budget) -> str:
    meta = ("Paraphrase the following classification prompt while preserving the "
            "{review} placeholder and binary Label: 0/1 output format. "
            "Return only the full prompt.\n\n" f"{prompt}")
    return _write(llm, meta, prompt, seed_prompt, budget, "evoprompt_para")


def mutate(llm, prompt, seed_prompt, budget) -> str:
    return _write(llm, profile.active().mutate_head + f"{prompt}", prompt, seed_prompt, budget, "evoprompt_mut")


def crossover(llm, a, b, seed_prompt, budget) -> str:
    return _write(llm, profile.active().crossover_head + f"Parent A:\n{a}\n\nParent B:\n{b}", a, seed_prompt, budget,
                  "evoprompt_xover")


def roulette(pop: Sequence[_Cand], rng: random.Random) -> _Cand:
    scores = [max(0.0, c.score) + 1e-6 for c in pop]
    r = rng.random() * sum(scores)
    acc = 0.0
    for c, s in zip(pop, scores):
        acc += s
        if acc >= r:
            return c
    return pop[-1]


class EvoPrompt:
    def __init__(self, *, mode: str = "ga", population_size: int = 10, generations: int = 10,
                 mutation_rate: float = 0.2, dev_subsample: int = 500):
        self.mode = mode.lower().strip()
        self.population_size, self.generations = population_size, generations
        self.mutation_rate, self.dev_subsample = mutation_rate, dev_subsample

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        rng = random.Random(seed)
        llm, budget = task.optimizer_llm, task.budget
        seed_prompt = task.seed_prompt or profile.active().default_seed
        dev = subsample(task.dev, self.dev_subsample, seed)
        prompts: List[str] = [seed_prompt]
        while len(prompts) < self.population_size:
            prompts.append(paraphrase(llm, seed_prompt, seed_prompt, budget))
        pop: List[_Cand] = []
        for i, p in enumerate(prompts):
            print(f"[evoprompt-{self.mode}] init {i + 1}/{len(prompts)}", flush=True)
            pop.append(_Cand(p, _score(task, p, dev, f"evo_{self.mode}_init")))
        pop.sort(key=lambda c: c.score, reverse=True)
        history = [{"gen": 0, "best": pop[0].score}]
        for gen in range(1, self.generations + 1):
            children: List[_Cand] = []
            for j in range(self.population_size):
                if self.mode == "de":
                    a, b, c = rng.sample(pop, k=min(3, len(pop)))
                    child = mutate(llm, crossover(llm, a.prompt, b.prompt, seed_prompt, budget), seed_prompt, budget)
                    if rng.random() < 0.5:
                        child = crossover(llm, child, c.prompt, seed_prompt, budget)
                elif len(pop) > 1 and rng.random() > self.mutation_rate:
                    child = crossover(llm, roulette(pop, rng).prompt, roulette(pop, rng).prompt, seed_prompt, budget)
                else:
                    child = mutate(llm, roulette(pop, rng).prompt, seed_prompt, budget)
                print(f"[evoprompt-{self.mode}] gen {gen} child {j + 1}/{self.population_size}", flush=True)
                children.append(_Cand(child, _score(task, child, dev, f"evo_{self.mode}_g{gen}")))
            pop = sorted(pop + children, key=lambda c: c.score, reverse=True)[: self.population_size]
            history.append({"gen": gen, "best": pop[0].score})
        best = pop[0]
        full = _score(task, best.prompt, task.dev, f"evo_{self.mode}_full_dev")
        return OptimizerResult(
            best_prompt=best.prompt, all_candidates=[c.prompt for c in pop],
            trace={"method": f"evoprompt_{self.mode}", "population_size": self.population_size,
                   "generations": self.generations, "dev_subsample": self.dev_subsample, "history": history,
                   "best_softmin_sub": best.score, "best_softmin_full_dev": full},
            scorer_calls=task.budget.scorer_calls, optimizer_calls=task.budget.optimizer_calls)

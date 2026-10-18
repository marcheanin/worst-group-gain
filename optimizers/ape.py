"""APE (Zhou et al., 2023), following its implementation in the GPO code (Li et al., 2023).

The optimizer LLM is shown K = 6 subsets of N = 36 input-output examples and asked to continue "The instruction
is to ..."; each continuation (temperature 0, at most 100 tokens) becomes a candidate prompt, wrapped into the
seed prompt by the prompt contract, and the seed is added. There is no Monte-Carlo search: the final is the
candidate with the best selection statistic on the validation set. The variants `ape_k48` (K = 48 proposals from
288 examples, six per proposal) and `ape_ut` (the meta-prompt also shows unlabelled texts from the target groups,
as in the GPO paper) are used in the CivilComments matrix.
"""
from __future__ import annotations

import json
import random
import re
from typing import List, Optional, Sequence, Tuple

from optimizers import profile
from optimizers.api import LabeledSet, OptimizerResult, Task, score_prompt_on_set
from optimizers.contract import ensure_prompt_contract

META_PROMPT = (
    "I provide my friend with an instruction. Based on the instruction, I gave him "
    "several inputs, and he generated the corresponding outputs. Here are the "
    "input-output examples:\n\n[full_DEMO]\n\nPlease briefly illustrate the "
    "instruction and describe the output format. The instruction is to [APE]"
)
DEMO = "Input: [INPUT]\nOutput: [OUTPUT]"


def format_demos(texts: Sequence[str], labels: Sequence[int]) -> str:
    return "\n\n".join(DEMO.replace("[INPUT]", t[:400].replace("\n", " ")).replace("[OUTPUT]", str(int(y)))
                       for t, y in zip(texts, labels))


def format_demos_with_unlabeled(texts, labels, unlabeled) -> str:
    parts: List[str] = [f"Input: {t[:400].replace(chr(10), ' ')}\nOutput: {int(y)}" for t, y in zip(texts, labels)]
    if unlabeled:
        parts.append("Here are additional unlabeled inputs from the target distribution "
                     "(no outputs). Use them to infer a more robust instruction:")
        parts += [f"Input: {t[:400].replace(chr(10), ' ')}\nOutput: ?" for t in unlabeled]
    return "\n\n".join(parts)


def extract_instruction(raw: str) -> str:
    text = (raw or "").strip()
    if "[APE]" in text:
        text = text.split("[APE]", 1)[-1].strip()
    for stop in ("\n\nInput:", "\nInput:", "\n\nComment:", "\n\n# "):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def _dedup(prompts: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _ask(llm, meta: str, max_tokens: int, temperature: float, budget, note: str, label: str) -> str:
    meta = meta[: -len("[APE]")] if meta.endswith("[APE]") else meta
    try:
        if llm is None:
            raise RuntimeError("no optimizer LLM")
        print(f"[{label}] generate candidate", flush=True)
        raw = llm.complete(meta, max_tokens=max_tokens, temperature=temperature)
        if budget is not None:
            budget.charge("mutator", n_calls=1, kind="optimizer", note=note)
        return extract_instruction(raw)
    except Exception:  # noqa: BLE001
        return profile.active().ape_fallback_instruction


def generate_candidates(train: LabeledSet, llm, *, k_prompts=6, n_shots=36, max_tokens=100, temperature=0.0,
                        seed=42, seed_prompt: Optional[str] = None, budget=None) -> List[str]:
    """K instruction prompts from shot subsets of the training set, plus the seed."""
    rng = random.Random(int(seed))
    # budget-matched APE (K = 48) needs six demonstrations per subset, up to 288; sample with replacement if short
    demos_per = 6 if k_prompts >= 12 else max(1, min(int(n_shots), len(train.texts)) // max(1, k_prompts))
    need = max(int(n_shots), k_prompts * demos_per)
    pool_idx = list(range(len(train.texts)))
    rng.shuffle(pool_idx)
    if len(pool_idx) < need:
        while len(pool_idx) < need:
            pool_idx.append(rng.choice(range(len(train.texts))))
    else:
        pool_idx = pool_idx[:need]
    texts = [train.texts[i] for i in pool_idx]
    labels = [train.labels[i] for i in pool_idx]
    candidates: List[str] = []
    for k in range(k_prompts):
        ct, cy = texts[k * demos_per:(k + 1) * demos_per], labels[k * demos_per:(k + 1) * demos_per]
        if not ct:
            ct, cy = texts[:demos_per], labels[:demos_per]
        instr = _ask(llm, META_PROMPT.replace("[full_DEMO]", format_demos(ct, cy)), max_tokens, temperature,
                     budget, "ape_gen", "ape")
        candidates.append(ensure_prompt_contract(instr, seed_fallback=seed_prompt))
    if seed_prompt:
        candidates.append(seed_prompt)
    return _dedup(candidates)


def generate_candidates_with_unlabeled(train: LabeledSet, unlabeled_texts, llm, *, k_prompts=6, n_shots=36,
                                       n_unlabeled=12, max_tokens=100, temperature=0.0, seed=42,
                                       seed_prompt: Optional[str] = None, budget=None) -> List[str]:
    rng = random.Random(int(seed))
    n = min(int(n_shots), len(train.texts))
    order = list(range(len(train.texts)))
    rng.shuffle(order)
    order = order[:n]
    texts, labels = [train.texts[i] for i in order], [train.labels[i] for i in order]
    u = list(unlabeled_texts)
    rng.shuffle(u)
    u = u[: max(0, int(n_unlabeled))]
    demos_per = max(1, n // k_prompts)
    u_per = max(0, len(u) // max(1, k_prompts))
    candidates: List[str] = []
    for k in range(k_prompts):
        ct, cy = texts[k * demos_per:(k + 1) * demos_per], labels[k * demos_per:(k + 1) * demos_per]
        if not ct:
            ct, cy = texts[:demos_per], labels[:demos_per]
        cu = u[k * u_per:(k + 1) * u_per] if u_per else u[:2]
        instr = _ask(llm, META_PROMPT.replace("[full_DEMO]", format_demos_with_unlabeled(ct, cy, cu)), max_tokens,
                     temperature, budget, "ape_ut_gen", "ape_ut")
        candidates.append(ensure_prompt_contract(instr, seed_fallback=seed_prompt))
    if seed_prompt:
        candidates.append(seed_prompt)
    return _dedup(candidates)


def select_best_on_dev(candidates: Sequence[str], task: Task, *, note: str = "ape_dev") -> Tuple[str, float, List[dict]]:
    """The candidate with the best selection statistic on the validation set."""
    best_prompt, best_score, scored = candidates[0], -1e9, []
    for i, p in enumerate(candidates):
        print(f"[ape] score candidate {i + 1}/{len(candidates)} on |dev|={len(task.dev.texts)}", flush=True)
        m = score_prompt_on_set(task.scorer, p, task.dev)
        task.budget.charge("val", n_calls=len(task.dev.texts), kind="scorer", note=note)
        s = float(m.get("R_soft_min_gba", m.get("R_global", m.get("fitness", 0.0))))
        scored.append({"score": s, "softmin": m.get("R_soft_min_gba"), "R_global": m.get("R_global"),
                       "worst_gba": m.get("R_worst_gba"), "prompt_len": len(p)})
        if s > best_score:
            best_score, best_prompt = s, p
    return best_prompt, best_score, scored


class APE:
    def __init__(self, *, k_prompts: int = 6, n_shots: int = 36, max_tokens: int = 100, temperature: float = 0.0,
                 n_unlabeled: Optional[int] = None):
        """`n_unlabeled` set: the APE-ut variant, which also shows that many unlabelled target texts."""
        self.k_prompts, self.n_shots, self.max_tokens, self.temperature = k_prompts, n_shots, max_tokens, temperature
        self.n_unlabeled = n_unlabeled

    def run(self, task: Task) -> OptimizerResult:
        seed = int(task.meta.get("seed", 42))
        if self.n_unlabeled is None:
            cands = generate_candidates(task.train, task.optimizer_llm, k_prompts=self.k_prompts,
                                        n_shots=self.n_shots, max_tokens=self.max_tokens,
                                        temperature=self.temperature, seed=seed, seed_prompt=task.seed_prompt,
                                        budget=task.budget)
            name = "ape"
        else:
            cands = generate_candidates_with_unlabeled(
                task.train, list(task.unlabeled.texts), task.optimizer_llm, k_prompts=self.k_prompts,
                n_shots=self.n_shots, n_unlabeled=self.n_unlabeled, max_tokens=self.max_tokens,
                temperature=self.temperature, seed=seed, seed_prompt=task.seed_prompt, budget=task.budget)
            name = "ape_ut"
        best, best_score, scored = select_best_on_dev(cands, task, note=f"{name}_dev")
        trace = {"method": name, "k": self.k_prompts, "n_shots": self.n_shots, "temperature": self.temperature,
                 "scored": scored, "best_softmin": best_score}
        if self.n_unlabeled is not None:
            trace["n_unlabeled"] = self.n_unlabeled
        return OptimizerResult(best_prompt=best, all_candidates=list(cands), trace=trace,
                               scorer_calls=task.budget.scorer_calls, optimizer_calls=task.budget.optimizer_calls)


def save_result(result: OptimizerResult, path) -> None:
    path.write_text(json.dumps({"best_prompt": result.best_prompt, "n_candidates": len(result.all_candidates),
                                "trace": result.trace, "scorer_calls": result.scorer_calls,
                                "optimizer_calls": result.optimizer_calls}, indent=2), encoding="utf-8")

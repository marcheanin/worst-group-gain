"""Run optimizers: the CivilComments matrix of separate runs, and the live loops with a chosen selection statistic.

    python -m optimizers.run matrix --seed 42 --methods ape,gpo --mock          # offline test, no key, no cost
    python -m optimizers.run matrix --seed 42 --methods ape,apo,gpo,evoprompt_ga,evoprompt_de,gepa
    python -m optimizers.run live --benchmark mnli --methods ape,gepa --protocols hard_min --seeds 42 --dry-run
    python -m optimizers.run live --benchmark toxlang --methods ape,apo,evoprompt_de,gepa --protocols hard_min

The optimizer LLM is DeepSeek-V4-Pro and the task model Gemma 3 12B, both through OpenRouter; a real run costs money
(the whole project cost $89.82, Appendix L) and takes 20-40 minutes per run. Use `--dry-run` to see the plan, `--mock`
to run end to end without any query, and `--max-spend` to cap a run. Results go to `runs/` and never overwrite the
released artifacts in `data/`.

matrix  the CivilComments pool of Section 3: methods
        ape, ape_k48 (48 proposals), ape_ut (unlabelled target texts), apo (ProTeGi), oracle (ProTeGi with all labels of
        the target pool, capped at 800), random_al (ProTeGi with 240 random target labels), gpo, evoprompt_ga,
        evoprompt_de, gepa. The budget of a method depends on the seed as in Appendix D (`BUDGETS`).
live    APE, EvoPrompt-DE, GEPA (and ProTeGi on toxicity) with the selection statistic of the run: soft_min (the
        smoothed minimum of the shrunk group accuracies), hard_min (their minimum), mean_gba (the mean of the raw
        balanced accuracies) or global (plain accuracy). Every prompt the loop evaluates is archived with all its
        scores, so another statistic can be replayed on the archive. A run is finished when `done.json` exists;
        an unfinished run is discarded and restarted. A file `STOP` in the output directory stops after the current run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from optimizers import data as odata
from optimizers import fitness, profile
from optimizers.ape import APE, save_result
from optimizers.api import Budget, Task, UnlabeledSet
from optimizers.evoprompt import EvoPrompt
from optimizers.gepa import GEPA
from optimizers.gpo import GPO
from optimizers.llm import Completer, MockCompleter, MockScorer, Scorer
from optimizers.protegi import ProTeGi, WithTargetLabels
from wgg import benchmarks
from wgg.benchmarks import ROOT
from wgg.models import key_usage, load_key

# Budget of each method on each seed (Appendix D): ProTeGi rounds, EvoPrompt population x generations, GEPA reflections.
BUDGETS = {
    42: dict(apo_rounds=6, random_rounds=4, oracle_rounds=2, evo_pop=6, evo_gen=5, gepa=12),
    43: dict(apo_rounds=2, random_rounds=2, oracle_rounds=2, evo_pop=4, evo_gen=3, gepa=8),
    44: dict(apo_rounds=2, random_rounds=2, oracle_rounds=2, evo_pop=4, evo_gen=3, gepa=8),
}
LIVE = dict(evo_pop=4, evo_gen=3, gepa=8, apo_rounds=2)
MATRIX_METHODS = ["ape", "ape_k48", "ape_ut", "apo", "gpo", "evoprompt_ga", "evoprompt_de", "gepa", "random_al", "oracle"]
# protocol -> the key of the fitness record the loop selects by (None: the default, the softmin)
PROTOCOLS = {"soft_min": None, "hard_min": "R_worst_gba", "mean_gba": "R_gba_mean", "global": "R_global"}
LOGGED_KEYS = ("R_soft_min_gba", "R_worst_gba", "R_gba_mean", "R_global", "R_worst_group", "toxic_recall",
               "invalid_rate", "pred_pos_rate")


class ArchiveTap:
    """Wraps `fitness.evaluate`: writes every evaluation to an archive and swaps in the selection statistic."""

    def __init__(self, protocol: str):
        self.key, self.records, self._orig = PROTOCOLS[protocol], {}, None

    def install(self):
        self._orig, tap = fitness.evaluate, self

        def wrapped(preds, labels, groups, prompt):
            fit = tap._orig(preds, labels, groups, prompt)
            rejected = fit.get("reject_reason")
            used = fit.get("R_soft_min_gba", fit.get("fitness"))
            if not rejected and tap.key and tap.key in fit:
                used = float(fit[tap.key])
                fit["R_soft_min_gba"] = used
                fit["fitness"] = fit["combined_score"] = used
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            rec = tap.records.setdefault(h, {"hash": h, "prompt": prompt, "evals": []})
            ev = {"rows": int(len(preds)), "selected_on": None if used is None else round(float(used), 5),
                  "rejected": rejected}
            for k in LOGGED_KEYS:
                if k in fit and fit[k] is not None:
                    try:
                        ev[k] = round(float(fit[k]), 5)
                    except (TypeError, ValueError):
                        pass
            rec["evals"].append(ev)
            return fit

        fitness.evaluate = wrapped

    def uninstall(self):
        if self._orig is not None:
            fitness.evaluate = self._orig


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_models(args, seed_prompt: str):
    if args.mock:
        return MockScorer(), (lambda temperature: MockCompleter(seed_prompt)), None
    key = load_key(args.key_var)
    scorer = Scorer(key, max_parallel=args.max_parallel)
    return scorer, (lambda temperature: Completer(key, temperature=temperature)), key


def run_matrix(args):
    b = benchmarks.load("civil")
    seed = args.seed
    bud = BUDGETS[seed]
    train, unlabeled, dev, reveal = odata.civil(b, seed)
    sp = odata.seed_prompt(b)
    scorer, make_llm, _ = make_models(args, sp)
    out = Path(args.out) / "civil" / "matrix" / f"seed{seed}"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if seed != 42 and "oracle" in methods:
        print("skip oracle: run on seed 42 only", flush=True)
        methods.remove("oracle")
    plan = {
        "ape": (lambda: APE(k_prompts=6, n_shots=36), 0.0, False),
        "ape_k48": (lambda: APE(k_prompts=48, n_shots=288), 0.0, False),
        "ape_ut": (lambda: APE(k_prompts=6, n_shots=36, n_unlabeled=12), 0.0, True),
        "gpo": (lambda: GPO(k_prompts=6, n_shots=36, conf_threshold=0.83), 0.0, True),
        "apo": (lambda: ProTeGi(rounds=bud["apo_rounds"], beam_size=4), 0.7, False),
        "evoprompt_ga": (lambda: EvoPrompt(mode="ga", population_size=bud["evo_pop"], generations=bud["evo_gen"]), 0.8, False),
        "evoprompt_de": (lambda: EvoPrompt(mode="de", population_size=4, generations=3), 0.8, False),
        "gepa": (lambda: GEPA(max_metric_calls=20_000, reflections=bud["gepa"]), 0.7, False),
        "random_al": (lambda: WithTargetLabels(reveal, mode="random", label_budget=240, rounds=bud["random_rounds"]), 0.7, True),
        "oracle": (lambda: WithTargetLabels(reveal, mode="all", max_target_labels=800, rounds=bud["oracle_rounds"]), 0.7, True),
    }
    if args.dry_run:
        for m in methods:
            print(f"[plan] seed {seed} {m}: {'done' if (out / m / 'best_prompt.txt').is_file() else 'to run'}")
        return
    for m in methods:
        if m not in plan:
            print(f"unknown method {m}", flush=True)
            continue
        mdir = out / m
        if (mdir / "best_prompt.txt").is_file():
            print(f"=== seed {seed} {m}: done, skipping ===", flush=True)
            continue
        mdir.mkdir(parents=True, exist_ok=True)
        make, temp, wants_pool = plan[m]
        print(f"=== matrix: seed {seed} method {m} ===", flush=True)
        task = Task(train=train, dev=dev, unlabeled=unlabeled if wants_pool else UnlabeledSet(texts=[]), scorer=scorer,
                    optimizer_llm=make_llm(temp), budget=Budget(), seed_prompt=sp, meta={"seed": seed})
        result = make().run(task)
        save_result(result, mdir / "optimizer_result.json")
        (mdir / "best_prompt.txt").write_text(result.best_prompt, encoding="utf-8")


def run_live(args):
    b = benchmarks.load(args.benchmark)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for p in protocols:
        if p not in PROTOCOLS:
            raise SystemExit(f"unknown statistic {p!r}; choose from {sorted(PROTOCOLS)}")
    profile.set_active(profile.MNLI if b.key == "mnli" else profile.CIVIL)
    sp = odata.seed_prompt(b)
    out = Path(args.out) / b.key / "live"
    plan = [(s, m, p) for s in seeds for m in methods for p in protocols]
    todo = [t for t in plan if not (out / f"seed{t[0]}" / f"{t[1]}__{t[2]}" / "done.json").is_file()]
    print(f"[plan] {len(plan)} runs, {len(plan) - len(todo)} done, {len(todo)} to run", flush=True)
    if args.dry_run or not todo:
        return
    scorer, make_llm, key = make_models(args, sp)
    out.mkdir(parents=True, exist_ok=True)
    start = key_usage(key) if key else None
    spent, trains = 0.0, {}
    for idx, (seed, method, protocol) in enumerate(todo, 1):
        if (out / "STOP").exists():
            print("[pause] STOP found; everything finished is saved", flush=True)
            return
        if key:
            now = key_usage(key)
            if now is not None and start is not None:
                spent = now - start
            if spent > args.max_spend:
                print(f"[stop] spend ${spent:.2f} exceeds the ceiling ${args.max_spend:.2f}", flush=True)
                return
        rundir = out / f"seed{seed}" / f"{method}__{protocol}"
        if rundir.exists():
            shutil.rmtree(rundir)
        rundir.mkdir(parents=True)
        print(f"\n=== [{idx}/{len(todo)}] seed {seed} {method} {protocol} (spent ${spent:.2f}) ===", flush=True)
        if seed not in trains:
            if b.key == "civil":
                tr, _, dv, _ = odata.civil(b, seed)
                trains[seed] = (tr, dv)
            else:
                trains[seed] = odata.live_loop(b, seed)
        train, dev = trains[seed]
        task_args = dict(train=train, dev=dev, unlabeled=UnlabeledSet(texts=[]), scorer=scorer, budget=Budget(),
                         seed_prompt=sp, meta={"seed": seed})
        makers = {
            "ape": (lambda: APE(k_prompts=6, n_shots=36), 0.0),
            "evoprompt_de": (lambda: EvoPrompt(mode="de", population_size=LIVE["evo_pop"], generations=LIVE["evo_gen"]), 0.8),
            "evoprompt_ga": (lambda: EvoPrompt(mode="ga", population_size=LIVE["evo_pop"], generations=LIVE["evo_gen"]), 0.8),
            "gepa": (lambda: GEPA(max_metric_calls=20_000, reflections=LIVE["gepa"]), 0.7),
            "apo": (lambda: ProTeGi(rounds=LIVE["apo_rounds"], beam_size=4), 0.7),
        }
        if method not in makers:
            raise SystemExit(f"method {method!r} is not a live-loop method")
        make, temp = makers[method]
        tap = ArchiveTap(protocol)
        tap.install()
        t0 = time.time()
        try:
            result = make().run(Task(optimizer_llm=make_llm(temp), **task_args))
        finally:
            tap.uninstall()
        atomic_write(rundir / "best_prompt.txt", result.best_prompt)
        atomic_write(rundir / "archive.json", json.dumps(
            {"protocol": protocol, "method": method, "seed": seed, "all_candidates": list(result.all_candidates or []),
             "evaluated": list(tap.records.values())}, ensure_ascii=False))
        atomic_write(rundir / "optimizer_result.json", json.dumps(
            {"n_candidates": len(result.all_candidates or []), "trace": result.trace, "scorer_calls": result.scorer_calls,
             "optimizer_calls": result.optimizer_calls}, ensure_ascii=False, indent=2, default=str))
        atomic_write(rundir / "done.json", json.dumps(
            {"seconds": round(time.time() - t0, 1), "unique_prompts_evaluated": len(tap.records),
             "scorer_calls": result.scorer_calls, "optimizer_calls": result.optimizer_calls,
             "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        print(f"    done in {(time.time() - t0) / 60:.1f} min: {len(tap.records)} distinct prompts evaluated", flush=True)
    print("\n[done] the whole plan is finished", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["matrix", "live"])
    ap.add_argument("--benchmark", default="civil", choices=list(benchmarks.KEYS))
    ap.add_argument("--seed", type=int, default=42, help="matrix: seed of the data draw and of the optimizer")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--protocols", default="soft_min,hard_min,global", help="live: selection statistics")
    ap.add_argument("--seeds", default="42,43,44", help="live: seeds")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--max-parallel", type=int, default=4, help="parallel queries to the task model (more gives 429 errors)")
    ap.add_argument("--max-spend", type=float, default=25.0, help="live: USD ceiling of this run")
    ap.add_argument("--key-var", default=None)
    ap.add_argument("--mock", action="store_true", help="no queries: a stand-in task model and optimizer LLM")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.mode == "matrix":
        args.methods = args.methods or ",".join(MATRIX_METHODS)
        run_matrix(args)
    else:
        args.methods = args.methods or ("ape,apo,evoprompt_de,gepa" if args.benchmark == "toxlang" else "ape,evoprompt_de,gepa")
        run_live(args)


if __name__ == "__main__":
    main()

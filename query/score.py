"""Score a pool of prompts on the test (and validation) set with a task model; stoppable at any moment.

    python -m query.score --benchmark toxlang --model gpt-4o-mini --prompts seed,s15:42_gepa__hard_min
    python -m query.score --benchmark mnli --model qwen3-235b --dry-run

Predictions go to `data/<benchmark>/predictions/<model>/<set>/<prompt>.npy` (int16, -1 = unparsable) and, for
models that return log-probabilities, to a companion `.lo.npy`. This costs money: use `--dry-run` first to see
how many queries are left, and `--max-spend` to cap the cost of one run.

How it stops and resumes. The plan is a list of steps `<set>:<length>`; each step brings every prompt up to the
first `length` rows of the set. Rows are shuffled, so every prefix is a stratified subsample and the matrix is
rectangular up to the length of the slowest prompt after an interruption. The state is the files themselves: the
length of `<prompt>.npy` is how many rows are done. Writes are atomic (temporary file and `os.replace`), and log-odds
are written before labels. A file `STOP` in the output directory (or Ctrl-C) makes the run finish the current block
and exit; running the same command again continues. A balance check every minute stops the run when the spend of
this run exceeds `--max-spend`.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from wgg import benchmarks
from wgg.data import load_texts
from wgg.models import MODELS, Worker, key_usage, load_key

_stop = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    print(f"[pause] signal {signum}: finishing the current block and exiting", flush=True)


def atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def load_state(pdir: Path, stem: str):
    f, g = pdir / f"{stem}.npy", pdir / f"{stem}.lo.npy"
    pred = np.load(f) if f.is_file() else np.zeros(0, dtype=np.int16)
    lo = np.load(g) if g.is_file() else np.zeros(0, dtype=np.float32)
    lo = lo[: len(pred)] if len(lo) >= len(pred) else np.concatenate([lo, np.full(len(pred) - len(lo), np.nan, dtype=np.float32)])
    return pred, lo


def parse_plan(text: str):
    return [(n.strip(), int(k)) for n, _, k in (item.partition(":") for item in text.split(","))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--prompts", default=None, help="comma-separated prompt names; default: every prompt of the benchmark")
    ap.add_argument("--plan", default="test:900,test:1800,test:5000", help="steps <set>:<length>; test or dev")
    ap.add_argument("--max-parallel", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=120, help="rows per block: what is lost at a hard stop")
    ap.add_argument("--max-spend", type=float, default=5.0, help="USD ceiling for this run")
    ap.add_argument("--key-var", default=None, help="environment variable holding the key")
    ap.add_argument("--dry-run", action="store_true", help="count the queries left and exit without querying")
    args = ap.parse_args()

    b = benchmarks.load(args.benchmark)
    spec = MODELS[args.model]
    pool = b.prompts()
    names = [n.strip() for n in args.prompts.split(",")] if args.prompts else list(pool)
    plan = parse_plan(args.plan)
    root = b.data / "predictions" / args.model
    texts = {s: load_texts(b, s) for s in {s for s, _ in plan}}
    targets = {}
    for s, k in plan:
        targets[s] = max(targets.get(s, 0), min(k, len(texts[s])))
    remaining = sum(max(0, targets[s] - len(load_state(root / s, n.replace(":", "__", 1))[0]))
                    for s in targets for n in names)
    print(f"[plan] {args.plan}: {len(names)} prompts, {remaining} queries left", flush=True)
    if args.dry_run:
        return

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)
    key = load_key(args.key_var)
    worker = Worker(spec["id"], key, spec["provider"], spec["logprobs"])
    stop_file = root / "STOP"
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(
        {"model": spec["id"], "provider": spec["provider"], "max_tokens": 16, "temperature": 0.0,
         "logprobs": spec["logprobs"], "plan": args.plan}, indent=2), encoding="utf-8")
    start, last_check, spent = key_usage(key), time.time(), 0.0
    t_start, done_total = time.time(), 0

    for step, (s, length) in enumerate(plan, 1):
        target = min(length, len(texts[s]))
        pdir = root / s
        pdir.mkdir(parents=True, exist_ok=True)
        print(f"[step {step}/{len(plan)}] {s} up to {target} rows", flush=True)
        for name in names:
            stem = name.replace(":", "__", 1)
            pred, lo = load_state(pdir, stem)
            while len(pred) < target:
                if _stop or stop_file.exists():
                    print("[pause] stopped on request; run the same command to continue", flush=True)
                    return
                if time.time() - last_check > 60:
                    now = key_usage(key)
                    if now is not None and start is not None:
                        spent = now - start
                    last_check = time.time()
                    if spent > args.max_spend:
                        print(f"[stop] spend ${spent:.2f} exceeds the ceiling; exiting cleanly", flush=True)
                        return
                a, c = len(pred), min(len(pred) + args.chunk, target)
                p, l = worker.label_rows(pool[name], texts[s][a:c], args.max_parallel)
                pred, lo = np.concatenate([pred, p]), np.concatenate([lo, l])
                atomic_save(pdir / f"{stem}.lo.npy", lo)
                atomic_save(pdir / f"{stem}.npy", pred)
                done_total += c - a
                print(f"  {name:30s} {len(pred):5d}/{target}  {done_total / max(time.time() - t_start, 1e-6):4.1f}/s  "
                      f"unparsable {int((p < 0).sum())}  retries {worker.retries}  spent ${spent:.2f}", flush=True)
    print(f"[done] spent ${spent:.2f}; providers {dict(worker.providers)}", flush=True)


if __name__ == "__main__":
    main()

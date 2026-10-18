"""Recompute every table and figure of the paper from the saved predictions. No API key and no queries needed.

    python reproduce.py              # everything (about two hours, most of it the split-half replication)
    python reproduce.py --quick      # without the slowest analyses (replication, 100,000-resample contrasts, protocols)
    python reproduce.py --only target_group --only figures
    python reproduce.py --list
    WGG_RESULTS=/tmp/results python reproduce.py --quick     # write somewhere else, e.g. to compare with results/

Each job is one command of the `analysis` package; its output goes to `results/<benchmark>/` (see the table in
README.md for which job feeds which table or figure).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

BENCH = ("civil", "mnli", "toxlang")


def per_bench(name, module, extra=(), benches=BENCH, slow=False):
    return [(f"{name}[{b}]", ["-m", module, *extra, "--benchmark", b], slow) for b in benches]


JOBS = []
JOBS += per_bench("compare_with_seed", "analysis.compare_with_seed")
JOBS += [(f"chance_baseline[{b},{pool}]", ["-m", "analysis.chance_baseline", "--benchmark", b, "--pool", pool], False)
         for b in BENCH for pool in ("all", "finals")]
JOBS += per_bench("target_group", "analysis.target_group")
JOBS += per_bench("target_group_100k", "analysis.target_group", ("--high-resolution",), slow=True)
JOBS += per_bench("argmin", "analysis.argmin")
JOBS += per_bench("replication", "analysis.replication", slow=True)
JOBS += [("random_groups", ["-m", "analysis.regroup", "random", "--benchmark", b, "--k", str(k), "--reps", str(r), "--pool", pool], True)
         for b, ks, pool in (("civil", ((2, 50), (4, 50), (8, 200)), "all"), ("mnli", ((2, 50), (4, 50), (10, 200)), "finals"),
                             ("toxlang", ((2, 50), (3, 50), (6, 200)), "all")) for k, r in ks]
JOBS += [(f"meaningful_groups[{b}]", ["-m", "analysis.regroup", "meaningful", "--benchmark", b], False) for b in BENCH]
JOBS += [("meaningful_groups[civil,worst_class]", ["-m", "analysis.regroup", "meaningful", "--benchmark", "civil", "--metric", "worst_class"], False),
         ("meaningful_groups[civil,global_acc]", ["-m", "analysis.regroup", "meaningful", "--benchmark", "civil", "--metric", "global_acc"], False),
         ("regroup_features[mnli]", ["-m", "analysis.regroup", "features", "--benchmark", "mnli"], False),
         ("cell_size[mnli]", ["-m", "analysis.regroup", "cell-size", "--benchmark", "mnli"], False)]
JOBS += [(f"protocols_{part}[{b},{t}]", ["-m", "analysis.protocols", part, "--benchmark", b, "--target", t], True)
         for b in ("civil", "mnli") for t in ("hard_min", "cvar25") for part in ("run", "decompose")]
JOBS += [("pool_composition[civil]", ["-m", "analysis.protocols", "composition", "--benchmark", "civil"], True)]
JOBS += [(f"test_draw_halves[{b}]", ["-m", "analysis.test_draw", "halves", "--benchmark", b], False) for b in ("civil", "mnli")]
JOBS += [("test_draw_split_half[civil]", ["-m", "analysis.test_draw", "split-half", "--benchmark", "civil"], False),
         ("wilds_metric", ["-m", "analysis.wilds_metric"], False),
         ("optimizer_selection", ["-m", "analysis.optimizer_selection"], False),
         ("optimizer_selection[unparsable as wrong]", ["-m", "analysis.optimizer_selection", "--invalid-as-error"], False)]
JOBS += [(f"transfer_gate[{m}]", ["-m", "analysis.transfer", "gate", "--model", m], False) for m in ("qwen3-235b", "qwen3-30b")]
JOBS += [(f"transfer_mnli[{m}]", ["-m", "analysis.transfer", "mnli", "--model", m], False) for m in ("gpt-4o-mini", "qwen3-235b", "qwen3-30b")]
JOBS += [(f"transfer_toxicity[{m}]", ["-m", "analysis.transfer", "toxicity", "--model", m], False) for m in ("gpt-4o-mini", "qwen3-235b")]
JOBS += [("second_model_table", ["-m", "analysis.second_model", "table"], False),
         ("second_model_reliability", ["-m", "analysis.second_model", "reliability"], False),
         ("chance_baseline[civil,gpt-4o-mini,all]", ["-m", "analysis.chance_baseline", "--benchmark", "civil", "--model", "gpt-4o-mini"], False),
         ("chance_baseline[civil,gpt-4o-mini,finals]", ["-m", "analysis.chance_baseline", "--benchmark", "civil", "--model", "gpt-4o-mini", "--pool", "finals"], False),
         ("argmin[civil,gpt-4o-mini]", ["-m", "analysis.argmin", "--benchmark", "civil", "--model", "gpt-4o-mini"], False),
         ("replication[civil,gpt-4o-mini]", ["-m", "analysis.replication", "--benchmark", "civil", "--model", "gpt-4o-mini"], True)]
JOBS += per_bench("predictability", "analysis.predictability")
JOBS += per_bench("live_loops", "analysis.live_loops")
JOBS += per_bench("dropped_rows", "analysis.dropped_rows")
JOBS += [("figures", ["-m", "analysis.figures"], False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the slow jobs")
    ap.add_argument("--only", action="append", default=[], help="run the jobs whose name contains this (repeatable)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    jobs = [j for j in JOBS if not (args.quick and j[2]) and (not args.only or any(o in j[0] for o in args.only))]
    if args.list:
        for name, argv, slow in jobs:
            print(f"{'slow ' if slow else '     '}{name:48s} python {' '.join(argv)}")
        return
    t0, failed = time.time(), []
    for i, (name, argv, _) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {name}", flush=True)
        t = time.time()
        rc = subprocess.call([sys.executable, *argv])
        print(f"    {'ok' if rc == 0 else 'FAILED'} in {time.time() - t:.0f}s", flush=True)
        if rc:
            failed.append(name)
    print(f"\n{len(jobs) - len(failed)} of {len(jobs)} jobs finished in {(time.time() - t0) / 60:.1f} min"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

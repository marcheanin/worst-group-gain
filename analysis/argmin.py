"""Which group attains the minimum, how stable that choice is, and how far the minimum is biased down.

Feeds the lower part of Table 3 and Appendix E. For every prompt the per-group balanced accuracies
are resampled within group x label cells (4,000 resamples, seed 2, the same draws for all prompts).
`stay` is the share of resamples in which the observed worst group is still the worst; `bias` is the
observed minimum minus the mean minimum over resamples (reported with the sign flipped as the
downward bias of the minimum: a minimum over noisy estimates falls below the true minimum).
Writes `results/<benchmark>/argmin.json` with medians over the prompts and per-prompt values.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from wgg import benchmarks
from wgg.benchmarks import MAIN_MODEL
from wgg.stats import load_preds, load_set, valid_rows


def run(b, n_boot=4000, model=MAIN_MODEL):
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y), model=model)
    names = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), names)
    ids = b.ids
    gnames = b.group_names()

    gvals = {n: np.array([0.5 * ((P[n][(c == g) & (y == 1) & keep] == 1).mean()
                                 + (P[n][(c == g) & (y == 0) & keep] == 0).mean()) for g in ids]) for n in names}
    rng = np.random.default_rng(2)
    cells = [(np.flatnonzero((c == g) & (y == 1) & keep), np.flatnonzero((c == g) & (y == 0) & keep)) for g in ids]
    draws = [(rng.integers(0, len(p), size=(n_boot, len(p))), rng.integers(0, len(q), size=(n_boot, len(q))))
             for p, q in cells]
    stay, bias, winners, gaps = [], [], [], []
    for n in names:
        G = np.empty((n_boot, len(ids)))
        for k, ((dp, dq), (p, q)) in enumerate(zip(draws, cells)):
            G[:, k] = 0.5 * ((P[n][p][dp] == 1).mean(axis=1) + (P[n][q][dq] == 0).mean(axis=1))
        obs_am = int(gvals[n].argmin())
        stay.append(float((G.argmin(axis=1) == obs_am).mean()))
        bias.append(float(gvals[n].min() - G.min(axis=1).mean()))
        winners.append(obs_am)
        gs = np.sort(gvals[n])
        gaps.append(float(gs[1] - gs[0]))
    stay, bias = np.array(stay), np.array(bias)
    cnt = Counter(ids[w] for w in winners)
    print(f"{b.title}, {model}: {len(names)} prompts; the observed worst group stays the worst in "
          f"{100 * np.median(stay):.0f}% of resamples (median; range {100 * stay.min():.0f}-{100 * stay.max():.0f}%), "
          f"seed {100 * stay[names.index('seed')]:.0f}%")
    print(f"  bias of the minimum: median {np.median(bias):+.4f}; groups as minimum: "
          + ", ".join(f"{gnames.get(g, g)} {k}" for g, k in cnt.most_common()) + f" ({len(cnt)} of {len(ids)})")

    res = {"argmin_stay_median": float(np.median(stay)), "argmin_bias_median": float(np.median(bias)),
           "argmin_groups": {str(g): k for g, k in cnt.items()}, "n_boot": n_boot, "n_prompts": len(names),
           "argmin_stay_min": float(stay.min()), "argmin_stay_max": float(stay.max()),
           "argmin_bias_min": float(bias.min()), "argmin_bias_max": float(bias.max()),
           "per_prompt": {n: {"argmin_group": str(ids[w]), "stay": float(st), "bias": float(bi),
                              "gap_second_minus_min": gp}
                          for n, w, st, bi, gp in zip(names, winners, stay, bias, gaps)}}
    out = b.out_dir(model) / "argmin.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    ap.add_argument("--model", default=MAIN_MODEL)
    args = ap.parse_args()
    run(benchmarks.load(args.benchmark), model=args.model)


if __name__ == "__main__":
    main()

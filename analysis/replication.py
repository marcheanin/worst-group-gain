"""Do the differences a statistic declares on one half of the test replicate on the other half?

Feeds the upper part of Table 3 and the "Replicability" paragraph of Appendix E. Each test set is
split into two halves within every group x label cell, 100 times. On half A every pair among the
seed and the optimizer finals is compared with a paired bootstrap (300 resamples); a difference is
declared when its 95% interval excludes zero, and it replicates when the observed difference on half
B has the same sign. A reversal is a declared difference that is significant with the opposite sign
on half B. All statistics are scored on the same splits and resamples, so they are compared paired.

Statistics: the minimum (hard_min), CVaR@25% (cvar25), the softmin with shrinkage weight kappa from
0 to 320 (softmin_w<kappa>: each group's balanced accuracy is mixed with the mean over groups as if
with kappa extra examples, then -0.1 log mean exp(-x / 0.1) is taken), the mean over groups and
worst-class accuracy. The faithfulness part rank-correlates each statistic with the minimum and with
the mean over the prompts on the full test set.

Writes `results/<benchmark>/statistic_reproducibility_n100.json` and `statistic_faithfulness.json`.
The replication part is slow (tens of minutes per benchmark).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from wgg import benchmarks
from wgg.benchmarks import MAIN_MODEL
from wgg.stats import load_preds, load_set, valid_rows

WEIGHTS = [0, 10, 20, 40, 80, 160, 320]
STATS = ["hard_min", "cvar25"] + [f"softmin_w{w}" for w in WEIGHTS] + ["mean_gba", "worst_class"]
N_BOOT = 300


def all_stats(pred, pos, neg):
    """Every statistic from one pass over the group rates, so that they share the resample."""
    g, n = [], []
    for p, q in zip(pos, neg):
        if len(p) and len(q):
            g.append(0.5 * ((pred[p] == 1).mean() + (pred[q] == 0).mean()))
            n.append(len(p) + len(q))
    g, n = np.asarray(g), np.asarray(n)
    v = np.sort(g)
    pooled = g.mean()
    allp = np.concatenate(pos) if len(pos) else np.array([], int)
    alln = np.concatenate(neg) if len(neg) else np.array([], int)
    out = {"hard_min": float(v[0]), "cvar25": float(v[: max(1, len(v) // 4)].mean()), "mean_gba": float(g.mean()),
           "worst_class": float(min((pred[allp] == 1).mean(), (pred[alln] == 0).mean()))}
    for w in WEIGHTS:
        sm = (n * g + w * pooled) / (n + w)
        out[f"softmin_w{w}"] = float(-0.1 * np.log(np.mean(np.exp(-sm / 0.1))))
    return out


def cells_of(ids, y, c, rows):
    pos = [np.flatnonzero((c[rows] == g) & (y[rows] == 1)) for g in ids]
    neg = [np.flatnonzero((c[rows] == g) & (y[rows] == 0)) for g in ids]
    return pos, neg


def boot_all(ids, preds, names, rows, y, c, rng):
    """{statistic: (prompts, N_BOOT)} on one half, one shared resampling scheme."""
    flat = [np.flatnonzero((c[rows] == g) & (y[rows] == lab)) for g in ids for lab in (0, 1)]
    flat = [x for x in flat if len(x)]
    idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in flat], axis=1)
    yy, cc = y[rows][idx], c[rows][idx]
    cells = [cells_of(ids, yy[b], cc[b], slice(None)) for b in range(N_BOOT)]
    out = {s: np.empty((len(names), N_BOOT)) for s in STATS}
    for i, n in enumerate(names):
        take = preds[n][rows][idx]
        for b in range(N_BOOT):
            vals = all_stats(take[b], *cells[b])
            for s in STATS:
                out[s][i, b] = vals[s]
    return out


def load(b, model=MAIN_MODEL):
    y, c, rec = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, controls=False, model=model)
    keep = valid_rows(P, len(y), verbose=False)
    names = [n for n in sorted(P) if b.is_final(n) or n == "seed"]
    return names, {n: P[n][keep] for n in names}, y[keep], c[keep], int(rec["cap_per_cell"])


def faithfulness(b, model=MAIN_MODEL):
    names, preds, y, c, cap = load(b, model)
    pos, neg = cells_of(b.ids, y, c, slice(None))
    vals = {n: all_stats(preds[n], pos, neg) for n in names}
    hm = np.array([vals[n]["hard_min"] for n in names])
    mg = np.array([vals[n]["mean_gba"] for n in names])
    rows = {}
    print(f"{b.title}: rank correlation over {len(names)} prompts with the minimum / with the mean")
    for st in ["hard_min", "cvar25"] + [f"softmin_w{w}" for w in WEIGHTS] + ["mean_gba", "worst_class"]:
        v = np.array([vals[n][st] for n in names])
        r_min, r_mean = spearmanr(v, hm).statistic, spearmanr(v, mg).statistic
        rows[st] = {"rho_vs_hard_min": float(r_min), "rho_vs_mean_gba": float(r_mean)}
        print(f"  {st:14s} {r_min:6.3f} {r_mean:6.3f}")
    out = b.out_dir(model) / "statistic_faithfulness.json"
    out.write_text(json.dumps({"dataset": b.key, "n_prompts": len(names), "rows_per_cell_full_test": cap,
                               "prompts": names, "faithfulness": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def replication(b, n_split=100, seed=0, model=MAIN_MODEL):
    names, preds, y, c, cap = load(b, model)
    rng = np.random.default_rng(seed)
    k = len(names)
    pairs = k * (k - 1) // 2
    tally = {s: [0, 0, 0] for s in STATS}  # declared, replicated, reversed
    print(f"{b.title}: {k} prompts, {pairs} pairs, {n_split} splits at {cap // 2} rows per cell per half")
    for sp in range(n_split):
        a_rows, b_rows = [], []
        for g in b.ids:
            for lab in (0, 1):
                perm = rng.permutation(np.flatnonzero((c == g) & (y == lab)))
                a_rows.append(perm[: len(perm) // 2])
                b_rows.append(perm[len(perm) // 2:])
        A, B = np.concatenate(a_rows), np.concatenate(b_rows)
        MA = boot_all(b.ids, preds, names, A, y, c, rng)
        MB = boot_all(b.ids, preds, names, B, y, c, rng)
        pos, neg = cells_of(b.ids, y, c, B)
        ob = {n: all_stats(preds[n][B], pos, neg) for n in names}
        for s in STATS:
            for i in range(k):
                for j in range(i + 1, k):
                    lo, hi = np.percentile(MA[s][i] - MA[s][j], [2.5, 97.5])
                    if not (lo > 0 or hi < 0):
                        continue
                    sign = 1.0 if lo > 0 else -1.0
                    tally[s][0] += 1
                    tally[s][1] += int(np.sign(ob[names[i]][s] - ob[names[j]][s]) == sign)
                    blo, bhi = np.percentile(MB[s][i] - MB[s][j], [2.5, 97.5])
                    if (blo > 0 or bhi < 0) and (1.0 if blo > 0 else -1.0) != sign:
                        tally[s][2] += 1
        if (sp + 1) % 10 == 0:
            print(f"  {sp + 1} splits done", flush=True)

    rows = {}
    print(f"  {'statistic':13s} {'declared':>9s} {'share':>7s} {'replicated':>11s} {'reversed':>9s}")
    for s in STATS:
        d, r, f = tally[s]
        share, rep = 100 * d / (pairs * n_split), (100 * r / d if d else None)
        rows[s] = {"declared": d, "declared_share_pct": share, "reproduced": r, "reproduced_share_pct": rep,
                   "sign_reversals": f}
        print(f"  {s:13s} {d:9d} {share:6.1f}% {rep if rep is None else round(rep, 1):>10}% {f:9d}")
    out = b.out_dir(model) / "statistic_reproducibility_n100.json"
    out.write_text(json.dumps({"dataset": b.key, "prompts": names, "n_prompts": k, "n_pairs": pairs,
                               "n_splits": n_split, "rows_per_cell_per_half": cap // 2, "n_boot": N_BOOT,
                               "seed": seed, "statistics": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run(b, model=MAIN_MODEL):
    faithfulness(b, model)
    replication(b, model=model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    ap.add_argument("--faithfulness-only", action="store_true")
    ap.add_argument("--model", default=MAIN_MODEL)
    args = ap.parse_args()
    b = benchmarks.load(args.benchmark)
    faithfulness(b, args.model) if args.faithfulness_only else run(b, args.model)


if __name__ == "__main__":
    main()

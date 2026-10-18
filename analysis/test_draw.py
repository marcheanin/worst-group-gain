"""How much the draw of the test set decides what is resolvable, at a fixed size (Figure 5, Appendix G).

`halves`: each test set is cut 12 times into a random half (half of every group x label cell), and on
every half all pairs of prompts are compared by CVaR@25% with a paired bootstrap of 400 resamples.
The mean interval width barely moves between halves; the share of pairs whose interval excludes zero
moves a lot (preregistered prediction H4). On CivilComments the pool is the seed and the finals; the
published halves were drawn after a 16-cell bootstrap of 100 rows per cell had consumed the random
stream, and that draw is replayed so that half k is the same half.

`split-half` (CivilComments): 20 random splits into two halves; a pair declared distinguishable on
half A is checked on the disjoint half B, by the sign of its observed difference and by B's own
interval.

Writes `results/<benchmark>/draw_lottery_halves.json` and `results/civil/split_half_signs.json`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.stats import Bootstrap, cell_index, load_preds, load_set, valid_rows

N_BOOT = 400


def summarize(P, names, cells, y, c, groups, rng, metric="cvar25"):
    bs = Bootstrap(y, c, N_BOOT, rng, cells=cells, groups=groups)
    obs, boot = {}, {}
    for n in names:
        o, bt = bs.metrics(P[n])
        obs[n], boot[n] = o[metric], bt[metric]
    diffs, widths, res = [], [], 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo, hi = np.percentile(boot[a] - boot[b], [2.5, 97.5])
            widths.append(hi - lo)
            diffs.append(abs(obs[a] - obs[b]))
            res += int(lo > 0 or hi < 0)
    sizes = [len(v) for v in bs.cells.values()]
    return {"rows": int(sum(sizes)), "pairs": len(widths), "resolvable": res, "share": res / len(widths),
            "median_abs_d": float(np.median(diffs)), "width_mean": float(np.mean(widths)),
            "width_median": float(np.median(widths))}


def halves(b, n_halves=12, seed=0):
    y, c, rec = load_set(b, b.test_set)
    per_cell = int(rec["cap_per_cell"]) // 2
    P = load_preds(b, b.test_set, controls=False)
    names = sorted(n for n in P if b.key != "civil" or b.is_final(n) or n == "seed")
    keep = valid_rows(P, len(y), verbose=False)
    full = cell_index(y, c, keep)
    cells = {(g, lab): full[(g, lab)] for g in b.ids for lab in (0, 1) if len(full.get((g, lab), ()))}
    rng = np.random.default_rng(seed)
    if b.key == "civil":
        for _ in range(16):
            rng.choice(np.arange(100), size=(N_BOOT, 100), replace=True)
    full_set = summarize(P, names, cells, y, c, b.ids, rng)
    out = []
    for _ in range(n_halves):
        half = {k: rng.permutation(rows)[: min(per_cell, len(rows) // 2)] for k, rows in cells.items()}
        out.append(summarize(P, names, half, y, c, b.ids, rng))
    sh = np.array([s["share"] for s in out])
    wm = np.array([s["width_mean"] for s in out])
    print(f"{b.title}: {len(names)} prompts, {n_halves} halves; mean width {wm.min():.4f}-{wm.max():.4f} "
          f"(ratio {wm.max() / wm.min():.2f}), share distinguishable {100 * sh.min():.1f}-{100 * sh.max():.1f}%")
    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / "draw_lottery_halves.json").write_text(json.dumps(
        {"prompts": names, "n_boot": N_BOOT, "full_set": full_set, "halves": out}, indent=1), encoding="utf-8")


def cvar25_cells(pred, cells_pos, cells_neg):
    vals = [0.5 * ((pred[p] == 1).mean() + (pred[q] == 0).mean()) for p, q in zip(cells_pos, cells_neg)
            if len(p) and len(q)]
    v = np.sort(vals)
    return float(v[: max(1, len(v) // 4)].mean())


def split_half(b, n_split=20, seed=0):
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, controls=False)
    keep = valid_rows(P, len(y), verbose=False)
    names = [n for n in sorted(P) if b.is_final(n) or n == "seed"]
    preds, y, c, ids = {n: P[n][keep] for n in names}, y[keep], c[keep], b.ids

    def boot_metrics(rows, rng):
        cells = [x for x in (np.flatnonzero((c[rows] == g) & (y[rows] == lab)) for g in ids for lab in (0, 1)) if len(x)]
        idx = np.concatenate([rng.choice(x, size=(N_BOOT, len(x)), replace=True) for x in cells], axis=1)
        yy, cc = y[rows][idx], c[rows][idx]
        pos = [[np.flatnonzero((cc[k] == g) & (yy[k] == 1)) for g in ids] for k in range(N_BOOT)]
        neg = [[np.flatnonzero((cc[k] == g) & (yy[k] == 0)) for g in ids] for k in range(N_BOOT)]
        M = np.empty((len(names), N_BOOT))
        for i, n in enumerate(names):
            take = preds[n][rows][idx]
            for k in range(N_BOOT):
                M[i, k] = cvar25_cells(take[k], pos[k], neg[k])
        return M

    rng = np.random.default_rng(seed)
    declared = agreed = reversed_ = 0
    for _ in range(n_split):
        a_rows, b_rows = [], []
        for g in ids:
            for lab in (0, 1):
                perm = rng.permutation(np.flatnonzero((c == g) & (y == lab)))
                a_rows.append(perm[: len(perm) // 2])
                b_rows.append(perm[len(perm) // 2:])
        A, B = np.concatenate(a_rows), np.concatenate(b_rows)
        MA, MB = boot_metrics(A, rng), boot_metrics(B, rng)
        pos = [np.flatnonzero((c[B] == g) & (y[B] == 1)) for g in ids]
        neg = [np.flatnonzero((c[B] == g) & (y[B] == 0)) for g in ids]
        obs_b = np.array([cvar25_cells(preds[n][B], pos, neg) for n in names])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                lo, hi = np.percentile(MA[i] - MA[j], [2.5, 97.5])
                if not (lo > 0 or hi < 0):
                    continue
                declared += 1
                sign_a = 1.0 if lo > 0 else -1.0
                agreed += int(np.sign(obs_b[i] - obs_b[j]) == sign_a)
                blo, bhi = np.percentile(MB[i] - MB[j], [2.5, 97.5])
                reversed_ += int((blo > 0 or bhi < 0) and (1.0 if blo > 0 else -1.0) != sign_a)
    print(f"{b.title}, {n_split} splits: {declared} pairs distinguishable on half A; same sign on half B "
          f"{agreed}, opposite sign {declared - agreed}, significant with the opposite sign {reversed_}")
    (b.results / "split_half_signs.json").write_text(json.dumps(
        {"n_split": n_split, "declared": declared, "same_sign": agreed, "opposite_sign": declared - agreed,
         "significant_opposite": reversed_}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["halves", "split-half"])
    benchmarks.add_argument(ap, choices=("civil", "mnli"))
    args = ap.parse_args()
    b = benchmarks.load(args.benchmark)
    halves(b) if args.part == "halves" else split_half(b)


if __name__ == "__main__":
    main()

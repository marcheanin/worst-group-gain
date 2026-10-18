"""The same predictions under other groupings of the same test examples.

Four modes, each with the paired bootstrap of the main analysis (4,000 resamples within group x label
cells) and Holm's correction over the contrasts against the seed:

  random      k random groups of equal size within each label, repeated over many partitions
              (Figure 4). Group heterogeneity disappears; predictions, accuracy and cell balance stay.
              -> regroup_pseudo_<metric>_<k>g_<reps>rep.json
  meaningful  coarser groupings built from existing categories (Appendix G, Table of groupings with a
              meaning; Appendix M, gap between the two worst groups), with the Reality Check under each
              grouping. -> semantic_regroup_<metric>.json
  features    MultiNLI regrouped into quartiles of a property of the text fixed in advance: the share
              of hypothesis words that occur in the premise, or the hypothesis length (preregistered
              prediction "regrouping MultiNLI by difficulty", Appendix B). -> regroup_features.json
  cell-size   MultiNLI under three partitions with thinner or thicker cells: genre (A), genre x
              negation (B) and negation alone (C), with the broad negation list of the preregistration
              (prediction H6, Appendix B). -> cell_size_h6.json

A grouping must be fixed before looking at where the seed fails; a grouping built from the errors
of the seed would select its worst group by the outcome.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.stats import (Bootstrap, boot_p, cell_index, delta_ci, gba_by_group, holm, load_preds, load_set,
                       valid_rows)

SCHEMES = {
    "civil": {
        "real8": None,
        # identity families of the Jigsaw annotation: gender and orientation | religion | race
        "sem3": {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3},
        "sem2_relig": {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1},
    },
    "mnli": {
        "real10": None,
        # matched against mismatched genres, the split of the benchmark itself
        "matched2": {2: 1, 3: 1, 7: 1, 8: 1, 9: 1, 1: 2, 4: 2, 5: 2, 6: 2, 10: 2},
        # transcribed speech (facetoface, telephone) against written genres
        "spoken2": {1: 1, 8: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 9: 2, 10: 2},
        # genres paired by register; verbatim keeps its own group
        "reg5": {1: 1, 8: 1, 2: 2, 4: 2, 3: 3, 6: 3, 7: 4, 9: 4, 5: 5, 10: 5},
    },
    "toxlang": {"real6": None},
}
FEATURES = {"overlap": "lexical_overlap", "hyp_len": "hypothesis_words"}


def setup(b, pool):
    y, c, rec = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y))
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), every, verbose=False)
    names = [n for n in every if n != "seed"]
    if pool == "finals":
        names = [n for n in names if b.is_final(n)]
    return y, c, rec, P, keep, names


def contrasts(P, names, y, groups_vec, groups, metric, n_boot, rng, keep):
    """Contrast of every prompt against the seed: estimate, interval, p-value; Holm over all."""
    bs = Bootstrap(y[keep], groups_vec[keep], n_boot, rng, groups=groups)
    obs_s, boot_s = bs.metrics(P["seed"][keep])
    rows = []
    for nm in names:
        obs_a, boot_a = bs.metrics(P[nm][keep])
        d, lo, hi = delta_ci(boot_a, boot_s, obs_a, obs_s, metric)
        rows.append((nm, d, lo, hi, boot_p(boot_a, boot_s, metric)))
    return rows, holm(np.array([r[4] for r in rows])), obs_s[metric], bs


def summarize(rows, rej):
    d = np.array([r[1] for r in rows])
    return dict(n=len(rows), mean_abs_d=float(np.abs(d).mean()), median_d=float(np.median(d)),
                mean_width=float(np.mean([r[3] - r[2] for r in rows])),
                res95=int(sum(not (r[2] <= 0 <= r[3]) for r in rows)), holm=int(rej.sum()),
                wins95=int(sum(r[2] > 0 for r in rows)),
                wins_holm=int(sum(rej[i] and rows[i][1] > 0 for i in range(len(rows)))))


def pseudo_groups(y, n_groups, rng):
    """Random groups of equal size within each label, so the cell balance does not change."""
    g = np.zeros(len(y), int)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        g[idx] = np.arange(len(idx)) % n_groups + 1
    return g


def random_groups(b, k, reps, pool, metric="hard_min", n_boot=4000, seed=0):
    y, c, _, P, keep, names = setup(b, pool)
    rng = np.random.default_rng(seed)
    acc = []
    for rep in range(reps):
        g = pseudo_groups(y, k, rng)
        rows, rej, seed_val, _ = contrasts(P, names, y, g, tuple(range(1, k + 1)), metric, n_boot,
                                           np.random.default_rng(seed + 1000 + rep), keep)
        s = summarize(rows, rej)
        s["seed_val"] = seed_val
        acc.append(s)
    col = {key: np.array([a[key] for a in acc], float) for key in ("holm", "mean_abs_d", "mean_width")}
    print(f"{b.title}, {k} random groups, {reps} partitions, {len(names)} contrasts: significant after Holm "
          f"median {np.median(col['holm']):.0f}, mean |delta| {np.median(col['mean_abs_d']):.4f}, "
          f"interval width {np.median(col['mean_width']):.4f}")
    out = b.results / f"regroup_pseudo_{metric}_{k}g_{reps}rep.json"
    out.write_text(json.dumps({"mode": "pseudo", "n_groups": k, "reps": reps, "metric": metric,
                               "contrasts": len(names), "per_rep": acc}, indent=1), encoding="utf-8")


def meaningful_one(b, scheme, mapping, pool, metric, n_boot=4000, seed=0):
    y, c, rec, P, keep, names = setup(b, pool)
    if mapping is None:
        cc, groups = c, tuple(b.ids)
    else:
        cc = np.zeros_like(c)
        for src, dst in mapping.items():
            cc[c == src] = dst
        groups = tuple(sorted(set(mapping.values())))
    g = gba_by_group(P["seed"], y, cc, keep, groups)
    vals = np.sort(np.array(list(g.values()), float))
    rows, rej, seed_m, bs = contrasts(P, names, y, cc, groups, metric, n_boot, np.random.default_rng(seed), keep)
    s = summarize(rows, rej)

    # Reality Check under this grouping (as in chance_baseline.py)
    obs_s, boot_s = bs.metrics(P["seed"][keep])
    obs, boot = [], []
    for n in names:
        o, bt = bs.metrics(P[n][keep])
        obs.append(o[metric] - obs_s[metric])
        boot.append(bt[metric] - boot_s[metric])
    obs = np.array(obs)
    mx = (np.stack(boot, 0) - obs[:, None]).max(axis=0)
    i = int(np.argmax(obs))
    best = float(obs[i])
    p_max = float((mx >= best).mean())
    print(f"  {scheme:10s} k={len(groups):2d} pool {pool:6s} N={len(names):2d}: gap {vals[1] - vals[0]:+.4f}, "
          f"Holm {s['holm']}, best {best:+.4f}, B(N) {mx.mean():+.4f}, p {p_max:.3f}, mean |delta| {s['mean_abs_d']:.4f}")
    return dict(stand=b.key, scheme=scheme, metric=metric, k=len(groups), pool=pool, n=s["n"],
                ncell={int(gg): int(((cc == gg) & keep).sum()) for gg in groups},
                gba_seed={str(k): float(v) for k, v in g.items()}, worst=float(vals[0]),
                second_worst=float(vals[1]), gap=float(vals[1] - vals[0]), spread=float(vals[-1] - vals[0]),
                seed_metric=float(seed_m), width=s["mean_width"], half_width=s["mean_width"] / 2,
                mean_abs_d=s["mean_abs_d"], res95=s["res95"], wins95=s["wins95"], holm=s["holm"],
                wins_holm=s["wins_holm"], best=best, best_name=names[i], ceiling_mean=float(mx.mean()),
                ceiling_p95=float(np.percentile(mx, 95)), p_max=p_max)


def meaningful(b, metric="hard_min"):
    print(f"{b.title}, groupings with a meaning, metric {metric}")
    out = [meaningful_one(b, s, m, pool, metric) for s, m in SCHEMES[b.key].items() for pool in ("all", "finals")]
    (b.results / f"semantic_regroup_{metric}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def quantile_groups(vals, q):
    edges = np.quantile(vals, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, vals, side="right") + 1


def features(b, metric="hard_min", n_boot=4000, seed=0):
    if b.key != "mnli":
        raise SystemExit("the feature regroupings are defined for MultiNLI")
    y, c, rec, P, keep, names = setup(b, "finals")
    out = {}
    for feat, field in FEATURES.items():
        g = quantile_groups(np.asarray(rec[field], float), 4)
        gids = tuple(sorted(set(int(x) for x in g)))
        rows, rej, seed_val, _ = contrasts(P, names, y, g, gids, metric, n_boot, np.random.default_rng(seed), keep)
        s = summarize(rows, rej)
        out[feat] = dict(s, seed_val=float(seed_val), k=len(gids))
        print(f"{b.title}, quartiles of {feat}: seed {seed_val:.4f}, mean |delta| {s['mean_abs_d']:.4f}, "
              f"width {s['mean_width']:.4f}, distinguishable {s['res95']}/{s['n']}, Holm {s['holm']}")
    (b.results / "regroup_features.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cell_size(b, metric="hard_min", n_boot=4000, seed=0):
    """Prediction H6: mean interval width ordered C < A < B, Holm count not decreasing B -> A -> C."""
    if b.key != "mnli":
        raise SystemExit("the cell-size check is defined for MultiNLI")
    y, c, rec = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, controls=False)
    keep = valid_rows(P, len(y), verbose=False)
    finals = sorted(n for n in P if b.is_final(n))
    neg = np.asarray(rec["neg_broad"], dtype=bool)
    raw = np.asarray(c) * 2 + neg.astype(int)
    partitions = {"A": np.asarray(c), "B": np.unique(raw, return_inverse=True)[1] + 1, "C": neg.astype(int) + 1}
    out = {}
    for idx, (key, gid) in enumerate(partitions.items()):
        full = cell_index(y, gid, keep)
        used = [g for g in sorted(set(gid.tolist())) if len(full.get((g, 0), ())) and len(full.get((g, 1), ()))]
        cells = {(g, lab): full[(g, lab)] for g in used for lab in (0, 1)}
        bs = Bootstrap(y, gid, n_boot, np.random.default_rng([seed, idx]), cells=cells, groups=used)
        obs, boot = {n: None for n in finals + ["seed"]}, {}
        for n in obs:
            obs[n], boot[n] = bs.metrics(P[n])
        d, lo, hi, p = (np.array(x) for x in zip(*[(*delta_ci(boot[f], boot["seed"], obs[f], obs["seed"], metric),
                                                    boot_p(boot[f], boot["seed"], metric)) for f in finals]))
        out[key] = {"groups": len(used), "cell_min": int(min(len(v) for v in cells.values())),
                    "width_mean": float((hi - lo).mean()), "abs_d_mean": float(np.abs(d).mean()),
                    "res95": int(((lo > 0) | (hi < 0)).sum()), "holm": int(holm(p).sum()), "n": len(finals)}
        print(f"{key}: {len(used)} groups, smallest cell {out[key]['cell_min']}, mean width "
              f"{out[key]['width_mean']:.4f}, distinguishable {out[key]['res95']}, Holm {out[key]['holm']}")
    (b.results / "cell_size_h6.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["random", "meaningful", "features", "cell-size"])
    benchmarks.add_argument(ap)
    ap.add_argument("--k", type=int, default=2, help="random: number of groups")
    ap.add_argument("--reps", type=int, default=50, help="random: number of partitions")
    ap.add_argument("--pool", choices=["all", "finals"], default="all", help="random: contrasts with the seed")
    ap.add_argument("--metric", default="hard_min")
    args = ap.parse_args()
    b = benchmarks.load(args.benchmark)
    b.results.mkdir(parents=True, exist_ok=True)
    if args.mode == "random":
        random_groups(b, args.k, args.reps, args.pool, args.metric)
    elif args.mode == "meaningful":
        meaningful(b, args.metric)
    elif args.mode == "features":
        features(b, args.metric)
    else:
        cell_size(b, args.metric)


if __name__ == "__main__":
    main()

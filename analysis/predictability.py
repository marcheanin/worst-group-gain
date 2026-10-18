"""Can a benchmark's ability to resolve a worst-group gain be predicted before the search? (Appendix M)

Three candidate predictors, each computable before an optimizer runs:

  gap        the gap between the worst and second-worst group under the seed, set against the number
             of significant gains, over the groupings of `regroup.py meaningful`;
  leverage   mean |change in worst-class accuracy| over the eleven one-line edits divided by mean
             |change in worst-group balanced accuracy| over the same edits, with the real groups and
             with random groups of the same size (median over 50 partitions within each label). Only
             rows that belong to a group enter. Computed on the rows of the main analysis and, as before
             the live loops on toxicity, on the rows the seed and the edits parse (the prompts that exist
             before any search);
  ceiling    the noise ceiling estimated on the edits before the search and extrapolated to the planned
             number of finals (multilingual toxicity, as fixed before the live loops: a log-linear fit
             of the ceiling over N = 2, 5, 10, 11), against the ceiling of the finals; and why it falls
             short: finals disagree with the seed on more examples than edits do, so their intervals
             are wider.

Writes `results/<benchmark>/predictability.json`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.stats import Bootstrap, load_preds, load_set, metric, valid_rows


def leverage(b, P, y, c, keep_rows, reps=50, seed=0):
    keep = keep_rows & np.isin(c, list(b.ids))
    edits = sorted(n for n in P if n.startswith("r15:"))
    k = len(b.ids)

    def hm_response(cc, groups):
        base = metric("hard_min", P["seed"], y, cc, keep, groups)
        return float(np.mean([abs(metric("hard_min", P[n], y, cc, keep, groups) - base) for n in edits]))

    base_wc = metric("worst_class", P["seed"], y, c, keep, b.ids)
    wc = float(np.mean([abs(metric("worst_class", P[n], y, c, keep, b.ids) - base_wc) for n in edits]))
    real = hm_response(c, b.ids)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pc = np.zeros_like(c)
        for lab in (0, 1):
            idx = np.flatnonzero((y == lab) & keep)
            pc[rng.permutation(idx)] = np.tile(np.arange(1, k + 1), len(idx) // k + 1)[:len(idx)]
        vals.append(hm_response(pc, tuple(range(1, k + 1))))
    pseudo = float(np.median(vals))
    return {"rows": int(keep.sum()), "worst_class_response": wc, "hard_min_response_real": real,
            "hard_min_response_pseudo_median": pseudo, "leverage_real": wc / real, "leverage_pseudo": wc / pseudo}


def noise_matrix(b, P, y, c, keep, names, seed=0):
    bs = Bootstrap(y[keep], c[keep], 4000, np.random.default_rng(seed), groups=b.ids)
    obs_s, boot_s = bs.metrics(P["seed"][keep])
    obs, boot = [], []
    for n in names:
        o, bt = bs.metrics(P[n][keep])
        obs.append(o["hard_min"] - obs_s["hard_min"])
        boot.append(bt["hard_min"] - boot_s["hard_min"])
    obs, boot = np.array(obs), np.stack(boot)
    half = float(np.mean([np.percentile(bt, 97.5) - np.percentile(bt, 2.5) for bt in boot]) / 2)
    return boot - obs[:, None], half


def edits_ceiling(b, P, y, c, keep, n_planned=12, seed=0):
    """Ceiling over subpools of the edits (mean and 95th percentile at N = 2, 5, 10, 11), extrapolated
    with ceiling ~ a + b ln N to the planned number of finals."""
    edits = sorted(n for n in P if n.startswith("r15:"))
    noise, _ = noise_matrix(b, P, y, c, keep, edits, seed)
    N = len(edits)
    rng = np.random.default_rng(seed + 7)
    pts = {}
    for k in sorted({2, 5, 10, N}):
        mean = float(np.mean([noise[rng.choice(N, size=k, replace=False)].max(axis=0).mean() for _ in range(200)]))
        p95 = float(np.percentile(noise[rng.choice(N, size=k, replace=False)].max(axis=0), 95))
        pts[k] = {"mean": mean, "p95": p95}
    x = np.log(list(pts))
    fit = {key: float(np.polyval(np.polyfit(x, [v[key] for v in pts.values()], 1), np.log(n_planned)))
           for key in ("mean", "p95")}
    return {"rows": int(keep.sum()), "by_N": pts, "extrapolated_to": n_planned, "extrapolated": fit}


def edits_vs_finals(b, P, y, c, keep):
    out = {}
    for kind, names in (("edits", [n for n in P if n.startswith("r15:")]), ("finals", [n for n in P if b.is_final(n)])):
        disagree = float(np.mean([(P[n][keep] != P["seed"][keep]).mean() for n in names]))
        _, half = noise_matrix(b, P, y, c, keep, sorted(names))
        out[kind] = {"n": len(names), "disagree_with_seed": disagree, "half_width": half}
    return out


def run(b):
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y), controls=False)
    keep_main = valid_rows(P, len(y), verbose=False)
    keep_before = valid_rows(P, len(y), ["seed"] + [n for n in P if n.startswith("r15:")], verbose=False)
    res = {"leverage_main_rows": leverage(b, P, y, c, keep_main),
           "leverage_before_search": leverage(b, P, y, c, keep_before),
           "edits_vs_finals": edits_vs_finals(b, P, y, c, keep_main)}
    lm, lv = res["leverage_main_rows"], res["leverage_before_search"]
    print(f"{b.title}: leverage {lm['leverage_real']:.2f} with real groups, {lm['leverage_pseudo']:.2f} with random "
          f"groups on the {lm['rows']} rows of the main analysis; {lv['leverage_real']:.2f} / {lv['leverage_pseudo']:.2f} "
          f"on the {lv['rows']} rows the seed and the edits parse")
    ev = res["edits_vs_finals"]
    print(f"  disagreement with the seed: edits {100 * ev['edits']['disagree_with_seed']:.1f}%, finals "
          f"{100 * ev['finals']['disagree_with_seed']:.1f}%; half-width {ev['edits']['half_width']:.4f} / "
          f"{ev['finals']['half_width']:.4f}")
    if b.key == "toxlang":
        res["edits_ceiling"] = edits_ceiling(b, P, y, c, keep_before)
        ec = res["edits_ceiling"]
        print(f"  ceiling on the edits extrapolated to {ec['extrapolated_to']} prompts: C95 {ec['extrapolated']['p95']:+.4f} "
              f"(N = 11: {ec['by_N'][11]['p95']:+.4f})")
    sem = b.results / "semantic_regroup_hard_min.json"
    if sem.is_file():
        res["gap"] = [{"grouping": r["scheme"], "k": r["k"], "gap": r["gap"], "holm": r["holm"]}
                      for r in json.loads(sem.read_text(encoding="utf-8")) if r["pool"] == "all"]
        print("  gap between the two worst groups vs significant gains: "
              + ", ".join(f"{g['grouping']} {g['gap']:.4f} -> {g['holm']}" for g in res["gap"]))
    (b.results / "predictability.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    run(benchmarks.load(ap.parse_args().benchmark))


if __name__ == "__main__":
    main()

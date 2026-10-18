"""The seven checks of Section 8 for a worst-group gain reported after prompt optimization.

Checks 1-6 need only the predictions an evaluation already produces; check 7 needs the predictions of a second
task model for the same prompts. For a seed prompt and a set of prompts compared with it, on one test set with
group ids:

  1  worst-class accuracy (a metric without groups) on the same predictions: do the prompts change the model at all?
  2  random groups of the same size: does the same comparison on random groups give the same count of findings?
  3  the noise ceiling: the chance baseline B(N), the ceiling C95(N) and the Reality Check p-value of the best of N gains
  4  the stability of the worst group: how often the observed worst group stays the worst under resampling
  5  the number of groups K and the interval width at that K
  6  the target-group gain, on the group that is worst for the seed, next to the worst-group gain
  7  a second task model: the target-group gain of each prompt on that model, and the split-half reliability of the
     effects on both models, which bounds the correlation between them

Usage from Python:

    from wgg.checks import run_checks, format_report
    report = run_checks(preds, y, groups, seed="seed", second=preds_on_other_model)
    print(format_report(report))

`finals` names the optimizer finals among them, the family of checks 6 and 7 (default: all prompts).
`preds` maps a prompt name to an integer vector (0/1, -1 for an unparsable answer) aligned with the labels `y` and
the group ids `groups`; the seed prompt must be among them. From the command line, on the released benchmarks or on
your own files (a directory of `<prompt>.npy` and a JSON with `labels` and `cluster_ids`):

    python -m wgg.checks --benchmark civil
    python -m wgg.checks --preds-dir my_preds --set my_set.json --exclude-groups 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.stats import spearmanr

from wgg.stats import Bootstrap, boot_p, cell_index, delta_ci, holm


def _valid(preds: Dict[str, np.ndarray]) -> np.ndarray:
    return ~np.any([p < 0 for p in preds.values()], axis=0)


def _contrasts(preds, y, c, keep, ids, prompts, seed, n_boot, rng_seed=0):
    bs = Bootstrap(y[keep], c[keep], n_boot, np.random.default_rng(rng_seed), groups=ids)
    obs, boot = {}, {}
    for n in [seed] + list(prompts):
        obs[n], boot[n] = bs.metrics(preds[n][keep])
    return bs, obs, boot


def _summarize(obs, boot, prompts, seed, metric):
    rows = []
    for n in prompts:
        d, lo, hi = delta_ci(boot[n], boot[seed], obs[n], obs[seed], metric)
        rows.append({"prompt": n, "delta": d, "lo": lo, "hi": hi, "p": boot_p(boot[n], boot[seed], metric)})
    rej = holm(np.array([r["p"] for r in rows]))
    for r, h in zip(rows, rej):
        r["holm"] = bool(h)
    return rows


def _target_contrasts(preds, y, c, keep, ids, prompts, seed, target, n_boot, rng_seed=0):
    bs = Bootstrap(y[keep], c[keep], n_boot, np.random.default_rng(rng_seed), groups=ids)

    def gba(pred):
        cells, idx, sl = bs.cells, bs.idx, bs.slices
        obs = 0.5 * ((pred[cells[(target, 1)]] == 1).mean() + (pred[cells[(target, 0)]] == 0).mean())
        boot = 0.5 * ((pred[idx[:, sl[(target, 1)]]] == 1).mean(1) + (pred[idx[:, sl[(target, 0)]]] == 0).mean(1))
        return float(obs), boot

    so, sb = gba(preds[seed][keep])
    rows = []
    for n in prompts:
        o, b = gba(preds[n][keep])
        d = b - sb
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / n_boot))
        rows.append({"prompt": n, "delta": o - so, "lo": float(lo), "hi": float(hi), "p": p})
    for r, h in zip(rows, holm(np.array([r["p"] for r in rows]))):
        r["holm"] = bool(h)
    return so, rows


def _random_groups(y, k, rng):
    g = np.zeros(len(y), int)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        g[idx] = np.arange(len(idx)) % k + 1
    return g


def _reliability(preds, y, c, keep, target, prompts, seed, reps=200, rng_seed=0):
    """Split-half reliability of the target-group effects (prompt minus seed), Spearman-Brown corrected."""
    cells = cell_index(y, c, keep)
    rng = np.random.default_rng(rng_seed)

    def effects(rows):
        def gba(p):
            pos, neg = rows & (c == target) & (y == 1), rows & (c == target) & (y == 0)
            return 0.5 * ((p[pos] == 1).mean() + (p[neg] == 0).mean())
        s = gba(preds[seed])
        return np.array([gba(preds[n]) - s for n in prompts])

    halves = []
    for _ in range(reps):
        a, b = np.zeros(len(y), bool), np.zeros(len(y), bool)
        for rows in cells.values():
            perm = rng.permutation(rows)
            a[perm[: len(perm) // 2]] = True
            b[perm[len(perm) // 2:]] = True
        halves.append((a, b))
    r = float(np.nanmean([spearmanr(effects(a), effects(b)).statistic for a, b in halves]))
    return r, (2 * r / (1 + r) if r > 0 else 0.0)


def run_checks(preds: Dict[str, np.ndarray], y, groups, *, seed: str = "seed", prompts: Optional[Iterable[str]] = None,
               finals: Optional[Iterable[str]] = None, second: Optional[Dict[str, np.ndarray]] = None, exclude_groups: Iterable[int] = (),
               n_boot: int = 4000, n_random: int = 50, random_boot: int = 1000, rng_seed: int = 0) -> dict:
    y, c = np.asarray(y), np.asarray(groups)
    prompts = [n for n in (prompts if prompts is not None else preds) if n != seed]
    fin = [n for n in (finals if finals is not None else prompts) if n in prompts]
    ids = tuple(g for g in sorted(set(c.tolist())) if g not in set(exclude_groups))
    pool = {n: np.asarray(preds[n]) for n in [seed] + prompts}
    keep = _valid(pool)
    rep: dict = {"rows": int(keep.sum()), "of": len(y), "n_prompts": len(prompts), "K": len(ids)}

    _, obs, boot = _contrasts(pool, y, c, keep, ids, prompts, seed, n_boot, rng_seed)
    w = _summarize(obs, boot, prompts, seed, "hard_min")
    w_by = {x["prompt"]: x for x in w}
    wc = _summarize(obs, boot, prompts, seed, "worst_class")

    # 1 the same contrasts without groups
    rep["check1"] = {"worst_group_significant": sum(r["holm"] for r in w), "worst_class_significant": sum(r["holm"] for r in wc),
                     "worst_class_best_gain": max(r["delta"] for r in wc)}

    # 3 chance baseline and Reality Check for the best of N gains in the worst-group metric
    obs_d = np.array([r["delta"] for r in w])
    noise = np.stack([boot[n]["hard_min"] - boot[seed]["hard_min"] for n in prompts]) - obs_d[:, None]
    mx = noise.max(axis=0)
    best = float(obs_d.max())
    rep["check3"] = {"N": len(prompts), "best_gain": best, "best_prompt": prompts[int(obs_d.argmax())],
                     "chance_baseline": float(mx.mean()), "noise_ceiling": float(np.percentile(mx, 95)),
                     "reality_check_p": float((mx >= best).mean())}

    # 4 stability of the worst group under resampling
    gvals = {n: np.array([0.5 * ((pool[n][keep & (c == g) & (y == 1)] == 1).mean()
                                 + (pool[n][keep & (c == g) & (y == 0)] == 0).mean()) for g in ids]) for n in pool}
    rng = np.random.default_rng(rng_seed + 2)
    cells = [(np.flatnonzero(keep & (c == g) & (y == 1)), np.flatnonzero(keep & (c == g) & (y == 0))) for g in ids]
    draws = [(rng.integers(0, len(p), size=(n_boot, len(p))), rng.integers(0, len(q), size=(n_boot, len(q)))) for p, q in cells]
    stay = {}
    for n in pool:
        G = np.empty((n_boot, len(ids)))
        for k, ((dp, dq), (p, q)) in enumerate(zip(draws, cells)):
            G[:, k] = 0.5 * ((pool[n][p][dp] == 1).mean(axis=1) + (pool[n][q][dq] == 0).mean(axis=1))
        stay[n] = float((G.argmin(axis=1) == int(gvals[n].argmin())).mean())
    rep["check4"] = {"seed_worst_group": ids[int(gvals[seed].argmin())], "seed_stays_worst": stay[seed],
                     "median_stays_worst": float(np.median(list(stay.values())))}

    # 5 the number of groups and the interval width at that number
    width = float(np.mean([r["hi"] - r["lo"] for r in w]))
    rep["check5"] = {"K": len(ids), "mean_interval_width": width}

    # 2 random groups of the same size, and 5 again at the number of groups of the real grouping
    rrng = np.random.default_rng(rng_seed + 1)
    counts, widths = [], []
    for _ in range(n_random):
        g = _random_groups(y[keep], len(ids), rrng)
        rids = tuple(range(1, len(ids) + 1))
        sub = {n: pool[n][keep] for n in pool}
        rb = Bootstrap(y[keep], g, random_boot, rrng, groups=rids)
        ro, rbt = {}, {}
        for n in [seed] + prompts:
            ro[n], rbt[n] = rb.metrics(sub[n])
        rr = _summarize(ro, rbt, prompts, seed, "hard_min")
        counts.append(sum(r["holm"] for r in rr))
        widths.append(float(np.mean([r["hi"] - r["lo"] for r in rr])))
    rep["check2"] = {"real_groups_significant": rep["check1"]["worst_group_significant"],
                     "random_groups_significant_median": float(np.median(counts)),
                     "random_groups_interval_width": float(np.median(widths))}

    # 6 the target group: the group that is worst for the seed
    target = ids[int(gvals[seed].argmin())]
    seed_t, t_rows = _target_contrasts(pool, y, c, keep, ids, fin, seed, target, n_boot, rng_seed)
    rep["check6"] = {"target_group": target, "seed_balanced_accuracy": seed_t, "significant": sum(r["holm"] for r in t_rows),
                     "best_target_gain": max(r["delta"] for r in t_rows),
                     "best_worst_group_gain": best,
                     "hidden": [r["prompt"] for r in t_rows if r["holm"] and not w_by[r["prompt"]]["holm"]],
                     "gains": t_rows}

    # 7 a second task model
    if second is not None:
        have = [n for n in fin if n in second and seed in second]
        sp = {n: np.asarray(second[n]) for n in [seed] + have}
        skeep = _valid(sp)
        _, s_rows = _target_contrasts(sp, y, c, skeep, ids, have, seed, target, n_boot, rng_seed)
        main_by = {r["prompt"]: r for r in t_rows}
        both = keep & skeep
        _, rel1 = _reliability(pool, y, c, both, target, have, seed, rng_seed=rng_seed) if len(have) >= 4 else (np.nan, np.nan)
        _, rel2 = _reliability(sp, y, c, both, target, have, seed, rng_seed=rng_seed) if len(have) >= 4 else (np.nan, np.nan)
        rep["check7"] = {
            "prompts": len(have),
            "transfers": [r["prompt"] for r in s_rows if r["holm"] and r["delta"] > 0 and main_by[r["prompt"]]["holm"]],
            "significant_on_main_model": [n for n in have if main_by[n]["holm"]],
            "reversed": [r["prompt"] for r in s_rows if main_by[r["prompt"]]["holm"] and r["delta"] < 0],
            "gains_second_model": s_rows, "reliability_main": rel1, "reliability_second": rel2,
            "correlation_ceiling": float(np.sqrt(max(rel1, 0) * max(rel2, 0))) if len(have) >= 4 else None}
    return rep


def format_report(rep: dict) -> str:
    c1, c2, c3, c4, c5, c6 = (rep[f"check{i}"] for i in range(1, 7))
    lines = [f"{rep['n_prompts']} prompts against the seed, {rep['rows']} of {rep['of']} rows usable, {rep['K']} groups", "",
             f"1 worst-class accuracy: {c1['worst_class_significant']} prompts differ from the seed (best gain "
             f"{c1['worst_class_best_gain']:+.3f}); worst-group balanced accuracy: {c1['worst_group_significant']}",
             f"2 random groups of the same size: {c2['random_groups_significant_median']:.0f} significant (median) "
             f"against {c2['real_groups_significant']} with the real groups",
             f"3 best of N = {c3['N']}: {c3['best_gain']:+.3f} ({c3['best_prompt']}); chance baseline "
             f"{c3['chance_baseline']:+.3f}, noise ceiling {c3['noise_ceiling']:+.3f}, Reality Check p = {c3['reality_check_p']:.3f}",
             f"4 the seed's worst group (id {c4['seed_worst_group']}) stays the worst in {100 * c4['seed_stays_worst']:.0f}% "
             f"of resamples; median over prompts {100 * c4['median_stays_worst']:.0f}%",
             f"5 K = {c5['K']}, mean interval width {c5['mean_interval_width']:.3f} "
             f"(random groups: {c2['random_groups_interval_width']:.3f})",
             f"6 target group {c6['target_group']}: {c6['significant']} significant gains, best {c6['best_target_gain']:+.3f} "
             f"against {c6['best_worst_group_gain']:+.3f} in the worst-group metric"
             + (f"; hidden by the minimum: {', '.join(c6['hidden'])}" if c6["hidden"] else "")]
    if "check7" in rep:
        c7 = rep["check7"]
        lines.append(f"7 second model: {len(c7['significant_on_main_model'])} significant gains on the main model, "
                     f"{len(c7['transfers'])} hold; reversed: {len(c7['reversed'])}; reliability of the effects "
                     f"{c7['reliability_main']:.2f} / {c7['reliability_second']:.2f}")
    return "\n".join(lines)


def _load_dir(d: Path) -> Dict[str, np.ndarray]:
    return {f.stem.replace("__", ":", 1): np.load(f) for f in sorted(d.glob("*.npy"))
            if not f.name.endswith((".lo.npy", ".partial.npy")) and not f.stem.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["civil", "mnli", "toxlang"], default=None)
    ap.add_argument("--model", default="gemma-3-12b")
    ap.add_argument("--second-model", default=None)
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--second-dir", type=Path, default=None)
    ap.add_argument("--set", type=Path, default=None, help="JSON with `labels` and `cluster_ids`")
    ap.add_argument("--seed-name", default="seed")
    ap.add_argument("--finals-prefix", default=None, help="names starting with this are the optimizer finals (the family of checks 6 and 7)")
    ap.add_argument("--exclude-groups", type=int, nargs="*", default=None)
    ap.add_argument("--json", type=Path, default=None, help="also write the full report as JSON")
    args = ap.parse_args()
    if args.benchmark:
        from wgg import benchmarks
        from wgg.stats import load_preds, load_set
        b = benchmarks.load(args.benchmark)
        y, c, _ = load_set(b, b.test_set)
        preds = load_preds(b, b.test_set, len(y), controls=False, model=args.model)
        second = load_preds(b, b.test_set, len(y), controls=False, model=args.second_model) if args.second_model else None
        excl = (0,) if b.key == "civil" else ()
        finals = [n for n in preds if b.is_final(n)]
    else:
        rec = json.loads(args.set.read_text(encoding="utf-8"))
        y, c = np.asarray(rec["labels"]), np.asarray(rec["cluster_ids"])
        preds = _load_dir(args.preds_dir)
        second = _load_dir(args.second_dir) if args.second_dir else None
        excl = ()
        finals = [n for n in preds if args.finals_prefix and n.startswith(args.finals_prefix)] or None
    if args.exclude_groups is not None:
        excl = tuple(args.exclude_groups)
    rep = run_checks(preds, y, c, seed=args.seed_name, finals=finals, second=second, exclude_groups=excl)
    print(format_report(rep))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

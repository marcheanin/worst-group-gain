"""How much the selection protocol moves the reported result, on a fixed pool of prompts.

Feeds Appendix G (72 protocols, decomposition, length penalty, composition of the pool), the
"75% to zero" sentence of Sections 3 and 8, and the protocol-versus-method ratio of Appendix H.

A protocol picks one prompt of the pool on validation data. It has four parts: how the validation
budget of 900 rows is spread (uniformly over group x label cells, or a pilot of 20 rows per cell
followed by the three worst groups of the seed on the pilot), the selection statistic (six options),
a length penalty of 0.00025 per word above 300 or none, and the tie rule (first, random, shortest):
2 x 6 x 2 x 3 = 72 protocols. Validation rows are drawn 200 times; on every draw each protocol picks
a winner, and the winner is compared with the seed on the test set, by point estimate and by a paired
bootstrap (2,000 resamples, significant after Holm's correction over the pool).

`run` writes `results/<benchmark>/lottery_summary_<target>[_sub15_<seed>].json`; `decompose` splits the
variation of each outcome over the four parts of the protocol (main effects of the balanced design)
and writes `lottery_decompose_<target>.json`; `composition` repeats the lottery on ten random subpools
of 15 prompts (the seed always included) and writes `pool_composition.json`.
"""
from __future__ import annotations

import argparse
import json
from itertools import product

import numpy as np

from wgg import benchmarks
from wgg.stats import METRICS, Bootstrap, cell_index, gba_by_group, holm, load_preds, load_set, metric, valid_rows

STATS = ["hard_min", "cvar_k2", "mean_gba", "softmin_w40", "worst_class", "global_acc"]
PENALTY = [0.0, 0.00025]
TIEBREAK = ["first", "random", "shortest"]
TARGET_GROUPS = 3
DIMS = ["allocation", "statistic", "length penalty", "tie rule"]


def break_tie(top, how, words, rng):
    if how == "first":
        return top[0]
    if how == "shortest":
        return min(top, key=lambda n: (words.get(n, 250), n))
    return rng.choice(top)


def dev_statistic(stat, pred, y, c, rows, groups, counts):
    """The selection statistic on validation rows. cvar_k2 is the mean of the two worst groups;
    softmin_w40 shrinks each group toward the mean over groups with a weight of 40 rows."""
    if stat == "worst_class":
        p, yy = pred[rows], y[rows]
        return float(min((p[yy == 0] == 0).mean(), (p[yy == 1] == 1).mean()))
    if stat == "global_acc":
        return float((pred[rows] == y[rows]).mean())
    mask = np.zeros(len(y), bool)
    mask[rows] = True
    g = gba_by_group(pred, y, c, mask, groups)
    if not g:
        return -1.0
    if stat == "softmin_w40":
        pooled = float(np.mean(list(g.values())))
        a = np.array([(counts[k] * x + 40.0 * pooled) / (counts[k] + 40.0) for k, x in g.items()])
        return float(-0.1 * np.log(np.mean(np.exp(-a / 0.1))))
    v = np.sort(list(g.values()))
    return float({"hard_min": v[:1], "cvar_k2": v[:2], "mean_gba": v}[stat].mean())


def run(b, target="cvar25", draws=200, n_boot=2000, subsample=0, subsample_seed=0):
    yT, cT, _ = load_set(b, b.test_set)
    yD, cD, _ = load_set(b, b.dev_set)
    PT, PD = load_preds(b, b.test_set, len(yT)), load_preds(b, b.dev_set, len(yD))
    # the validation matrix of MultiNLI was scored only as far as the lottery needed it: use the common prefix
    nT, nD = len(next(iter(PT.values()))), len(next(iter(PD.values())))
    yT, cT, yD, cD = yT[:nT], cT[:nT], yD[:nD], cD[:nD]
    names = sorted(n for n in set(PT) & set(PD) if not n.startswith("CONTROL:"))
    if subsample:
        rest = [n for n in names if n != "seed"]
        pick = np.random.default_rng(subsample_seed).choice(len(rest), min(subsample - 1, len(rest)), replace=False)
        names = ["seed"] + sorted(rest[i] for i in pick)
    words = b.prompt_words()
    keepT, keepD = valid_rows(PT, len(yT), names, verbose=False), valid_rows(PD, len(yD), names, verbose=False)
    truth = {n: metric(target, PT[n], yT, cT, rows=keepT, groups=b.ids) for n in names}
    best, seed_val = max(truth.values()), truth["seed"]

    dev_cells = cell_index(yD, cD, keepD)
    rng = np.random.default_rng(0)
    protos = list(product(["uniform", "targeted"], STATS, PENALTY, TIEBREAK))
    draws_by = {p: [] for p in protos}
    picks = {p: [] for p in protos}

    # which prompts beat the seed on the test, by a paired bootstrap on the same rows
    bsT = Bootstrap(yT, cT, n_boot, np.random.default_rng(1), cells=cell_index(yT, cT, keepT), groups=b.ids)
    bootT = {n: bsT.metrics(PT[n])[1] for n in names}
    sig, pv = {}, {}
    for n in names:
        if n == "seed":
            continue
        d = bootT[n][target] - bootT["seed"][target]
        sig[n] = bool(np.percentile(d, [2.5, 97.5])[0] > 0)
        pv[n] = float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d)))
    ns = [n for n in names if n != "seed"]
    sig_holm = {n: bool(r and sig[n]) for n, r in zip(ns, holm(np.array([pv[n] for n in ns])))}
    sig["seed"] = sig_holm["seed"] = False

    for _ in range(draws):
        uni = np.concatenate([rng.choice(dev_cells[(g, l)], min(b.uniform_per_cell, len(dev_cells[(g, l)])),
                                         replace=False) for g in b.sample_groups for l in (0, 1)])
        pilot = np.concatenate([rng.choice(dev_cells[(g, l)], min(b.pilot_per_cell, len(dev_cells[(g, l)])),
                                           replace=False) for g in b.sample_groups for l in (0, 1)])
        pm = np.zeros(len(yD), bool)
        pm[pilot] = True
        sg = gba_by_group(PD["seed"], yD, cD, pm, b.ids)
        worst3 = sorted(sg, key=lambda g: sg[g])[:TARGET_GROUPS]
        per_cell = (b.dev_budget - len(pilot)) // (2 * TARGET_GROUPS)
        tgt = [pilot]
        for g in worst3:
            for l in (0, 1):
                pool = np.setdiff1d(dev_cells[(g, l)], pilot, assume_unique=False)
                tgt.append(rng.choice(pool, min(per_cell, len(pool)), replace=False))
        tgt = np.concatenate(tgt)
        for which, rows, groups in (("uniform", uni, b.ids), ("targeted", tgt, tuple(worst3))):
            counts = {g: int((cD[rows] == g).sum()) for g in groups}
            for stat, lam in product(STATS, PENALTY):
                sc = {n: dev_statistic(stat, PD[n], yD, cD, rows, groups, counts) - lam * max(0, words.get(n, 250) - 300)
                      for n in names}
                best_sc = max(sc.values())
                top = [n for n in names if sc[n] == best_sc]
                for how in TIEBREAK:
                    pick = break_tie(top, how, words, rng)
                    draws_by[(which, stat, lam, how)].append(truth[pick])
                    picks[(which, stat, lam, how)].append(pick)

    M = np.array([draws_by[p] for p in protos])  # (protocols, draws)
    within = float(np.mean(M.var(axis=1)))
    # every protocol sees the same draws, so the spread of protocol means is inflated only by the
    # protocol x draw interaction, about within / draws; subtract it
    between_raw = float(np.var(M.mean(axis=1)))
    between = max(0.0, between_raw - within / M.shape[1])
    single = float(np.mean(M.std(axis=0)))  # spread of what the protocols report on the same draw
    by_method = {}
    for n in names:
        m = b.method_of(n)
        if m is not None:
            by_method.setdefault(m, []).append(truth[n])
    method_means = np.array([np.mean(v) for v in by_method.values()]) if by_method else np.array([])

    wp = np.array([np.mean(np.array(v) > seed_val) for v in draws_by.values()])
    ws = np.array([np.mean([sig[n] for n in picks[k]]) for k in protos])
    wh = np.array([np.mean([sig_holm[n] for n in picks[k]]) for k in protos])
    print(f"{b.title}, target {target}, {len(names)} prompts, {len(protos)} protocols x {draws} draws: "
          f"point wins median {100 * np.median(wp):.1f}% ({100 * wp.min():.1f}-{100 * wp.max():.1f}), "
          f"significant after Holm median {100 * np.median(wh):.1f}% ({100 * wh.min():.1f}-{100 * wh.max():.1f})")
    rows_out = {"|".join(map(str, k)): {"mean_published": float(np.mean(v)), "mean_regret": float(best - np.mean(v)),
                                        "p_beats_seed": float(np.mean(np.array(v) > seed_val)),
                                        "win_point": float(a), "win_sig": float(s_), "win_holm": float(h)}
                for (k, v), a, s_, h in zip(draws_by.items(), wp, ws, wh)}
    stem = f"lottery_summary_{target}" + (f"_sub{subsample}_{subsample_seed}" if subsample else "")
    out = {"target": target, "n_draws": draws, "pool": "all", "pool_size": len(names),
           "pool_spread": float(np.ptp(list(truth.values()))), "truth": truth,
           "rows_used": {b.test_set: int(keepT.sum()), b.dev_set: int(keepD.sum())},
           "between_var": between, "between_var_raw": between_raw, "within_var": within,
           "sd_protocols_one_study": single,
           "sd_methods": float(method_means.std()) if len(method_means) else None,
           "protocol_set": "all",
           "win_point": {"median": float(np.median(wp)), "iqr": [float(np.percentile(wp, 25)), float(np.percentile(wp, 75))],
                         "range": [float(wp.min()), float(wp.max())]},
           "win_sig": {"median": float(np.median(ws)), "range": [float(ws.min()), float(ws.max())]},
           "win_holm": {"median": float(np.median(wh)), "range": [float(wh.min()), float(wh.max())]},
           "protocols": rows_out}
    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def decompose(b, target):
    """Share of the total sum of squares of each outcome on each part of the protocol."""
    d = json.loads((b.results / f"lottery_summary_{target}.json").read_text(encoding="utf-8"))
    keys = list(d["protocols"])
    parts = [k.split("|") for k in keys]
    out = {}
    for field in ("win_point", "win_sig", "win_holm", "mean_published"):
        vals = np.array([d["protocols"][k][field] for k in keys], float)
        grand, rows = vals.mean(), {}
        sst = ((vals - grand) ** 2).sum()
        for j, dim in enumerate(DIMS):
            levels = sorted({p[j] for p in parts})
            means, ss = {}, 0.0
            for lv in levels:
                m = np.array([p[j] == lv for p in parts])
                means[lv] = float(vals[m].mean())
                ss += m.sum() * (means[lv] - grand) ** 2
            rows[dim] = {"share": float(ss / sst) if sst > 0 else 0.0, "means": means}
        out[field] = rows
    print(f"{b.title}, {target}: share of variation in significant wins after Holm: "
          + ", ".join(f"{dim} {100 * r['share']:.1f}%" for dim, r in out["win_holm"].items()))
    (b.results / f"lottery_decompose_{target}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def composition(b, n_sub=10, size=15, draws=60):
    rows = []
    for s in range(n_sub):
        d = run(b, "cvar25", draws=draws, subsample=size, subsample_seed=s)
        wp = d["win_point"]
        rows.append({"subsample_seed": s, "pool_spread": d["pool_spread"], "sd_protocols": d["sd_protocols_one_study"],
                     "win_point_median": wp["median"], "win_point_range": wp["range"]})
    med = [r["win_point_median"] for r in rows]
    span = [r["win_point_range"][1] - r["win_point_range"][0] for r in rows]
    print(f"{n_sub} subpools of {size}: median point wins {100 * min(med):.0f}-{100 * max(med):.0f}%, "
          f"range across protocols {100 * min(span):.0f}-{100 * max(span):.0f} points")
    (b.results / "pool_composition.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["run", "decompose", "composition"])
    benchmarks.add_argument(ap, choices=("civil", "mnli"))
    ap.add_argument("--target", default="cvar25", choices=METRICS)
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    b = benchmarks.load(args.benchmark)
    if args.part == "run":
        run(b, args.target, args.draws)
    elif args.part == "decompose":
        decompose(b, args.target)
    else:
        composition(b)


if __name__ == "__main__":
    main()

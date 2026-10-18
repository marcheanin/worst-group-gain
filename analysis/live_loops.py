"""Live optimization loops: what decides the quality of a final, and does the ranking of methods carry
over between benchmarks? (Appendix H; Section 7, last paragraph)

On CivilComments and MultiNLI, APE, EvoPrompt-DE and GEPA were each run with three selection statistics
(worst-group balanced accuracy, its smoothed version, overall accuracy) and three seeds, 27 runs per
benchmark (`data/<benchmark>/live_loop/runs/seed<s>/<method>__<statistic>/`). A final that equals the
seed prompt enters with a contrast of zero and stays out of the Holm family.

  contrasts   every final against the seed on the test set for five metrics (paired bootstrap, 4,000
              resamples, seed 0; Holm over the finals that differ from the seed);
  design      the variation of the finals' CVaR@25% split over seed, method and statistic (sums of squares
              of the 3 x 3 x 3 design), permutation tests of the method and statistic effects (labels
              permuted within the other two factors, 10,000 permutations), and the ratio of the spread
              across statistics to the spread across methods, with a bootstrap over seeds;
  reselection each run's archive re-selected by the other statistics: how often the winner changes and how
              much the test CVaR@25% of the winner changes (cluster bootstrap over runs);
  ranking     per method, the runs whose final beats the seed in CVaR@25% (point estimate, interval
              excluding zero, Holm); on toxicity, where the loops ran APE, ProTeGi, EvoPrompt-DE and GEPA
              with one statistic, the mean worst-group gain over three seeds.

Writes `results/<benchmark>/live_loops.json` and `results/toxlang/live_loops_ranking.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from wgg import benchmarks
from wgg.stats import METRICS, Bootstrap, boot_p, cell_index, holm, load_preds, load_set, metric, valid_rows

METHODS = ("ape", "evoprompt_de", "gepa")
PROTOCOLS = ("soft_min", "hard_min", "global")
SEEDS = (42, 43, 44)
RUN_PREFIX = {"civil": "s12", "mnli": "s13"}
# statistic of the run -> archive field the loop selected by
PROTOCOL_KEY = {"soft_min": "R_soft_min_gba", "hard_min": "R_worst_gba", "mean_gba": "R_gba_mean", "global": "R_global"}
# statistics computable from the archive of any run; the true soft-min is stored only by soft_min runs
STAT_KEY = {"hard_min": "R_worst_gba", "global": "R_global", "mean_gba": "R_gba_mean"}
TIE_TOL = 1e-12


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class Run:
    def __init__(self, b, d):
        self.seed = int(d.parent.name.replace("seed", ""))
        self.method, _, self.protocol = d.name.partition("__")
        self.name = f"{RUN_PREFIX.get(b.key, b.final_prefix)}:{self.seed}_{self.method}__{self.protocol}"
        arch = json.loads((d / "archive.json").read_text(encoding="utf-8"))
        self.evaluated = list(arch.get("evaluated") or [])
        self.final_text = (d / "best_prompt.txt").read_text(encoding="utf-8")
        self.final_hash = text_hash(self.final_text)

    @property
    def cell(self):
        return (self.seed, self.method, self.protocol)


def load_runs(b):
    return [Run(b, d) for d in sorted((b.data / "live_loop" / "runs").glob("seed*/*")) if d.is_dir()]


def load_live_preds(b, n):
    """Predictions of the finals and of the alternative picks, by file name."""
    P = {}
    dirs = [b.data / "live_loop" / "predictions" / "test"]
    if b.key == "mnli":
        dirs.insert(0, b.preds_dir(b.test_set))  # the MultiNLI finals are the pool of the main matrix
    for d in dirs:
        for f in sorted(d.glob("*.npy")):
            name = f.stem.replace("__", ":", 1)
            if name not in P:
                P[name] = np.load(f)
    m = min(len(v) for v in P.values())
    return {k: v[:m] for k, v in P.items()}, m


def index_by_hash(P, runs, seed_hash):
    run_hash = {r.name: r.final_hash for r in runs}
    by_hash = {}
    for name in ["seed"] + [r.name for r in runs] + sorted(k for k in P if k.startswith("alt:")):
        if name not in P:
            continue
        h = seed_hash if name == "seed" else (name[4:] if name.startswith("alt:") else run_hash[name])
        by_hash.setdefault(h, (name, P[name]))
    return by_hash


def contrasts(test_runs, ob, seed_hash):
    so, sb = ob[seed_hash]
    out = {}
    for m in METRICS:
        recs = []
        for r in test_runs:
            o, bt = ob[r.final_hash]
            zero = r.final_hash == seed_hash
            if zero:
                lo = hi = 0.0
                p = 1.0
            else:
                d = bt[m] - sb[m]
                lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
                p = min(1.0, boot_p(bt, sb, m))
            recs.append({"name": r.name, "method": r.method, "protocol": r.protocol, "seed": r.seed, "zero": zero,
                         "delta": float(o[m] - so[m]), "lo": lo, "hi": hi, "p": p})
        nz = [i for i, x in enumerate(recs) if not x["zero"]]
        rej = holm(np.array([recs[i]["p"] for i in nz])) if nz else np.zeros(0, bool)
        for x in recs:
            x["holm_reject"] = False
        for i, rj in zip(nz, rej):
            recs[i]["holm_reject"] = bool(rj)
        for x in recs:
            x["point_win"] = bool(x["delta"] > TIE_TOL)
            x["sig_win"] = bool(not x["zero"] and x["lo"] > 0)
            x["holm_win"] = bool(x["holm_reject"] and x["lo"] > 0)
        out[m] = {"n": len(recs), "n_zero": len(recs) - len(nz), "resolved_holm": int(rej.sum()),
                  "point_win": sum(x["point_win"] for x in recs), "sig_win": sum(x["sig_win"] for x in recs),
                  "holm_win": sum(x["holm_win"] for x in recs), "runs": {x["name"]: x for x in recs}}
    return out


def ss_components(Yb):
    """Sums of squares for a batch of tables of shape (B, seeds, methods, statistics); one observation per
    cell, so the residual absorbs every interaction with the seed."""
    mu = Yb.mean(axis=(1, 2, 3), keepdims=True)
    m_s, m_m = Yb.mean(axis=(2, 3), keepdims=True), Yb.mean(axis=(1, 3), keepdims=True)
    m_p, m_mp = Yb.mean(axis=(1, 2), keepdims=True), Yb.mean(axis=1, keepdims=True)
    S, M, P = Yb.shape[1:]

    def sq(a):
        return (a ** 2).sum(axis=(1, 2, 3))

    out = {"seed": M * P * sq(m_s - mu), "method": S * P * sq(m_m - mu), "protocol": S * M * sq(m_p - mu),
           "method_x_protocol": S * sq(m_mp - m_m - m_p + mu), "total": sq(Yb - mu)}
    res = out["total"] - out["seed"] - out["method"] - out["protocol"] - out["method_x_protocol"]
    out["residual"] = np.where(res > -1e-12, np.maximum(res, 0.0), res)
    return out


def perm_pvalues(Y, n_perm, rng, chunk=2000):
    """Permutation tests of the main effects: statistic labels permuted within (seed, method), method labels
    within (seed, statistic); p = (1 + #{SS_perm >= SS_obs}) / (1 + n_perm)."""
    S, M, P = Y.shape
    obs = ss_components(Y[None])
    res = {}
    for kind, axis in (("protocol", 3), ("method", 2)):
        o = float(obs[kind][0])
        tol = 1e-12 * max(1.0, abs(o))
        cnt, done = 0, 0
        while done < n_perm:
            bsz = min(chunk, n_perm - done)
            idx = np.argsort(rng.random((bsz, S, M, P)), axis=axis)
            Yp = np.take_along_axis(np.broadcast_to(Y, (bsz, S, M, P)), idx, axis=axis)
            cnt += int((ss_components(Yp)[kind] >= o - tol).sum())
            done += bsz
        res[kind] = (1 + cnt) / (1 + n_perm)
    return res


def spread_ratio(Y, n_boot, rng):
    """sd of the means over statistics / sd of the means over methods, with a bootstrap over seeds."""
    S = Y.shape[0]

    def sds(Yc):
        ym = Yc.mean(axis=-3)
        return ym.mean(axis=-1).std(axis=-1), ym.mean(axis=-2).std(axis=-1)

    sd_m, sd_p = (float(v) for v in sds(Y))
    bm, bp = sds(Y[rng.integers(0, S, size=(n_boot, S))])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_b = np.where(bm > 0, bp / bm, np.nan)
    ok = ratio_b[~np.isnan(ratio_b)]
    return {"sd_method": sd_m, "sd_protocol": sd_p, "ratio": sd_p / sd_m if sd_m > 0 else float("nan"),
            "ci95": [float(v) for v in np.percentile(ok, [2.5, 97.5])]}


def archive_picks(run):
    """The prompt each computable statistic would pick on the run's archive (first evaluation of a
    candidate, rejected candidates skipped, ties to the first)."""
    rows = [(rec["hash"], rec["evals"][0]) for rec in run.evaluated
            if rec.get("evals") and not rec["evals"][0].get("rejected")]
    keys = dict(STAT_KEY)
    if run.protocol == "soft_min":
        keys["soft_min"] = "R_soft_min_gba"
    picks = {}
    for st, key in keys.items():
        best_h, best_v = None, None
        for h, e in rows:
            v = e.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if best_v is None or v > best_v:
                best_h, best_v = h, v
        picks[st] = best_h
    return picks


def reselection(b, runs, by_hash, seed_hash, y, c):
    alt_pairs, n_any, n_any_diff = [], 0, 0
    for r in runs:
        alts = {s: h for s, h in archive_picks(r).items() if s != r.protocol}
        diffs = {s: (h is not None and h != r.final_hash) for s, h in alts.items()}
        alt_pairs += [(r, s, alts[s]) for s, d in diffs.items() if d]
        n_any += bool(alts)
        n_any_diff += any(diffs.values())
    pairs = [(r, s, h) for r, s, h in alt_pairs if h in by_hash and r.final_hash in by_hash]
    hashes = sorted({h for _, _, h in pairs} | {r.final_hash for r, _, _ in pairs})
    arrs = [by_hash[h][1] for h in hashes] + ([by_hash[seed_hash][1]] if seed_hash in by_hash else [])
    keep = ~np.any([a < 0 for a in arrs], axis=0)
    tv = {h: metric("cvar25", by_hash[h][1], y, c, rows=keep, groups=b.ids) for h in hashes}
    by_run = {}
    for r, s, h in pairs:
        by_run.setdefault(r.name, []).append(tv[h] - tv[r.final_hash])
    runs_list = list(by_run.values())
    allv = np.array([d for v in runs_list for d in v])
    rng = np.random.default_rng(2)
    boot = np.array([float(np.mean([d for i in rng.integers(0, len(runs_list), len(runs_list)) for d in runs_list[i]]))
                     for _ in range(4000)])
    return {"winner_changes": n_any_diff, "runs": n_any, "pairs": len(pairs), "mean": float(allv.mean()),
            "ci95_cluster_boot": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def run(b):
    y, c, _ = load_set(b, b.test_set)
    seed_hash = text_hash(b.prompts()["seed"])
    runs = load_runs(b)
    P, n = load_live_preds(b, len(y))
    y, c = y[:n], c[:n]
    by_hash = index_by_hash(P, runs, seed_hash)
    test_runs = [r for r in runs if r.final_hash in by_hash]
    need = {seed_hash} | {r.final_hash for r in test_runs}
    keep = valid_rows({by_hash[h][0]: by_hash[h][1] for h in need}, n, verbose=False)
    bs = Bootstrap(y, c, 4000, np.random.default_rng(0), cells=cell_index(y, c, keep), groups=b.ids)
    ob = {h: bs.metrics(by_hash[h][1]) for h in need}
    con = contrasts(test_runs, ob, seed_hash)

    vals = {r.cell: float(ob[r.final_hash][0]["cvar25"]) for r in test_runs}
    Y = np.array([[[vals[(s, m, p)] for p in PROTOCOLS] for m in METHODS] for s in SEEDS], dtype=float)
    ss = {k: float(v[0]) for k, v in ss_components(Y[None]).items()}
    design = {"share": {k: ss[k] / ss["total"] for k in ss}, "perm_p": perm_pvalues(Y, 10000, np.random.default_rng(1)),
              "spread_ratio": spread_ratio(Y, 4000, np.random.default_rng(2))}
    ranking = {}
    for m in METHODS:
        rs = [x for x in con["cvar25"]["runs"].values() if x["method"] == m]
        ranking[m] = {k: sum(x[k] for x in rs) for k in ("point_win", "sig_win", "holm_win")} | {"n": len(rs)}
    res = {"rows_used": int(keep.sum()), "returned_seed": sum(r.final_hash == seed_hash for r in runs),
           "significant_after_holm": {m: con[m]["holm_win"] for m in METRICS},
           "finals_differing_from_seed": con["cvar25"]["n"] - con["cvar25"]["n_zero"],
           "contrasts": con, "design": design, "reselection": reselection(b, runs, by_hash, seed_hash, y, c),
           "ranking_cvar25": ranking}

    print(f"{b.title}: {len(runs)} runs, {res['returned_seed']} returned the seed; finals beating the seed after Holm: "
          + ", ".join(f"{m} {v}" for m, v in res["significant_after_holm"].items()))
    sh, pp, sr = design["share"], design["perm_p"], design["spread_ratio"]
    print(f"  CVaR@25% of the finals: method {100 * sh['method']:.1f}% (p {pp['method']:.4f}), statistic "
          f"{100 * sh['protocol']:.1f}% (p {pp['protocol']:.2f}); spread statistic / method {sr['ratio']:.2f} "
          f"[{sr['ci95'][0]:.2f}, {sr['ci95'][1]:.2f}]")
    rs = res["reselection"]
    print(f"  re-selection by another statistic changes the winner in {rs['winner_changes']} of {rs['runs']} runs; "
          f"test CVaR@25% changes by {rs['mean']:+.4f} [{rs['ci95_cluster_boot'][0]:+.4f}, {rs['ci95_cluster_boot'][1]:+.4f}]")
    print("  runs beating the seed in CVaR@25% (point / interval / Holm): "
          + ", ".join(f"{m} {v['point_win']}/{v['sig_win']}/{v['holm_win']}" for m, v in ranking.items()))
    # the protocol/method ratio on the fixed pool (Appendix G lotteries), for comparison
    lots = [b.results / f"lottery_summary_{t}.json" for t in ("hard_min", "cvar25")]
    if all(p.is_file() for p in lots):
        ratios = []
        for p in lots:
            d = json.loads(p.read_text(encoding="utf-8"))
            ratios.append(d["sd_protocols_one_study"] / d["sd_methods"])
        res["fixed_pool_protocol_over_method"] = ratios
        print(f"  fixed pool: spread across protocols within one study / across methods {min(ratios):.2f}-{max(ratios):.2f}")
    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / "live_loops.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def toxicity_ranking():
    """Mean worst-group gain per method over three seeds on toxicity; a run that returned the seed gains 0."""
    b = benchmarks.load("toxlang")
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y), controls=False)
    keep = valid_rows(P, len(y), verbose=False)
    seed_val = metric("hard_min", P["seed"], y, c, rows=keep, groups=b.ids)
    seed_hash = text_hash(b.prompts()["seed"])
    gains = {}
    for d in sorted((b.data / "live_loop" / "runs").glob("seed*/*")):
        r = Run(b, d)
        name = f"s15:{r.seed}_{r.method}__{r.protocol}"
        g = 0.0 if r.final_hash == seed_hash else metric("hard_min", P[name], y, c, rows=keep, groups=b.ids) - seed_val
        gains.setdefault(r.method, []).append(g)
    res = {m: {"mean_gain": float(np.mean(v)), "gains": v} for m, v in gains.items()}
    print("toxicity, mean worst-group gain over three seeds: " + ", ".join(f"{m} {v['mean_gain']:+.3f}" for m, v in res.items()))
    (b.results / "live_loops_ranking.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    b = benchmarks.load(ap.parse_args().benchmark)
    toxicity_ranking() if b.key == "toxlang" else run(b)


if __name__ == "__main__":
    main()

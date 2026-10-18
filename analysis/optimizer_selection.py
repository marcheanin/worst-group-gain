"""Does the optimizer's own selection throw away a better candidate? (Appendix I, multilingual toxicity)

For each of the 12 live runs on toxicity, the final the method returned is compared with every other
candidate of the run's archive on the first 1,000 test examples, in worst-group balanced accuracy,
mean balanced accuracy and worst-class accuracy. Three estimates of the regret (best candidate minus
final):

  naive               best candidate on the test minus the final; biased upward by the selection;
  selection-adjusted  a Reality Check within the run: the maximum of the centred bootstrap differences
                      "candidate minus final" (4,000 resamples within language x label cells), with
                      Holm's correction across the 12 runs;
  cross-fit           the best candidate is picked on one half of the test and compared with the final
                      on the other, over 200 stratified splits and both directions.

Unparsable answers: by default an example is dropped within a run when some candidate of the run did
not answer it, and a candidate whose inclusion would leave fewer than 700 examples is excluded (the
final never is); this rule was fixed in an amendment to the plan after a first look at the data
(Appendix I, "Deviation from the plan"). `--invalid-as-error` scores unparsable answers as wrong and
excludes nothing. The script also counts the candidates that break the answer format.

Writes `results/toxlang/archive_regret[_inv_err].json` and `results/toxlang/broken_candidates.json`.
"""
from __future__ import annotations

import argparse
import json
import zlib

import numpy as np

from wgg import benchmarks
from wgg.stats import holm, load_set

METRICS = ("hard_min", "mean_gba", "worst_class")
N_ROWS = 1000
N_BOOT = 4000
N_SPLITS = 200
FULL_DEV = 900
MIN_KEEP_ROWS = 700


def run_seed(name: str) -> int:
    """A fixed random seed per run, derived from its name."""
    return zlib.crc32(name.encode("utf-8")) % (2 ** 31)


def val_info(evals):
    """Validation score on the largest sample the candidate was scored on."""
    e = max(evals, key=lambda x: x["rows"])
    return float(e["selected_on"]), int(e["rows"])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    for r, v in ((ra, a), (rb, b)):
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def select_pool(names, final, P, min_keep):
    """Add candidates from the most to the least reliable while the rows parsable for all stay at or
    above `min_keep`; the final is always kept."""
    rates = {n: float((P[n] < 0).mean()) for n in names}
    keep = np.ones(N_ROWS, bool)
    pool, excluded = [], []
    for n in [final] + sorted((n for n in names if n != final), key=lambda n: rates[n]):
        cand_keep = keep & (P[n] >= 0)
        if n == final or int(cand_keep.sum()) >= min_keep:
            pool.append(n)
            keep = cand_keep
        else:
            excluded.append({"name": n, "invalid_rate": rates[n]})
    return pool, excluded, keep


class HalfMetrics:
    """Every metric of every prompt on both halves of every split, vectorized over prompts."""

    def __init__(self, ids, K, y, c, rng):
        n = K.shape[1]
        cells = [(g, l) for g in ids for l in (1, 0)]
        Cm = np.stack([(c == g) & (y == l) for g, l in cells]).astype(float)
        A = np.zeros((N_SPLITS, n), float)
        for s in range(N_SPLITS):
            for row in Cm.astype(bool):
                idx = rng.permutation(np.flatnonzero(row))
                A[s, idx[: len(idx) // 2]] = 1.0
        self.half = {}
        for tag, H in (("A", A), ("B", 1.0 - A)):
            HC = H[:, None, :] * Cm[None]
            num = K @ HC.reshape(-1, n).T
            rate = (num / HC.sum(axis=2).reshape(-1)).reshape(K.shape[0], N_SPLITS, len(cells))
            gba = 0.5 * (rate[:, :, 0::2] + rate[:, :, 1::2])
            numr = num.reshape(K.shape[0], N_SPLITS, len(cells))
            pos = numr[:, :, 0::2].sum(axis=2) / (H[:, None, :] * Cm[None, 0::2]).sum(axis=(1, 2))
            neg = numr[:, :, 1::2].sum(axis=2) / (H[:, None, :] * Cm[None, 1::2]).sum(axis=(1, 2))
            self.half[tag] = {"hard_min": gba.min(axis=2), "mean_gba": gba.mean(axis=2),
                              "worst_class": np.minimum(pos, neg)}

    def crossfit(self, metric, fi, rng):
        gains = []
        for sel, ev in (("A", "B"), ("B", "A")):
            S, E = self.half[sel][metric], self.half[ev][metric]
            for s in range(N_SPLITS):
                col = S[:, s]
                best = col.max()
                pick = fi if col[fi] >= best else int(rng.choice(np.flatnonzero(col == best)))
                gains.append(E[pick, s] - E[fi, s])
        g = np.asarray(gains)
        return float(g.mean()), float((g > 0).mean())


class Boot:
    """The shared paired bootstrap, over the cells that are non-empty on the rows of this run."""

    def __init__(self, ids, y, c, n_boot, rng):
        self.cells = {k: v for k, v in {(int(g), int(l)): np.flatnonzero((c == g) & (y == l))
                                        for g in ids for l in (0, 1)}.items() if len(v)}
        self.slices, start, cols = {}, 0, []
        for k, rows in self.cells.items():
            cols.append(rng.choice(rows, size=(n_boot, len(rows)), replace=True))
            self.slices[k] = slice(start, start + len(rows))
            start += len(rows)
        self.idx = np.concatenate(cols, axis=1)

    def metrics(self, pred, groups):
        obs_cell, boot_cell = {}, {}
        for k, rows in self.cells.items():
            obs_cell[k] = float((pred[rows] == k[1]).mean())
            boot_cell[k] = (pred[self.idx[:, self.slices[k]]] == k[1]).mean(axis=1)

        def assemble(get):
            g_vals = {g: 0.5 * (get((g, 1)) + get((g, 0))) for g in groups if (g, 1) in self.cells and (g, 0) in self.cells}
            arr = np.sort(np.stack(list(g_vals.values())), axis=0)
            gp = [g for g in groups if (g, 1) in self.cells]
            gn = [g for g in groups if (g, 0) in self.cells]
            pos = sum(get((g, 1)) * len(self.cells[(g, 1)]) for g in gp) / sum(len(self.cells[(g, 1)]) for g in gp)
            neg = sum(get((g, 0)) * len(self.cells[(g, 0)]) for g in gn) / sum(len(self.cells[(g, 0)]) for g in gn)
            return {"hard_min": arr[0], "mean_gba": arr.mean(axis=0), "worst_class": np.minimum(pos, neg)}

        return assemble(lambda k: obs_cell[k]), assemble(lambda k: boot_cell[k])


def load_archive(b):
    amap = json.loads((b.data / "archive" / "archive_map.json").read_text(encoding="utf-8"))
    names = sorted({x["name"] for r in amap["runs"] for x in r["candidates"]} | {r["final"] for r in amap["runs"]})
    P = {}
    for n in names:
        stem = n.replace(":", "__", 1)
        d = b.data / "archive" / "predictions" / "test" if n.startswith("arc:") else b.preds_dir(b.test_set)
        P[n] = np.load(d / f"{stem}.npy")[:N_ROWS].astype(int)
    return amap, P


def broken_candidates(b):
    """Archive slots per method and how many candidates leave >= 50% / >= 95% of answers unparsable."""
    amap, P = load_archive(b)
    out = {}
    for r in amap["runs"]:
        row = out.setdefault(r["method"], {"slots": 0, "unparsable_ge_50": 0, "unparsable_ge_95": 0})
        for x in r["candidates"]:
            rate = float((P[x["name"]] < 0).mean())
            row["slots"] += 1
            row["unparsable_ge_50"] += int(rate >= 0.5)
            row["unparsable_ge_95"] += int(rate >= 0.95)
    print("archive slots, >=50% and >=95% unparsable: "
          + "; ".join(f"{m} {v['slots']}/{v['unparsable_ge_50']}/{v['unparsable_ge_95']}" for m, v in out.items()))
    (b.results / "broken_candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def run(invalid_as_error=False):
    b = benchmarks.load("toxlang")
    amap, P = load_archive(b)
    y, c, _ = load_set(b, b.test_set)
    y, c = y[:N_ROWS], c[:N_ROWS]
    if invalid_as_error:
        for n in P:
            P[n] = np.where(P[n] < 0, 1 - y, P[n])

    tie_rng = np.random.default_rng(2)
    rows_out = []
    for r in amap["runs"]:
        cand, vinfo = [], {}
        for x in r["candidates"]:
            v = val_info(x["evals"])
            if x["name"] not in vinfo:
                cand.append(x["name"])
                vinfo[x["name"]] = v
            elif v[1] > vinfo[x["name"]][1]:
                vinfo[x["name"]] = v
        f = r["final"]
        all_names = sorted(set(cand) | {f})
        if invalid_as_error:
            pool, excluded, keep = all_names, [], np.ones(N_ROWS, bool)
        else:
            pool, excluded, keep = select_pool(all_names, f, P, MIN_KEEP_ROWS)
        yk, ck = y[keep], c[keep]
        bs = Boot(b.ids, yk, ck, N_BOOT, np.random.default_rng(run_seed(r["run"])))
        obs, boot = {}, {}
        for n in pool:
            o, bt = bs.metrics(P[n][keep], groups=b.ids)
            obs[n], boot[n] = {m: float(o[m]) for m in METRICS}, {m: bt[m] for m in METRICS}
        K = np.stack([(P[n][keep] == yk).astype(float) for n in pool])
        half = HalfMetrics(b.ids, K, yk, ck, np.random.default_rng(run_seed(r["run"] + "h")))
        fi = pool.index(f)
        rec = {"run": r["run"], "method": r["method"], "seed": r["seed"], "final": f, "k": len(cand),
               "k_pool": len(pool), "k_excluded": len(excluded), "excluded": excluded, "rows_kept": int(keep.sum())}
        for m in METRICS:
            o = np.array([obs[n][m] for n in pool])
            bt = np.stack([boot[n][m] for n in pool])
            R = float(o.max() - o[fi])
            Mstar = ((bt - bt[fi]) - (o - o[fi])[:, None]).max(axis=0)
            cf, cf_pos = half.crossfit(m, fi, tie_rng)
            rec[m] = {"final_test": float(o[fi]), "best_test": float(o.max()), "best": pool[int(o.argmax())],
                      "naive_regret": R, "c95": float(np.percentile(Mstar, 95)),
                      "p_sel": float(max((Mstar >= R - 1e-12).mean(), 1.0 / N_BOOT)),
                      "crossfit_regret": cf, "crossfit_share_pos": cf_pos}
        rows_out.append(rec)
    for x, rj in zip(rows_out, holm([x["hard_min"]["p_sel"] for x in rows_out])):
        x["hard_min"]["holm_reject"] = bool(rj)

    summary = {"rows_total": N_ROWS, "min_keep_rows": MIN_KEEP_ROWS,
               "invalid_mode": "as_error" if invalid_as_error else "excluded_by_floor",
               "n_prompts": len(P), "n_runs": len(rows_out),
               "total_excluded_candidates": sum(x["k_excluded"] for x in rows_out)}
    for m in METRICS:
        cf = np.array([x[m]["crossfit_regret"] for x in rows_out])
        # interval of the mean cross-fit regret: bootstrap over runs
        rb = np.random.default_rng(0).integers(0, len(cf), size=(10000, len(cf)))
        summary[m] = {"mean_naive_regret": float(np.mean([x[m]["naive_regret"] for x in rows_out])),
                      "mean_crossfit_regret": float(cf.mean()),
                      "crossfit_ci": [float(v) for v in np.percentile(cf[rb].mean(axis=1), [2.5, 97.5])],
                      "n_p_below_05": int(sum(x[m]["p_sel"] < 0.05 for x in rows_out))}
    summary["hard_min"]["n_holm"] = int(sum(x["hard_min"]["holm_reject"] for x in rows_out))

    print(f"toxicity, 12 runs, unparsable answers {'scored as wrong' if invalid_as_error else 'dropped within the run'}")
    print(f"  {'run':<27} {'kept':>4} {'excl':>4} {'rows':>5} {'naive':>8} {'cross-fit':>9} {'p':>6} {'C95':>6}")
    for x in rows_out:
        h = x["hard_min"]
        print(f"  {x['run']:<27} {x['k_pool']:>4} {x['k_excluded']:>4} {x['rows_kept']:>5} {h['naive_regret']:>+8.4f} "
              f"{h['crossfit_regret']:>+9.4f} {h['p_sel']:>6.3f} {h['c95']:>6.3f}")
    for m in METRICS:
        s = summary[m]
        print(f"  {m:<12} naive {s['mean_naive_regret']:+.4f}  cross-fit {s['mean_crossfit_regret']:+.4f} "
              f"[{s['crossfit_ci'][0]:+.4f}, {s['crossfit_ci'][1]:+.4f}]")
    print(f"  significant after Holm: {summary['hard_min']['n_holm']} of 12; excluded candidates "
          f"{summary['total_excluded_candidates']}")
    b.results.mkdir(parents=True, exist_ok=True)
    out = b.results / ("archive_regret_inv_err.json" if invalid_as_error else "archive_regret.json")
    out.write_text(json.dumps({"summary": summary, "runs": rows_out}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--invalid-as-error", action="store_true")
    args = ap.parse_args()
    run(args.invalid_as_error)
    if not args.invalid_as_error:
        broken_candidates(benchmarks.load("toxlang"))


if __name__ == "__main__":
    main()

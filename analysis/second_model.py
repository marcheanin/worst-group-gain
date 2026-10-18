"""The main and the second task model side by side (Appendix J).

`table`: both models under the seed prompt against the same human labels, on the rows of each
benchmark's main analysis: balanced accuracy per group and its mean, the target group (worst for the
main model), each model's worst group, the Spearman correlation of the group order, and the share of
examples of each group on which the two models give the same answer.
-> results/second_model_seed_table.json

`reliability` (CivilComments): a correlation between the effects of the same 25 finals on two models
can only be as high as the reliability of each measurement allows. Split-half reliability of the
effects "final minus seed" is the Spearman correlation between two random halves of the test (within
group x label cells), averaged over 500 halvings and corrected to the full test with the
Spearman-Brown formula (zero when the half correlation is negative); the ceiling on an observable
correlation is sqrt(r1 r2). -> results/civil/target_rho_reliability.json

The replication of Sections 4 and 6 on GPT-4o-mini is the same analyses run with `--model gpt-4o-mini`
(chance_baseline.py, replication.py, argmin.py).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from wgg import benchmarks
from wgg.benchmarks import MAIN_MODEL
from wgg.stats import cell_index, gba_by_group, load_preds, load_set, metric, valid_rows

SECOND = "gpt-4o-mini"


def table():
    out = {"benchmarks": {}}
    for key in ("toxlang", "mnli", "civil"):
        b = benchmarks.load(key)
        y, c, rec = load_set(b, b.test_set)
        n = len(y)
        names = b.group_names(rec)
        P = load_preds(b, b.test_set, n)
        keep = valid_rows(P, n, sorted(k for k in P if not k.startswith("CONTROL:")), verbose=False)
        g_seed = P["seed"].astype(int)
        q_seed = np.load(b.preds_dir(b.test_set, SECOND) / "seed.npy").astype(int)
        mask = keep & (q_seed >= 0)
        target = names[min((gs := gba_by_group(P["seed"], y, c, keep, b.ids)), key=gs.get)]
        bg, bq = gba_by_group(g_seed, y, c, mask, b.ids), gba_by_group(q_seed, y, c, mask, b.ids)
        groups = {}
        for g in b.ids:
            m = mask & (c == g)
            agree = g_seed[m] == q_seed[m]
            groups[names[g]] = {"n_rows": int(m.sum()), "ba_gemma": float(bg[g]), "ba_gpt": float(bq[g]),
                                "agreement": float(agree.mean())}
        ga = np.array([v["ba_gemma"] for v in groups.values()])
        qa = np.array([v["ba_gpt"] for v in groups.values()])

        def worst(k):
            o = sorted(((nm, v[k]) for nm, v in groups.items()), key=lambda t: (round(t[1], 12), t[0]))
            return {"worst": o[0][0], "worst_value": o[0][1], "second_worst": o[1][0], "second_value": o[1][1],
                    "gap_second_minus_worst": o[1][1] - o[0][1]}

        a_t = groups[target]["agreement"]
        rank = 1 + sum(1 for v in groups.values() if round(v["agreement"], 12) < round(a_t, 12))
        summary = {"mean_ba_gemma": float(ga.mean()), "mean_ba_gpt": float(qa.mean()), "target_group": target,
                   "target_ba_gemma": groups[target]["ba_gemma"], "target_ba_gpt": groups[target]["ba_gpt"],
                   "gemma": worst("ba_gemma"), "gpt": worst("ba_gpt"),
                   # values rounded before ranking, so that 0.9159999999999999 and 0.916 count as a tie
                   "spearman_rho_tie_robust": float(spearmanr(rankdata(np.round(ga, 12)), rankdata(np.round(qa, 12))).statistic),
                   "ba_range_gemma": [float(ga.min()), float(ga.max())],
                   "target_agreement": a_t, "target_agreement_rank_from_lowest": rank}
        out["benchmarks"][key] = {"main_rows": {"n_rows": int(mask.sum()), "groups": groups, "summary": summary}}
        print(f"{b.title}, {int(mask.sum())} rows: mean BA {summary['mean_ba_gemma']:.3f} / {summary['mean_ba_gpt']:.3f}; "
              f"{target} {summary['target_ba_gemma']:.3f} / {summary['target_ba_gpt']:.3f}; worst "
              f"{summary['gemma']['worst']} / {summary['gpt']['worst']}; rho {summary['spearman_rho_tie_robust']:.2f}; "
              f"agreement on {target} {100 * a_t:.1f}% (rank {rank} from the lowest)")
    root = benchmarks.results_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "second_model_seed_table.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def reliability(reps=500):
    b = benchmarks.load("civil")
    y, c, _ = load_set(b, b.test_set)
    p1 = load_preds(b, b.test_set, controls=False, model=MAIN_MODEL)
    p2 = load_preds(b, b.test_set, controls=False, model=SECOND)
    names = sorted(set(p1) & set(p2))
    keep = np.all([(p1[n] >= 0) & (p2[n] >= 0) for n in names], axis=0)
    finals = [n for n in names if b.is_final(n)]

    def gba(pred, g, rows):
        return 0.5 * ((pred[rows & (c == g) & (y == 1)] == 1).mean() + (pred[rows & (c == g) & (y == 0)] == 0).mean())

    seed_g = {g: gba(p1["seed"], g, keep) for g in b.ids}
    target = min(seed_g, key=seed_g.get)

    def effects(P, kind, rows):
        f = (lambda p: gba(p, target, rows)) if kind == "target" else (lambda p: metric(kind, p, y, c, rows=rows, groups=b.ids))
        s = f(P["seed"])
        return np.array([f(P[n]) - s for n in finals])

    rng = np.random.default_rng(0)
    cells = cell_index(y, c, keep)
    halves = []
    for _ in range(reps):
        h1, h2 = np.zeros(len(y), bool), np.zeros(len(y), bool)
        for rows in cells.values():
            perm = rng.permutation(rows)
            h1[perm[: len(perm) // 2]] = True
            h2[perm[len(perm) // 2:]] = True
        halves.append((h1, h2))

    def rel(P, kind):
        r = float(np.mean([spearmanr(effects(P, kind, a), effects(P, kind, bb)).statistic for a, bb in halves]))
        return r, (2 * r / (1 + r) if r > 0 else 0.0)

    res = {"rows": int(keep.sum()), "finals": len(finals), "target_group_id": int(target), "reps": reps, "rows_by": {}}
    print(f"CivilComments, {len(finals)} finals, {int(keep.sum())} rows: rho between models, reliability main / second, ceiling")
    for kind in ("target", "hard_min", "worst_class", "mean_gba"):
        rho = float(spearmanr(effects(p1, kind, keep), effects(p2, kind, keep)).statistic)
        h1, r1 = rel(p1, kind)
        h2, r2 = rel(p2, kind)
        res["rows_by"][kind] = {"rho_cross": rho, "half_rho_gemma": h1, "rel_gemma": r1, "half_rho_gpt": h2,
                                "rel_gpt": r2, "rho_ceiling": float(np.sqrt(r1 * r2))}
        print(f"  {kind:12s} {rho:6.2f}   {r1:5.2f} / {r2:5.2f}   {np.sqrt(r1 * r2):5.2f}")
    (b.results / "target_rho_reliability.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["table", "reliability"])
    args = ap.parse_args()
    table() if args.part == "table" else reliability()


if __name__ == "__main__":
    main()

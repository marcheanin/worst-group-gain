"""The target-group gain and the operating points on the target group.

Feeds the target-group block of Table 2, Section 5, Figure 3 and Appendix F. The same contrasts on
the other task models are in `transfer.py`.

The target group g0 is the group on which the seed is worst on the main task model (white on
CivilComments, verbatim on MultiNLI, Amharic on toxicity). The target-group gain of a prompt is its
balanced accuracy on g0 minus that of the seed, with a paired bootstrap stratified by group x label
cell (4,000 resamples, seed 0). Rows with an unparsable answer from any prompt of the pool are
dropped for all prompts. Holm's correction runs over the optimizer finals; the one-line edits are
left out of that family.

Operating points are the false-positive rate, recall and balanced accuracy on g0 of every prompt.
A final lies "above the curve" when its balanced accuracy on g0 exceeds the best of the one-line
edits and the seed (the dashed line of Figure 3).

On toxicity the script also reports the change on German for the two GEPA finals and the sensitivity
of the p-values to the bootstrap seed and to dropping rows over the finals only. `--high-resolution`
repeats the contrasts of the finals with 100,000 resamples under three seeds.

Writes `results/<benchmark>/target_group_contrast.json` and `target_group_operating_points.json`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.stats import Bootstrap, boot_p, cell_index, gba_by_group, holm, load_preds, load_set, valid_rows

N_BOOT = 4000


class Contrast:
    """Prompt-minus-seed contrasts on the target group and in W, on fixed rows."""

    def __init__(self, b, y, c, keep, seed_pred, rng_seed=0, n_boot=N_BOOT):
        self.keep, self.n_boot, self.ids = keep, n_boot, b.ids
        self.bs = Bootstrap(y[keep], c[keep], n_boot, np.random.default_rng(rng_seed), groups=b.ids)
        self.o_seed, self.b_seed = self.bs.metrics(seed_pred[keep])

    def group_gba(self, pred, g):
        cells, idx, sl = self.bs.cells, self.bs.idx, self.bs.slices
        obs = 0.5 * ((pred[cells[(g, 1)]] == 1).mean() + (pred[cells[(g, 0)]] == 0).mean())
        boot = 0.5 * ((pred[idx[:, sl[(g, 1)]]] == 1).mean(1) + (pred[idx[:, sl[(g, 0)]]] == 0).mean(1))
        return float(obs), boot

    def group_contrast(self, pred_full, seed_full, g):
        so, sb = self.group_gba(seed_full[self.keep], g)
        o, bt = self.group_gba(pred_full[self.keep], g)
        d = bt - sb
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / self.n_boot))
        return {"delta": o - so, "ci": [float(lo), float(hi)], "p": p, "gba": o, "gba_seed": so}

    def hard_min_contrast(self, pred_full):
        o, bt = self.bs.metrics(pred_full[self.keep])
        d = bt["hard_min"] - self.b_seed["hard_min"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": float(o["hard_min"] - self.o_seed["hard_min"]), "ci": [float(lo), float(hi)],
                "p": boot_p(bt, self.b_seed, "hard_min")}


def add_holm(block, names, key, m_family=None):
    """Holm over `names`; with a larger family the missing contrasts enter with p = 1."""
    ps = [block[n][key]["p"] for n in names]
    if m_family and m_family > len(ps):
        ps = ps + [1.0] * (m_family - len(ps))
    for n, r in zip(names, holm(np.asarray(ps))[:len(names)]):
        block[n][key]["holm"] = bool(r)


def contrasts_for(b, P, y, c, keep, target, finals, rng_seed=0, m_family=None):
    ct = Contrast(b, y, c, keep, P["seed"], rng_seed)
    res = {}
    for n in finals:
        tg = ct.group_contrast(P[n], P["seed"], target)
        res[n] = {"target": {k: tg[k] for k in ("delta", "ci", "p")}, "hard_min": ct.hard_min_contrast(P[n])}
    add_holm(res, finals, "target", m_family)
    add_holm(res, finals, "hard_min", m_family)
    return ct, res


def operating_points(P, y, c, keep, target, names):
    pos, neg = (c == target) & (y == 1) & keep, (c == target) & (y == 0) & keep
    out = {}
    for n in names:
        fpr, tpr = float((P[n][neg] == 1).mean()), float((P[n][pos] == 1).mean())
        out[n] = {"fpr": fpr, "tpr": tpr, "gba": 0.5 * (tpr + (1 - fpr))}
    return out


def op_summary(ops, edits, finals):
    curve = edits + ["seed"]
    best_edit = max(ops[n]["gba"] for n in curve)
    above = [n for n in finals if ops[n]["gba"] > best_edit]
    return {"edits_fpr_range_incl_seed": [min(ops[n]["fpr"] for n in curve), max(ops[n]["fpr"] for n in curve)],
            "edits_fpr_range_edits_only": [min(ops[n]["fpr"] for n in edits), max(ops[n]["fpr"] for n in edits)],
            "best_edit_gba": best_edit, "best_edit": max(curve, key=lambda n: ops[n]["gba"]),
            "n_finals": len(finals), "n_finals_above_curve": len(above), "finals_above_curve": above}


def fmt(r):
    return f"{r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] p={r['p']:.4f}" + \
        (f" Holm={'yes' if r.get('holm') else 'no'}" if "holm" in r else "")


def run(b):
    y, c, rec = load_set(b, b.test_set)
    n = len(y)
    names_g = b.group_names(rec)
    P = load_preds(b, b.test_set, n)
    pool = sorted(k for k in P if not k.startswith("CONTROL:"))
    keep = valid_rows(P, n, pool)
    finals = [k for k in pool if b.is_final(k)]
    edits = [k for k in pool if k.startswith("r15:")]

    gs = gba_by_group(P["seed"], y, c, keep, b.ids)
    target = min(gs, key=gs.get)
    tname = names_g[target]
    print(f"{b.title}: {int(keep.sum())} rows, {len(finals)} finals, target group {tname} "
          f"(seed {gs[target]:.4f})")

    ct, res = contrasts_for(b, P, y, c, keep, target, finals)
    for f in finals:
        print(f"  {f:<30} {fmt(res[f]['target'])}   W {res[f]['hard_min']['delta']:+.4f}")
    n_t = sum(res[f]["target"]["holm"] for f in finals)
    n_h = sum(res[f]["hard_min"]["holm"] for f in finals)
    best = max(finals, key=lambda f: res[f]["target"]["delta"])
    print(f"  significant after Holm: target {n_t}/{len(finals)}, W {n_h}/{len(finals)}; best {best}")

    out = {"dataset": b.key, "model": "gemma-3-12b-it", "target_group": tname, "target_group_id": int(target),
           "seed_gba_target": float(ct.group_gba(P["seed"][keep], target)[0]),
           "n_rows": int(n), "n_rows_kept": int(keep.sum()), "n_boot": N_BOOT, "rng_seed": 0,
           "n_finals": len(finals), "n_holm_target": int(n_t), "n_holm_hard_min": int(n_h),
           "best_final": {"name": best, **res[best]["target"]},
           "finals": {f: {"delta": res[f]["target"]["delta"], "ci": res[f]["target"]["ci"],
                          "p": res[f]["target"]["p"], "holm": res[f]["target"]["holm"],
                          "hard_min": res[f]["hard_min"]} for f in finals}}

    ops = operating_points(P, y, c, keep, target, edits + finals + ["seed"])
    op_out = {"dataset": b.key, "target_group": tname, "n_rows_kept": int(keep.sum()),
              "models": {"gemma-3-12b-it": {"points": ops, "summary": op_summary(ops, edits, finals)}}}
    s = op_out["models"]["gemma-3-12b-it"]["summary"]
    print(f"  edits cover FPR {s['edits_fpr_range_incl_seed'][0]:.3f}-{s['edits_fpr_range_incl_seed'][1]:.3f}, "
          f"best edit {s['best_edit']} {s['best_edit_gba']:.4f}, finals above it "
          f"{s['n_finals_above_curve']}/{s['n_finals']}")

    if b.key == "toxlang":
        de = [k for k, v in names_g.items() if v == "de"][0]
        gepa = [f for f in finals if "gepa" in f]
        out["german"] = {}
        for f in gepa:
            r = ct.group_contrast(P[f], P["seed"], de)
            out["german"][f] = {k: r[k] for k in ("delta", "ci", "p")}
            out["german"][f]["p_bonferroni_6_groups"] = min(1.0, 6 * r["p"])
            print(f"  German {f:<22} {fmt(r)}")

        sens = {"main_exclusion_seeds": {}}
        for sd in range(5):
            _, r = contrasts_for(b, P, y, c, keep, target, finals, rng_seed=sd)
            sens["main_exclusion_seeds"][str(sd)] = {f: r[f]["target"]["p"] for f in gepa}
        keep_alt = valid_rows(P, n, finals + ["seed"])
        _, r_alt = contrasts_for(b, P, y, c, keep_alt, target, finals)
        sens["finals_plus_seed_exclusion"] = {
            "n_rows_kept": int(keep_alt.sum()), "p": {f: r_alt[f]["target"]["p"] for f in gepa},
            "delta": {f: r_alt[f]["target"]["delta"] for f in gepa},
            "holm": {f: r_alt[f]["target"]["holm"] for f in gepa}}
        out["p_sensitivity_s15_gepa"] = sens

        # per-language recall, specificity and balanced accuracy of the seed and the GEPA finals (Appendix F)
        per_lang = {}
        for n in ["seed"] + gepa:
            per_lang[n] = {}
            for g in b.ids:
                pos, neg = (c == g) & (y == 1) & keep, (c == g) & (y == 0) & keep
                tpr, tnr = float((P[n][pos] == 1).mean()), float((P[n][neg] == 0).mean())
                per_lang[n][names_g[g]] = {"tpr": tpr, "tnr": tnr, "ba": 0.5 * (tpr + tnr)}
            print(f"  {n:<24} " + " ".join(f"{k} {v['ba']:.3f}" for k, v in per_lang[n].items()))
        out["per_language"] = per_lang

    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / "target_group_contrast.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (b.results / "target_group_operating_points.json").write_text(json.dumps(op_out, indent=1), encoding="utf-8")
    print(f"wrote {b.results / 'target_group_contrast.json'} and target_group_operating_points.json")


def high_resolution(b, n_boot=100_000, seeds=(0, 1, 2), chunk=4000):
    """The target-group contrasts of the finals again with 100,000 resamples under several seeds.

    A (resamples x rows) index matrix does not fit in memory at this size, so resamples are drawn in
    chunks from one random stream per seed; every chunk is shared by all prompts, so contrasts stay
    paired. The first chunk of seed 0 is the 4,000-resample draw of `run`. Writes
    `results/<benchmark>/target_group_contrast_b100k.json`.
    """
    y, c, rec = load_set(b, b.test_set)
    names_g = b.group_names(rec)
    P = load_preds(b, b.test_set, len(y))
    pool = sorted(k for k in P if not k.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), pool)
    gs = gba_by_group(P["seed"], y, c, keep, b.ids)
    target = min(gs, key=gs.get)
    finals = [k for k in pool if b.is_final(k)]
    names = finals + ["seed"]
    cells = cell_index(y, c, keep)

    def summarize(obs, boot, B):
        res, ps = {}, []
        for n in finals:
            d = boot[n] - boot["seed"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / B))
            res[n] = {"delta": float(obs[n] - obs["seed"]), "ci": [float(lo), float(hi)], "p": p,
                      "p_mc_se": float(np.sqrt(p * (1 - p) / B))}
            ps.append(p)
        for n, r in zip(finals, holm(np.asarray(ps))):
            res[n]["holm"] = bool(r)
        return res

    per_seed, pooled, obs = {}, {n: [] for n in names}, {}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        boot, done = {n: [] for n in names}, 0
        while done < n_boot:
            m = min(chunk, n_boot - done)
            bs = Bootstrap(y, c, m, rng, cells=cells, groups=b.ids)
            s1, s0 = bs.slices[(target, 1)], bs.slices[(target, 0)]
            for n in names:
                pr = P[n]
                boot[n].append(0.5 * ((pr[bs.idx[:, s1]] == 1).mean(1) + (pr[bs.idx[:, s0]] == 0).mean(1)))
                if done == 0:
                    obs[n] = float(0.5 * ((pr[cells[(target, 1)]] == 1).mean() + (pr[cells[(target, 0)]] == 0).mean()))
            done += m
        boot = {n: np.concatenate(v) for n, v in boot.items()}
        if sd == seeds[0]:
            ref = summarize(obs, {n: v[:chunk] for n, v in boot.items()}, chunk)
        per_seed[sd] = summarize(obs, boot, n_boot)
        for n in names:
            pooled[n].append(boot[n])
        print(f"  seed {sd}: significant after Holm {sum(v['holm'] for v in per_seed[sd].values())}/{len(finals)}")
    pooled_res = summarize(obs, {n: np.concatenate(v) for n, v in pooled.items()}, n_boot * len(seeds))

    out = {"dataset": b.key, "n_boot": n_boot, "chunk": chunk, "seeds": list(seeds), "n_rows": int(len(y)),
           "n_rows_kept": int(keep.sum()), "family": "optimizer finals", "target_group": names_g[target],
           "seed_gba_target": obs["seed"], "n_finals": len(finals),
           "n_holm": {str(sd): int(sum(v["holm"] for v in r.values())) for sd, r in per_seed.items()},
           "b4000_seed0": {"n_boot": chunk, "n_holm": int(sum(v["holm"] for v in ref.values())), "contrasts": ref},
           "seeds_results": {str(sd): r for sd, r in per_seed.items()},
           "pooled_over_seeds": {"n_boot": n_boot * len(seeds),
                                 "n_holm": int(sum(v["holm"] for v in pooled_res.values())),
                                 "contrasts": pooled_res}}
    path = b.results / "target_group_contrast_b100k.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    ap.add_argument("--high-resolution", action="store_true",
                    help="repeat the contrasts of the finals with 100,000 resamples under seeds 0, 1, 2")
    args = ap.parse_args()
    b = benchmarks.load(args.benchmark)
    high_resolution(b) if args.high_resolution else run(b)


if __name__ == "__main__":
    main()

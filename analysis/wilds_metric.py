"""The worst-group accuracy of WILDS on the CivilComments pool (Appendix E, "The WILDS metric").

WILDS scores CivilComments by the minimum of plain accuracy over the 16 cells formed by the eight
identities and the two labels, WGA16 = min over (identity, label) of the cell accuracy, whereas this
paper averages the two classes within a group first. The analysis of Section 4 is repeated for WGA16
on the same predictions and rows: paired bootstrap within cells (4,000 resamples, seed 0), every
prompt against the seed with Holm's correction, the chance baseline and the Reality Check (also with
the contrasts centred at their bootstrap means), and the contrast on the cell that is worst for the
seed. Two further quantities show what WGA16 tracks: its Spearman correlation with the share of
comments a prompt labels toxic, and, for the final with the largest gain on the seed's worst cell,
the recall on the toxic comments of the same identity.

Our cells assign each comment to one identity, so this applies the WILDS metric to our test set; it
does not reproduce the WILDS benchmark. Writes `results/civil/wilds_wga16.json`.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from wgg import benchmarks
from wgg.stats import Bootstrap, holm, load_preds, load_set, valid_rows

N_BOOT = 4000


def p_two_sided(d):
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d)))


def ceiling(obs, boot, best):
    out = {}
    for center, nz in (("observed", boot - obs[:, None]), ("bootmean", boot - boot.mean(axis=1, keepdims=True))):
        mx = nz.max(axis=0)
        out[center] = {"mean": float(mx.mean()), "p95": float(np.percentile(mx, 95)), "p_noise": float((mx >= best).mean())}
    return out


def run():
    b = benchmarks.load("civil")
    y, c, rec = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y))
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), every, verbose=False)
    names = [n for n in every if n != "seed"]
    finals = [n for n in names if b.is_final(n)]
    gnames = b.group_names(rec)
    bs = Bootstrap(y[keep], c[keep], N_BOOT, np.random.default_rng(0), groups=b.ids)

    def cells(pred):
        obs, boot = bs.cell_rates(pred)
        keys = [k for k in obs if k[0] in b.ids]
        return {k: obs[k] for k in keys}, {k: boot[k] for k in keys}

    def wga(obs, boot):
        return min(obs.values()), np.min(np.stack([boot[k] for k in obs]), axis=0), min(obs, key=obs.get)

    so_cells, sb_cells = cells(P["seed"][keep])
    s_obs, s_boot, s_arg = wga(so_cells, sb_cells)
    rows, W_obs, W_boot, arg_cells = [], [], [], {}
    for n in names:
        o, bt, arg = wga(*cells(P[n][keep]))
        d = bt - s_boot
        lo, hi = np.percentile(d, [2.5, 97.5])
        cell = f"{gnames[arg[0]]}|{arg[1]}"
        rows.append({"name": n, "delta": float(o - s_obs), "ci": [float(lo), float(hi)], "p": p_two_sided(d),
                     "argmin_cell": cell})
        arg_cells[cell] = arg_cells.get(cell, 0) + 1
        W_obs.append(o - s_obs)
        W_boot.append(bt - s_boot)
    W_obs, W_boot = np.array(W_obs), np.stack(W_boot)
    for r, h in zip(rows, holm([r["p"] for r in rows])):
        r["holm"] = bool(h)
    i_best = int(np.argmax(W_obs))
    best = float(W_obs[i_best])
    fin_idx = [i for i, n in enumerate(names) if n in finals]
    best_f = float(W_obs[fin_idx].max())
    width = float(np.mean([r["ci"][1] - r["ci"][0] for r in rows]))
    resolved = sum(1 for r in rows if r["ci"][0] > 0 or r["ci"][1] < 0)

    # the contrast on the cell that is worst for the seed, over the finals
    t_rows = []
    for n in finals:
        oc, bc = cells(P[n][keep])
        d = bc[s_arg] - sb_cells[s_arg]
        lo, hi = np.percentile(d, [2.5, 97.5])
        t_rows.append({"name": n, "delta": float(oc[s_arg] - so_cells[s_arg]), "ci": [float(lo), float(hi)],
                       "p": p_two_sided(d)})
    for r, h in zip(t_rows, holm([r["p"] for r in t_rows])):
        r["holm"] = bool(h)
    tb = max(t_rows, key=lambda r: r["delta"])

    # what WGA16 tracks: the share labelled toxic, and recall on the same identity for the best cell gain
    yk, ck = y[keep], c[keep]
    wga_vals = [wga(*cells(P[n][keep]))[0] for n in names]
    toxic_share = [float((P[n][keep] == 1).mean()) for n in names]
    rho = float(spearmanr(wga_vals, toxic_share).statistic)
    g = s_arg[0]

    def recall_gba(pred):
        pos, neg = (ck == g) & (yk == 1), (ck == g) & (yk == 0)
        tpr, tnr = float((pred[pos] == 1).mean()), float((pred[neg] == 0).mean())
        return {"recall_toxic": tpr, "accuracy_nontoxic": tnr, "gba": 0.5 * (tpr + tnr)}

    res = {"rows": int(keep.sum()), "n_boot": N_BOOT, "seed_wga16": float(s_obs),
           "seed_worst_cell": f"{gnames[s_arg[0]]}|{s_arg[1]}",
           "pool": {"N": len(names), "mean_ci_width": width, "resolved_95": resolved,
                    "holm": int(sum(r["holm"] for r in rows)),
                    "wins_holm": sum(1 for r in rows if r["holm"] and r["delta"] > 0),
                    "best_name": names[i_best], "best_delta": best, "ceiling": ceiling(W_obs, W_boot, best)},
           "finals": {"N": len(finals), "best_delta": best_f, "ceiling": ceiling(W_obs[fin_idx], W_boot[fin_idx], best_f)},
           "argmin_cells_over_pool": arg_cells,
           "spearman_wga16_vs_share_toxic": rho,
           "seed_worst_cell_contrast": {"cell": f"{gnames[s_arg[0]]}|{s_arg[1]}", "seed_value": float(so_cells[s_arg]),
                                        "holm": int(sum(r["holm"] for r in t_rows)),
                                        "best": {"name": tb["name"], "delta": tb["delta"], "p": tb["p"]},
                                        "best_vs_seed_on_identity": {"seed": recall_gba(P["seed"][keep]),
                                                                     "final": recall_gba(P[tb["name"]][keep])},
                                        "contrasts": t_rows},
           "contrasts": rows}
    pc = res["pool"]["ceiling"]
    print(f"CivilComments, WGA16: seed {s_obs:.3f} ({res['seed_worst_cell']}), width {width:.3f}, "
          f"distinguishable {resolved}/{len(names)}, Holm {res['pool']['holm']} (wins {res['pool']['wins_holm']}); "
          f"best {best:+.3f}, B(N) {pc['observed']['mean']:+.3f}, C95 {pc['observed']['p95']:+.3f}, "
          f"p {pc['observed']['p_noise']:.3f} (bootstrap-mean centring {pc['bootmean']['p_noise']:.3f})")
    print(f"  minimum cells over the pool: {arg_cells}; Spearman with share toxic {rho:+.3f}")
    bv = res["seed_worst_cell_contrast"]["best_vs_seed_on_identity"]
    print(f"  {tb['name']}: {tb['delta']:+.3f} on {res['seed_worst_cell']}; recall on toxic "
          f"{bv['seed']['recall_toxic']:.3f} -> {bv['final']['recall_toxic']:.3f}, balanced accuracy "
          f"{bv['seed']['gba']:.3f} -> {bv['final']['gba']:.3f}")
    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / "wilds_wga16.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    run()

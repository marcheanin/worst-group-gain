"""Do the target-group gains of the main task model hold on other task models? (Section 7, Appendix J)

Four finals have a significant target-group gain on Gemma 3 12B: the two GEPA finals on multilingual
toxicity (Amharic) and an APE and a GEPA final on MultiNLI (verbatim). The same prompts were scored by
GPT-4o-mini and Qwen3-235B, and on MultiNLI also by Qwen3-30B.

  gate      the accuracy rule fixed before the further models were run: with the seed prompt, mean
            balanced accuracy over groups at most 0.03 below the main model, on the target group at most
            0.05 below, and at most 2% unparsable answers; on the main model's rows.
            -> results/transfer_gate_<model>.json
  mnli      the two MultiNLI finals against the seed in W and on verbatim, on the further model's own
            parsable rows (paired bootstrap, 4,000 resamples, seed 0; Holm over the two planned contrasts),
            the main model on the same rows, and both models' seed accuracy and agreement per genre.
            -> results/mnli/transfer_<model>.json
  toxicity  the toxicity finals the further model scored against the seed on Amharic and in W, Holm over
            the family of all nine finals (finals the model did not score enter with p = 1), and the
            operating points on Amharic. -> results/toxlang/transfer_<model>.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from analysis.target_group import contrasts_for, op_summary, operating_points
from wgg import benchmarks
from wgg.benchmarks import MAIN_MODEL
from wgg.stats import Bootstrap, boot_p, gba_by_group, holm, load_preds, load_set, valid_rows

MNLI_FINALS = ["s13:43_ape__global", "s13:42_gepa__hard_min"]
MAX_MEAN_DROP, MAX_TARGET_DROP, MAX_INVALID = 0.03, 0.05, 0.02
N_BOOT = 4000


def gate(model):
    out = {}
    for key in ("toxlang", "mnli"):
        b = benchmarks.load(key)
        y, c, rec = load_set(b, b.test_set)
        names = b.group_names(rec)
        P = load_preds(b, b.test_set, len(y))
        keep = valid_rows(P, len(y), sorted(k for k in P if not k.startswith("CONTROL:")), verbose=False)
        seed_path = b.preds_dir(b.test_set, model) / "seed.npy"
        if not seed_path.is_file():
            continue
        q = np.load(seed_path).astype(int)
        mask = keep & (q >= 0)
        g = P["seed"].astype(int)
        gd = {names[i]: float(v) for i, v in gba_by_group(g, y, c, rows=mask, groups=b.ids).items()}
        qd = {names[i]: float(v) for i, v in gba_by_group(q, y, c, rows=mask, groups=b.ids).items()}
        target = min(gd, key=gd.get)
        r = {"rows": int(mask.sum()), "invalid_share": float((q < 0).mean()),
             "mean_ba_main": float(np.mean(list(gd.values()))), "mean_ba_third": float(np.mean(list(qd.values()))),
             "target": target, "target_ba_main": gd[target], "target_ba_third": qd[target],
             "worst_third": min(qd, key=qd.get), "agreement_all": float((g[mask] == q[mask]).mean()),
             "spearman_group_order": float(spearmanr(list(gd.values()), list(qd.values())).statistic),
             "per_group_main": gd, "per_group_third": qd}
        r["pass"] = bool(r["mean_ba_third"] >= r["mean_ba_main"] - MAX_MEAN_DROP
                         and r["target_ba_third"] >= r["target_ba_main"] - MAX_TARGET_DROP
                         and r["invalid_share"] <= MAX_INVALID)
        out[key] = r
        print(f"{model} on {b.title}: mean BA {r['mean_ba_third']:.3f} vs {r['mean_ba_main']:.3f}, {target} "
              f"{r['target_ba_third']:.3f} vs {r['target_ba_main']:.3f}, unparsable {100 * r['invalid_share']:.1f}%, "
              f"group order rho {r['spearman_group_order']:.2f} -> {'passes' if r['pass'] else 'fails'}")
    path = benchmarks.results_root() / f"transfer_gate_{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def mnli_contrasts(b, P, y, c, keep, target):
    bs = Bootstrap(y[keep], c[keep], N_BOOT, np.random.default_rng(0), groups=b.ids)
    o_s, b_s = bs.metrics(P["seed"][keep])

    def tgt(pred):
        cells, sl = bs.cells, bs.slices
        ob = 0.5 * ((pred[cells[(target, 1)]] == 1).mean() + (pred[cells[(target, 0)]] == 0).mean())
        bt = 0.5 * ((pred[bs.idx[:, sl[(target, 1)]]] == 1).mean(1) + (pred[bs.idx[:, sl[(target, 0)]]] == 0).mean(1))
        return ob, bt

    ts_o, ts_b = tgt(P["seed"][keep])
    out = {}
    for n in MNLI_FINALS:
        o, bt = bs.metrics(P[n][keep])
        d = bt["hard_min"] - b_s["hard_min"]
        to, tb = tgt(P[n][keep])
        td = tb - ts_b
        out[n] = {"hard_min": {"delta": float(o["hard_min"] - o_s["hard_min"]),
                               "ci": [float(v) for v in np.percentile(d, [2.5, 97.5])], "p": boot_p(bt, b_s, "hard_min")},
                  "target": {"delta": float(to - ts_o), "ci": [float(v) for v in np.percentile(td, [2.5, 97.5])],
                             "p": float(max(2 * min((td <= 0).mean(), (td >= 0).mean()), 1 / N_BOOT))}}
    for key in ("hard_min", "target"):
        for n, r in zip(MNLI_FINALS, holm([out[n][key]["p"] for n in MNLI_FINALS])):
            out[n][key]["holm"] = bool(r)
    return out


def mnli(model):
    b = benchmarks.load("mnli")
    y, c, rec = load_set(b, b.test_set)
    n = len(y)
    names = {int(k): v for k, v in rec["group_names"].items()}
    load = lambda m: {k: np.load(b.preds_dir(b.test_set, m) / f"{k.replace(':', '__', 1)}.npy").astype(int)[:n]  # noqa: E731
                      for k in ["seed"] + MNLI_FINALS}
    G, Q = load(MAIN_MODEL), load(model)
    keep_q = np.all([v >= 0 for v in Q.values()], axis=0)
    keep_g = np.all([v >= 0 for v in G.values()], axis=0)
    both = keep_q & keep_g

    def group_gba(pred, yy, cc, g):
        return 0.5 * ((pred[(cc == g) & (yy == 1)] == 1).mean() + (pred[(cc == g) & (yy == 0)] == 0).mean())

    seed_g = {g: group_gba(G["seed"][keep_g], y[keep_g], c[keep_g], g) for g in b.ids}
    target = min(seed_g, key=seed_g.get)
    res = {"model": model, "rows": n, "rows_valid": int(keep_q.sum()), "rows_valid_both": int(both.sum()),
           "invalid": {k: int((v < 0).sum()) for k, v in Q.items()}, "target_group": names[target],
           "other": mnli_contrasts(b, Q, y, c, keep_q, target), "gemma_same_rows": mnli_contrasts(b, G, y, c, both, target)}
    acc = {names[g]: {"gemma": float(group_gba(G["seed"][both], y[both], c[both], g)),
                      "other": float(group_gba(Q["seed"][both], y[both], c[both], g)),
                      "agree": float((G["seed"][both & (c == g)] == Q["seed"][both & (c == g)]).mean())} for g in b.ids}
    res["seed_accuracy"] = acc
    res["seed_accuracy_mean"] = {k: float(np.mean([a[k] for a in acc.values()])) for k in ("gemma", "other", "agree")}
    for f in MNLI_FINALS:
        h, t, g = res["other"][f]["hard_min"], res["other"][f]["target"], res["gemma_same_rows"][f]["target"]
        print(f"{model} on MultiNLI, {f}: verbatim {t['delta']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] "
              f"Holm {'yes' if t['holm'] else 'no'} (main model {g['delta']:+.3f}); W {h['delta']:+.3f} "
              f"Holm {'yes' if h['holm'] else 'no'}")
    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / f"transfer_{model}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def toxicity(model):
    b = benchmarks.load("toxlang")
    y, c, rec = load_set(b, b.test_set)
    n = len(y)
    P = load_preds(b, b.test_set, n)
    pool = sorted(k for k in P if not k.startswith("CONTROL:"))
    keep = valid_rows(P, n, pool, verbose=False)
    finals = [k for k in pool if b.is_final(k)]
    edits = [k for k in pool if k.startswith("r15:")]
    gs = gba_by_group(P["seed"], y, c, keep, b.ids)
    target = min(gs, key=gs.get)

    Q = load_preds(b, b.test_set, n, controls=False, model=model)
    keep_q = valid_rows(Q, n, sorted(Q), verbose=False)
    q_finals = [f for f in finals if f in Q]
    q_edits = [e for e in edits if e in Q]
    ctq, rq = contrasts_for(b, Q, y, c, keep_q, target, q_finals, m_family=len(finals))
    _, rg = contrasts_for(b, P, y, c, keep_q, target, q_finals, m_family=len(finals))
    ops = operating_points(Q, y, c, keep_q, target, q_edits + q_finals + ["seed"])
    out = {"model": model, "target_group": b.group_names(rec)[target], "n_rows": int(keep_q.sum()),
           "finals_scored": q_finals, "edits_scored": q_edits, "n_finals_total": len(finals),
           "invalid": {k: int((v < 0).sum()) for k, v in Q.items()},
           "seed_gba_target": float(ctq.group_gba(Q["seed"][keep_q], target)[0]),
           "other": {f: {**{k: rq[f]["target"][k] for k in ("delta", "ci", "p", "holm")}, "hard_min": rq[f]["hard_min"]}
                     for f in q_finals},
           "gemma_same_rows": {f: {k: rg[f]["target"][k] for k in ("delta", "ci", "p", "holm")} for f in q_finals},
           "operating_points": {"points": ops, "summary": op_summary(ops, q_edits, q_finals)}}
    for f in q_finals:
        a = out["other"][f]
        print(f"{model} on toxicity, {f}: Amharic {a['delta']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] "
              f"p={a['p']:.3f} Holm {'yes' if a['holm'] else 'no'}; operating point {ops[f]['gba']:.3f}")
    s = out["operating_points"]["summary"]
    print(f"  best of the edits and the seed on Amharic: {s['best_edit']} {s['best_edit_gba']:.3f}")
    b.results.mkdir(parents=True, exist_ok=True)
    (b.results / f"transfer_{model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["gate", "mnli", "toxicity"])
    ap.add_argument("--model", required=True, choices=["gpt-4o-mini", "qwen3-235b", "qwen3-30b"])
    args = ap.parse_args()
    {"gate": gate, "mnli": mnli, "toxicity": toxicity}[args.part](args.model)


if __name__ == "__main__":
    main()

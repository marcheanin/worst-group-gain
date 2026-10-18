"""Every prompt of the pool against the seed on the test set, for five metrics.

Feeds Table 2 (mean interval width, distinguishable and significant contrasts in W and WC), Table 1
(test examples used) and the metrics table of Appendix E. Writes `results/<benchmark>/power_summary.json`
with each prompt's observed metric values (`truth_values`, also read by later analyses) and, per
metric, the mean 95% interval width, the number of contrasts whose interval excludes zero
(`resolved`) and the number that survive Holm's correction over the pool (`resolved_holm`).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.stats import METRICS, Bootstrap, boot_p, cell_index, delta_ci, holm, load_preds, load_set, valid_rows


def run(b, n_boot=4000):
    y, c, rec = load_set(b, b.test_set)
    P = load_preds(b, b.test_set)
    n = len(next(iter(P.values())))
    y, c = y[:n], c[:n]
    keep = valid_rows(P, len(y))
    bs = Bootstrap(y, c, n_boot, np.random.default_rng(0), cells=cell_index(y, c, keep), groups=b.ids)
    obs, boot = {}, {}
    for name, p in P.items():
        obs[name], boot[name] = bs.metrics(p)

    others = sorted(name for name in P if name != "seed" and not name.startswith("CONTROL:"))
    print(f"{b.title}: {int(keep.sum())} of {len(y)} test examples used, {len(others)} prompts against the seed")
    print(f"{'metric':12s} {'mean CI width':>13s} {'resolved':>10s} {'Holm':>6s}  best contrast")
    summary = {}
    for m in METRICS:
        widths, res, best, pv = [], 0, None, []
        for name in others:
            d, lo, hi = delta_ci(boot[name], boot["seed"], obs[name], obs["seed"], m)
            widths.append(hi - lo)
            res += int(lo > 0 or hi < 0)
            pv.append(boot_p(boot[name], boot["seed"], m))
            if best is None or d > best[0]:
                best = (d, lo, hi, name)
        rejected = holm(np.array(pv))
        summary[m] = {"ci_width": float(np.mean(widths)), "resolved": res,
                      "resolved_holm": int(rejected.sum()), "of": len(others), "min_p": float(min(pv))}
        print(f"{m:12s} {np.mean(widths):13.4f} {res:7d}/{len(others)} {int(rejected.sum()):6d}  "
              f"{best[3]} {best[0]:+.4f} [{best[1]:+.4f}, {best[2]:+.4f}]")

    # gains of the optimizer finals in W, by method (Appendix K on CivilComments)
    by_method = {}
    for name in others:
        m = b.method_of(name)
        if m is not None:
            by_method.setdefault(m, []).append(float(obs[name]["hard_min"] - obs["seed"]["hard_min"]))
    methods = {m: {"finals": len(v), "mean_gain": float(np.mean(v)), "best_gain": float(max(v))}
               for m, v in sorted(by_method.items())}

    b.results.mkdir(parents=True, exist_ok=True)
    out = b.results / "power_summary.json"
    out.write_text(json.dumps(
        {"n": int(len(y)), "n_used": int(keep.sum()), "fingerprint": rec["fingerprint"], "prompts": len(P),
         "truth_values": {name: {m: float(obs[name][m]) for m in METRICS} for name in P},
         "resolvable": summary, "methods": methods}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    benchmarks.add_argument(ap)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    run(benchmarks.load(args.benchmark), args.n_boot)


if __name__ == "__main__":
    main()

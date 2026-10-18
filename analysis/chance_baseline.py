"""The chance baseline B(N), the noise ceiling C95(N) and the Reality Check for the best of N prompts.

Feeds Table 2, Figure 1 and Figure 2. Under the null that no prompt differs from the seed, the
largest of N noisy contrasts is still positive. In each paired bootstrap resample every contrast is
centred at its observed value, which removes any true effect but keeps the correlation between
contrasts (they share the seed and the rows), and the maximum over the N contrasts is recorded.
Its mean is B(N), its 95th percentile C95(N), and the share of resamples in which it reaches the
observed best gain is the Reality Check p-value.

As a sensitivity check the contrasts are also centred at their own bootstrap means
(`p_max_bootmean`); the verdict is the same on all three benchmarks.

`--pool finals` drops the one-line edits ("finals only" in Table 2). Writes
`results/<benchmark>/best_of_n_ceiling_<metric>_<pool>.json`, including `ceiling_by_N`, the mean
maximum over random subpools of size N (Figure 2).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.benchmarks import MAIN_MODEL
from wgg.stats import Bootstrap, load_preds, load_set, valid_rows


def run(b, metric="hard_min", pool="all", n_boot=4000, seed=0, model=MAIN_MODEL):
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y), model=model)
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), every)
    names = [n for n in every if n != "seed"]
    if pool == "finals":
        names = [n for n in names if b.is_final(n)]
    bs = Bootstrap(y[keep], c[keep], n_boot, np.random.default_rng(seed), groups=b.ids)
    obs_s, boot_s = bs.metrics(P["seed"][keep])
    obs, boot = [], []
    for n in names:
        o, bt = bs.metrics(P[n][keep])
        obs.append(o[metric] - obs_s[metric])
        boot.append(bt[metric] - boot_s[metric])
    obs = np.array(obs)                      # (N,)
    boot = np.stack(boot, axis=0)            # (N, n_boot)
    noise = boot - obs[:, None]
    N = len(names)
    i_best = int(np.argmax(obs))
    best = float(obs[i_best])
    half = float(np.mean([np.percentile(boot[i], 97.5) - np.percentile(boot[i], 2.5) for i in range(N)]) / 2)
    mx = noise.max(axis=0)
    p_max = float((mx >= best).mean())
    p_max_bootmean = float(((boot - boot.mean(axis=1, keepdims=True)).max(axis=0) >= best).mean())

    print(f"{b.title}, {model}, pool {pool}: N = {N}, {metric}, {int(keep.sum())} rows")
    print(f"  best {names[i_best]} {best:+.4f}; B(N) {mx.mean():+.4f}, C95 {np.percentile(mx, 95):+.4f}, "
          f"Reality Check p {p_max:.3f} (bootstrap-mean centring {p_max_bootmean:.3f})")

    rng = np.random.default_rng(seed + 7)
    curve = {}
    for k in sorted({2, 5, 10, 15, 20, 25, 30, N} & set(range(2, N + 1))):
        vals = [noise[rng.choice(N, size=k, replace=False)].max(axis=0).mean() for _ in range(200)]
        curve[k] = float(np.mean(vals))
        rng.choice(N, size=k, replace=False)  # keeps the random stream of the published curve
    print("  B(N) over random subpools: " + ", ".join(f"{k}: {v:+.4f}" for k, v in curve.items()))

    out = b.out_dir(model) / f"best_of_n_ceiling_{metric}_{pool}.json"
    out.write_text(json.dumps({"stand": b.key, "metric": metric, "pool": pool, "N": N,
                               "best_name": names[i_best], "best_delta": best,
                               "ceiling_mean": float(mx.mean()), "ceiling_p95": float(np.percentile(mx, 95)),
                               "p_max": p_max, "half_width": half, "ceiling_by_N": curve,
                               "p_max_bootmean": p_max_bootmean,
                               "contrast_boot_bias": float(np.mean(boot.mean(axis=1) - obs))},
                              indent=1), encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    ap.add_argument("--metric", default="hard_min")
    ap.add_argument("--pool", choices=["all", "finals"], default="all")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--model", default=MAIN_MODEL)
    args = ap.parse_args()
    run(benchmarks.load(args.benchmark), args.metric, args.pool, args.n_boot, model=args.model)


if __name__ == "__main__":
    main()

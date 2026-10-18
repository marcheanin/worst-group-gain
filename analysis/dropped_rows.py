"""Test examples dropped because of unparsable answers, and the verdicts with and without them (Appendix C).

An example is dropped for every prompt when some prompt of the pool gave no readable answer on it. `query/requery.py`
asked the model again for exactly those (prompt, example) pairs; `data/<benchmark>/requery.json` holds the answers.
An answer that parses on the retry is a transport failure; one that still does not parse is broken (the prompt
produces no label). For each benchmark this script reports, in W with Holm's correction over the pool, the number of
significant contrasts, the best gain and the Reality Check p-value

  as reported     the unparsable examples dropped,
  recovered       transport failures replaced by the answers of the retry, broken answers still dropped,
  all kept        (toxicity only) every example kept, broken answers scored as wrong.

Writes `results/<benchmark>/dropped_rows.json`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.stats import Bootstrap, boot_p, holm, load_preds, load_set, valid_rows


def verdict(b, P, y, c, keep, n_boot=4000):
    names = sorted(n for n in P if not n.startswith("CONTROL:") and n != "seed")
    bs = Bootstrap(y[keep], c[keep], n_boot, np.random.default_rng(0), groups=b.ids)
    o_s, b_s = bs.metrics(P["seed"][keep])
    obs, boot, ps = [], [], []
    for n in names:
        o, bt = bs.metrics(P[n][keep])
        obs.append(o["hard_min"] - o_s["hard_min"])
        boot.append(bt["hard_min"] - b_s["hard_min"])
        ps.append(boot_p(bt, b_s, "hard_min"))
    obs, boot = np.array(obs), np.stack(boot)
    mx = (boot - obs[:, None]).max(axis=0)
    best = float(obs.max())
    return {"rows": int(keep.sum()), "holm": int(holm(np.array(ps)).sum()), "best_gain": best,
            "rc_p": float((mx >= best).mean()), "noise_ceiling": float(np.percentile(mx, 95))}


def run(b):
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y), controls=False)
    rq = json.loads((b.data / "requery.json").read_text(encoding="utf-8"))
    pool = sorted(P)
    keep = valid_rows(P, len(y), pool, verbose=False)
    res = {"dropped": int((~keep).sum()), "total": len(y), "pairs_requeried": rq["n_pairs"], "kinds": rq["kinds"],
           "as_reported": verdict(b, P, y, c, keep)}
    Q = {n: v.copy() for n, v in P.items()}
    for p in rq["pairs"]:
        if p["kind"] == "recovered":
            Q[p["prompt"]][p["row"]] = p["label"]
    res["recovered"] = verdict(b, Q, y, c, valid_rows(Q, len(y), pool, verbose=False))
    if b.key == "toxlang":
        W = {n: np.where(v < 0, 1 - y, v) for n, v in Q.items()}
        res["all_kept"] = verdict(b, W, y, c, np.ones(len(y), bool))
    print(f"{b.title}: dropped {res['dropped']} of {len(y)}, {rq['kinds']}")
    for k in ("as_reported", "recovered", "all_kept"):
        if k in res:
            v = res[k]
            print(f"  {k:12s} {v['rows']} rows: Holm {v['holm']}, best {v['best_gain']:+.3f}, Reality Check p {v['rc_p']:.3f}, "
                  f"noise ceiling {v['noise_ceiling']:+.3f}")
    (b.results / "dropped_rows.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    b = benchmarks.load(ap.parse_args().benchmark)
    b.results.mkdir(parents=True, exist_ok=True)
    run(b)


if __name__ == "__main__":
    main()

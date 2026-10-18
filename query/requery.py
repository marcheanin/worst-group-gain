"""Why were test examples dropped? Query again exactly the (prompt, example) pairs whose answer was unparsable.

    python -m query.requery --benchmark toxlang --dry-run

An answer is written as -1 both when the query failed after its retries (transport) and when the model's answer
cannot be parsed (the prompt itself produces no label). At temperature 0 the second reproduces and the first goes
away on a retry, so only a new query tells them apart. The raw answers go to `data/<benchmark>/requery.json`;
the saved predictions are not touched. `analysis/dropped_rows.py` turns the result into Appendix C.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wgg import benchmarks
from wgg.data import load_texts
from wgg.models import INVALID, MODELS, Worker, load_key, parse_binary
from wgg.stats import load_preds, load_set


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchmarks.add_argument(ap)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--key-var", default=None)
    args = ap.parse_args()
    b = benchmarks.load(args.benchmark)
    y, c, _ = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y), controls=False)
    pool = b.prompts()
    pairs = [(n, int(i)) for n in sorted(P) for i in np.flatnonzero(P[n] < 0)]
    print(f"{b.title}: {len(pairs)} (prompt, example) pairs with an unparsable answer")
    if args.dry_run:
        return
    texts = load_texts(b, b.test_set)
    spec = MODELS[benchmarks.MAIN_MODEL]
    worker = Worker(spec["id"], load_key(args.key_var), spec["provider"], False)
    out = []
    for n, i in pairs:
        row = {"prompt": n, "row": i, "group": int(c[i]), "gold": int(y[i])}
        try:
            answer, _ = worker.ask(pool[n].format(review=texts[i]))
            label = parse_binary(answer)
            out.append({**row, "raw": answer, "label": int(label), "kind": "recovered" if label != INVALID else "content"})
        except Exception as exc:  # noqa: BLE001
            out.append({**row, "raw": None, "label": -1, "kind": "transport", "error": str(exc)[:200]})
    kinds = {k: sum(1 for o in out if o["kind"] == k) for k in ("recovered", "content", "transport")}
    path = b.data / "requery.json"
    path.write_text(json.dumps({"benchmark": b.key, "n_pairs": len(pairs), "kinds": kinds, "pairs": out},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{kinds} -> {path}")


if __name__ == "__main__":
    main()

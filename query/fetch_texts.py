"""Rebuild the texts of the test and validation sets from the public datasets and verify them against the
released labels, groups and fingerprints (see `wgg/data.py`). Needs network access (or a populated Hugging Face
cache) but no API key.

    python -m query.fetch_texts            # all three benchmarks
    python -m query.fetch_texts --benchmark mnli
"""
from __future__ import annotations

import argparse
import json

from wgg import benchmarks
from wgg.data import build_texts, texts_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=list(benchmarks.KEYS), default=None)
    args = ap.parse_args()
    for key in ([args.benchmark] if args.benchmark else benchmarks.KEYS):
        b = benchmarks.load(key)
        for s in ("test", "dev"):
            texts = build_texts(b, s)
            p = texts_path(b, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
            print(f"{key}/{s}: {len(texts)} texts verified -> {p}")


if __name__ == "__main__":
    main()

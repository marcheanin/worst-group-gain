"""Test and validation sets: the released files hold indices, labels and groups, and the texts are rebuilt from
the public datasets and checked against them.

    data/<benchmark>/sets/{test,dev}.json   indices, labels, group ids, fingerprint; no text
    python -m query.fetch_texts --benchmark civil   rebuilds the texts into data/<benchmark>/texts/<set>.json

CivilComments (WILDS version, `pietrolesci/civilcomments-wilds`): `indices` are positions in the official test
split (validation split for the validation set); the group of a comment is the first of the eight identities
annotated with a score of at least 0.5, and 0 (no identity) otherwise. MultiNLI (`nyu-mll/multi_nli`): `indices`
are positions in the list of the matched and then the mismatched validation examples whose label is 0, 1 or 2;
the text is "Premise: ...\\nHypothesis: ...", the label 1 for contradiction, and the group the genre.
Multilingual toxicity (`textdetox/multilingual_toxicity_dataset`): positions in the list of the six language
splits read in the order en, de, ru, ar, hi, am with repeated texts inside a language removed (first occurrence
kept); the label is the field `toxic`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wgg.benchmarks import Benchmark

IDENTITY_VARS = ("male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white")
TOX_LANGS = {1: "en", 2: "de", 3: "ru", 4: "ar", 5: "hi", 6: "am"}


def _civil_rows(split: str):
    from datasets import load_dataset
    df = load_dataset("pietrolesci/civilcomments-wilds")[split].to_pandas()
    texts = [str(t) for t in df["comment_text"].tolist()]
    labels = [1 if float(y) >= 0.5 else 0 for y in df["toxicity"].tolist()]
    mask = df.loc[:, list(IDENTITY_VARS)].to_numpy(dtype=float) >= 0.5
    groups = np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, 0).astype(int).tolist()
    return texts, labels, groups


def _mnli_rows():
    from datasets import load_dataset
    rows = []
    for split in ("validation_matched", "validation_mismatched"):
        for r in load_dataset("nyu-mll/multi_nli", split=split):
            if int(r["label"]) in (0, 1, 2):
                rows.append(r)
    genres = sorted({r["genre"] for r in rows})
    gid = {g: i + 1 for i, g in enumerate(genres)}
    texts = [f"Premise: {r['premise'].strip()}\nHypothesis: {r['hypothesis'].strip()}" for r in rows]
    return texts, [int(r["label"] == 2) for r in rows], [gid[r["genre"]] for r in rows]


def _tox_rows():
    from datasets import load_dataset
    texts, labels, groups = [], [], []
    for gid, lang in TOX_LANGS.items():
        seen = set()
        for r in load_dataset("textdetox/multilingual_toxicity_dataset", split=lang):
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            texts.append(r["text"].strip())
            labels.append(int(r["toxic"]))
            groups.append(gid)
    return texts, labels, groups


def build_texts(b: Benchmark, set_name: str) -> list:
    """The texts of a released set, rebuilt from the public data and verified against the released labels,
    groups and (MultiNLI, toxicity) fingerprint."""
    rec = json.loads(b.set_path(set_name).read_text(encoding="utf-8"))
    if b.key == "civil":
        split = "test" if set_name == "test" else "validation"
        texts, labels, groups = _civil_rows(split)
    else:
        texts, labels, groups = (_mnli_rows if b.key == "mnli" else _tox_rows)()
    idx = rec["indices"]
    out = [texts[i] for i in idx]
    if [labels[i] for i in idx] != rec["labels"] or [groups[i] for i in idx] != rec["cluster_ids"]:
        raise SystemExit(f"{b.key}/{set_name}: labels or groups of the public data differ from the released set")
    if b.key != "civil":
        fp = hashlib.sha256(json.dumps([out, rec["labels"]], ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        if fp != rec["fingerprint"]:
            raise SystemExit(f"{b.key}/{set_name}: fingerprint {fp} differs from the released {rec['fingerprint']}")
    else:
        fp = hashlib.sha256(",".join(str(int(i)) for i in sorted(idx)).encode()).hexdigest()[:12]
        if fp != rec["fingerprint"]:
            raise SystemExit(f"{b.key}/{set_name}: fingerprint {fp} differs from the released {rec['fingerprint']}")
    return out


def texts_path(b: Benchmark, set_name: str) -> Path:
    return b.data / "texts" / f"{set_name}.json"


def load_texts(b: Benchmark, set_name: str) -> list:
    """The texts of a set: from `data/<benchmark>/texts/` when fetched, else rebuilt (and cached there)."""
    p = texts_path(b, set_name)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    texts = build_texts(b, set_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return texts

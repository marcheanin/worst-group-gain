"""The data an optimizer works on, per benchmark: labelled training examples, a labelled validation set to select on,
and, on CivilComments, an unlabelled pool from the target groups.

CivilComments (positions refer to the original train and validation splits of the public dataset; `data/civil/optimizer_data.json`):
  train       S_source, 360 examples, 45 per group x label cell, from comments that mention no identity or the male,
              female or christian identities; U_target, 4,000 comments from the LGBTQ, muslim, other-religion, black
              and white identities, labels hidden (a labelled subset is revealed for the two variants of ProTeGi
              that use target labels). One draw per seed 42, 43 and 44.
  validation  D_dev, 900 examples, 50 per group x label cell, including the comments that mention no identity.
MultiNLI:
  train       36 examples per genre x label cell from the training split, which holds only five of the ten genres
              (fiction, government, slate, telephone, travel), 360 examples; the other five genres, including
              verbatim, are never seen by the search.
  validation  the first 900 examples of the shuffled validation set `dev` (a stratified subsample).
Multilingual toxicity:
  train       36 examples per language x label cell from the same source as the test sets, without the texts of
              `test` and `dev`, 432 examples.
  validation  the first 900 examples of `dev`.
"""
from __future__ import annotations

import json

import numpy as np

from optimizers.api import LabeledSet, UnlabeledSet
from wgg.benchmarks import Benchmark
from wgg.data import TOX_LANGS, _civil_rows, load_texts


def seed_prompt(b: Benchmark) -> str:
    return b.prompts()["seed"]


def civil(b: Benchmark, seed: int):
    """(train, unlabelled target pool, dev, reveal) of CivilComments for a seed."""
    d = json.loads((b.data / "optimizer_data.json").read_text(encoding="utf-8"))
    texts_tr, labels_tr, groups_tr = _civil_rows("train")
    texts_va, _, _ = _civil_rows("validation")
    rec = d["seeds"][str(seed)]

    def labelled(r):
        ids = r["indices"]
        return LabeledSet(texts=[texts_tr[i] for i in ids], labels=list(r["labels"]), group_ids=list(r["groups"]),
                          example_ids=list(ids))

    src, tgt = rec["s_source"], rec["u_target"]
    train = labelled(src)
    unlabeled = UnlabeledSet(texts=[texts_tr[i] for i in tgt["indices"]], group_ids=list(tgt["groups"]),
                             example_ids=list(tgt["indices"]))
    dev = LabeledSet(texts=[texts_va[i] for i in d["dev"]["indices"]], labels=list(d["dev"]["labels"]),
                     group_ids=list(d["dev"]["groups"]))

    def reveal(ids):
        return LabeledSet(texts=[texts_tr[i] for i in ids], labels=[int(labels_tr[i]) for i in ids],
                          group_ids=[int(groups_tr[i]) for i in ids], example_ids=list(ids))

    return train, unlabeled, dev, reveal


def mnli_train(seed: int, group_names: dict, per_cell: int = 36) -> LabeledSet:
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    gid = {v: int(k) for k, v in group_names.items()}
    cells: dict = {}
    for i, (g, lab) in enumerate(zip(ds["genre"], ds["label"])):
        if int(lab) in (0, 1, 2):
            cells.setdefault((g, int(int(lab) == 2)), []).append(i)
    rng = np.random.RandomState(seed)
    rows = [(int(i), key) for key in sorted(cells) for i in rng.choice(cells[key], per_cell, replace=False)]
    texts, labels, groups = [], [], []
    for j in rng.permutation(len(rows)):
        i, (g, y) = rows[j]
        r = ds[i]
        texts.append(f"Premise: {r['premise'].strip()}\nHypothesis: {r['hypothesis'].strip()}")
        labels.append(y)
        groups.append(gid[g])
    return LabeledSet(texts=texts, labels=labels, group_ids=groups)


def toxlang_train(b: Benchmark, seed: int, group_names: dict, per_cell: int = 36) -> LabeledSet:
    from datasets import load_dataset
    used = set(load_texts(b, "test")) | set(load_texts(b, "dev"))
    gid = {v: int(k) for k, v in group_names.items()}
    rng = np.random.RandomState(seed)
    rows = []
    for lang in TOX_LANGS.values():
        ds = load_dataset("textdetox/multilingual_toxicity_dataset", split=lang)
        cells: dict = {}
        for i, (t, y) in enumerate(zip(ds["text"], ds["toxic"])):
            if t in used:
                continue
            cells.setdefault(int(y), []).append(i)
        for y in sorted(cells):
            for i in rng.choice(cells[y], per_cell, replace=False):
                rows.append((ds[int(i)]["text"], y, gid[lang]))
    texts, labels, groups = [], [], []
    for j in rng.permutation(len(rows)):
        t, y, g = rows[j]
        texts.append(t)
        labels.append(y)
        groups.append(g)
    return LabeledSet(texts=texts, labels=labels, group_ids=groups)


def live_loop(b: Benchmark, seed: int):
    """(train, dev) of a live loop on MultiNLI or multilingual toxicity."""
    rec = json.loads(b.set_path("dev").read_text(encoding="utf-8"))
    n = 900
    texts = load_texts(b, "dev")[:n]
    dev = LabeledSet(texts=texts, labels=rec["labels"][:n], group_ids=rec["cluster_ids"][:n])
    names = rec["group_names"]
    train = mnli_train(seed, names) if b.key == "mnli" else toxlang_train(b, seed, names)
    return train, dev

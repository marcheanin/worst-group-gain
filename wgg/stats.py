"""Metrics, the paired stratified bootstrap and Holm's correction, shared by every analysis.

A prompt's predictions on a set are an int16 vector aligned with the set's labels `y` and group ids
`c`; -1 marks an answer the parser could not read. Every contrast is a prompt against the seed on
the same rows, and every resample is shared by all prompts, so contrasts stay paired.
"""
from __future__ import annotations

import json

import numpy as np

from wgg.benchmarks import MAIN_MODEL, Benchmark

METRICS = ["hard_min", "cvar25", "mean_gba", "worst_class", "global_acc"]


def load_set(b: Benchmark, name: str):
    rec = json.loads(b.set_path(name).read_text(encoding="utf-8"))
    return np.asarray(rec["labels"]), np.asarray(rec["cluster_ids"]), rec


def load_preds(b: Benchmark, set_name: str, n=None, controls=True, model: str = MAIN_MODEL) -> dict:
    """Predictions of every prompt on a set, plus two constant predictors as controls.

    The constant answers are scored on the same rows as the prompts so that a table can show what
    silence would score. A name such as `s13:42_ape__soft_min` is stored as `s13__42_ape__soft_min`:
    only the first `__` stands for the colon.
    """
    d = {}
    for f in sorted(b.preds_dir(set_name, model).glob("*.npy")):
        d[f.stem.replace("__", ":", 1)] = np.load(f)
    # A scoring run stopped mid-prompt leaves some vectors longer than others. Rows are scored in
    # the set's own order, so the common prefix lines up across prompts and with the labels.
    if d:
        lens = {len(v) for v in d.values()}
        if len(lens) > 1:
            m = min(lens)
            print(f"ragged matrix on {set_name}: lengths {sorted(lens)} -> cut to the common prefix of "
                  f"{m} rows")
            d = {k: v[:m] for k, v in d.items()}
    if controls and d:
        rows = n or len(next(iter(d.values())))
        d["CONTROL:always_0"] = np.zeros(rows, dtype=np.int16)
        d["CONTROL:always_1"] = np.ones(rows, dtype=np.int16)
    return d


def gba_by_group(pred, y, c, rows=None, groups=()):
    """Balanced accuracy of each group that holds both labels."""
    out = {}
    for g in groups:
        m = (c == g) if rows is None else ((c == g) & rows)
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.any() and neg.any():
            out[g] = 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())
    return out


def metric(name, pred, y, c, rows=None, groups=()):
    """One metric on the observed rows: hard_min is the worst-group balanced accuracy W, cvar25 the
    mean of the floor(K/4) lowest groups, mean_gba the mean over groups, worst_class min(TPR, TNR)
    over all rows, global_acc plain accuracy."""
    sel = slice(None) if rows is None else rows
    if name == "worst_class":
        p, yy = pred[sel], y[sel]
        return float(min((p[yy == 0] == 0).mean(), (p[yy == 1] == 1).mean()))
    if name == "global_acc":
        return float((pred[sel] == y[sel]).mean())
    g = gba_by_group(pred, y, c, rows, groups)
    if not g:
        return float("nan")
    v = np.sort(list(g.values()))
    return float({"hard_min": v[:1], "cvar25": v[: max(1, len(v) // 4)], "mean_gba": v}[name].mean())


def cell_index(y, c, keep=None):
    """Row indices of every group x label cell, optionally restricted to the rows in `keep`."""
    ok = np.ones(len(y), bool) if keep is None else keep
    return {(int(g), int(l)): np.flatnonzero((c == g) & (y == l) & ok)
            for g in sorted(set(c.tolist())) for l in (0, 1)}


def valid_rows(preds, n, names=None, verbose=True):
    """Rows on which every prompt gave a parsable answer.

    An unparsable answer is dropped rather than scored, and dropped for all prompts at once, so that
    every comparison stays paired.
    """
    pool = [k for k in preds if not k.startswith("CONTROL:")] if names is None else list(names)
    bad, per_prompt = np.zeros(n, bool), {}
    for name in pool:
        b = preds[name] < 0
        if b.any():
            per_prompt[name] = int(b.sum())
        bad |= b
    if per_prompt and verbose:
        worst = sorted(per_prompt.items(), key=lambda kv: -kv[1])[:4]
        print(f"unparsable rows dropped: {int(bad.sum())}/{n} ({100 * bad.mean():.3f}%) from "
              f"{len(per_prompt)}/{len(pool)} prompts; worst: " + ", ".join(f"{k} {v}" for k, v in worst))
    return ~bad


class Bootstrap:
    """One resampling scheme shared by every prompt and metric.

    Rows are drawn with replacement inside each group x label cell, and the same draws are reused
    for all prompts. Cells are concatenated in a fixed order, so each cell keeps a fixed slice of the
    resampled array and every metric is computed by slicing.
    """

    def __init__(self, y, c, n_boot, rng, cells=None, groups=()):
        self.y, self.c, self.groups = y, c, tuple(groups)
        cells = cells or cell_index(y, c)
        self.cells = {k: v for k, v in cells.items() if len(v)}
        self.slices, start = {}, 0
        cols = []
        for k, rows in self.cells.items():
            cols.append(rng.choice(rows, size=(n_boot, len(rows)), replace=True))
            self.slices[k] = slice(start, start + len(rows))
            start += len(rows)
        self.idx = np.concatenate(cols, axis=1)  # (n_boot, n)
        self.n_boot = n_boot

    def cell_rates(self, pred):
        """Per cell: share of rows predicted as the cell's own label, observed and per resample."""
        obs, boot = {}, {}
        for k, rows in self.cells.items():
            g, l = k
            obs[k] = float((pred[rows] == l).mean())
            boot[k] = (pred[self.idx[:, self.slices[k]]] == l).mean(axis=1)
        return obs, boot

    def metrics(self, pred, groups=None):
        """Observed value and bootstrap distribution of every metric at once."""
        groups = self.groups if groups is None else groups
        obs_cell, boot_cell = self.cell_rates(pred)
        n = {k: len(v) for k, v in self.cells.items()}
        all_groups = set(x[0] for x in self.cells)

        def assemble(get):
            g_vals = {g: 0.5 * (get((g, 1)) + get((g, 0))) for g in groups
                      if (g, 1) in self.cells and (g, 0) in self.cells}
            arr = np.sort(np.stack(list(g_vals.values())), axis=0)
            k = max(1, len(arr) // 4)
            tot = sum(n[c_] for c_ in self.cells)
            acc = sum(get(c_) * n[c_] for c_ in self.cells) / tot
            pos = sum(get((g, 1)) * n[(g, 1)] for g in all_groups) / sum(n[(g, 1)] for g in all_groups)
            neg = sum(get((g, 0)) * n[(g, 0)] for g in all_groups) / sum(n[(g, 0)] for g in all_groups)
            return {"hard_min": arr[0], "cvar25": arr[:k].mean(axis=0), "mean_gba": arr.mean(axis=0),
                    "worst_class": np.minimum(pos, neg), "global_acc": acc}

        return assemble(lambda k: obs_cell[k]), assemble(lambda k: boot_cell[k])


def delta_ci(boot_a, boot_b, obs_a, obs_b, name):
    """Observed contrast a - b and its 95% percentile interval."""
    d = boot_a[name] - boot_b[name]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(obs_a[name] - obs_b[name]), float(lo), float(hi)


def boot_p(boot_a, boot_b, name):
    """Two-sided bootstrap p-value of one contrast, floored at the resampling resolution."""
    d = boot_a[name] - boot_b[name]
    tail = min((d <= 0).mean(), (d >= 0).mean())
    return float(max(2 * tail, 1.0 / len(d)))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni; the boolean vector of rejections in input order."""
    order = np.argsort(pvals)
    m, out, blocked = len(pvals), np.zeros(len(pvals), bool), False
    for rank, i in enumerate(order):
        if blocked or pvals[i] > alpha / (m - rank):
            blocked = True
        else:
            out[i] = True
    return out

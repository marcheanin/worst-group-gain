"""The number an optimizer selects by: a smoothed worst-group balanced accuracy of a prompt on labelled data.

For a prompt's predictions on a labelled set with group ids, `evaluate` returns the quantities every method
reads. With the settings used in all runs (the same for every method):

  * an example with an unparsable answer (-1) counts as an error for both classes; a prompt is rejected
    (fitness -1e9) when more than 2% of its answers are unparsable or when it labels fewer than 2% or more than
    98% of the parsable examples as positive;
  * the balanced accuracy of a group is (TPR + TNR) / 2, over groups with at least 10 examples of each class
    (the group of comments that mention no identity, id 0, is left out);
  * each group's TPR and TNR are shrunk toward the overall TPR and TNR with a prior worth 40 examples, which
    gives the shrunk balanced accuracies;
  * `R_soft_min_gba` is the softmin of the shrunk balanced accuracies, -tau log mean exp(-x / tau) with
    tau = 0.10, and the fitness is that plus 0.01 times their mean;
  * `R_worst_gba` is the minimum of the shrunk balanced accuracies, `R_gba_mean` the mean of the raw balanced
    accuracies, `R_global` plain accuracy.

The selection statistic of a live loop (Section 3 and Appendix H) replaces `R_soft_min_gba` by one of the
others: `hard_min` by `R_worst_gba`, `mean_gba` by `R_gba_mean`, `global` by `R_global`.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

REJECT = -1e9
PRIOR_WEIGHT = 40.0
TAU = 0.10
EPS_GLOBAL = 0.01
MIN_CELL = 10
MAX_INVALID = 0.02
MIN_POS_RATE = 0.02
EXCLUDE_GROUPS = {0}


def softmin(values, tau: float) -> float:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return 0.0
    z = -v / tau
    zmax = float(z.max())
    return float(-tau * (np.log(np.mean(np.exp(z - zmax))) + zmax))


def evaluate(preds, labels, groups, prompt: str) -> Dict[str, Any]:
    p, y, g = np.asarray(preds), np.asarray(labels), np.asarray(groups)
    out: Dict[str, Any] = {}
    invalid = p < 0
    out["invalid_rate"] = float(invalid.mean()) if len(p) else 0.0
    valid = p[~invalid]
    out["pred_pos_rate"] = float((valid == 1).mean()) if len(valid) else 0.0
    out["R_global"] = float((p == y).mean()) if len(p) else 0.0
    out["toxic_recall"] = float((p[y == 1] == 1).mean()) if (y == 1).any() else 0.0

    accs = {int(c): float((p[g == c] == y[g == c]).mean()) for c in np.unique(g)}
    if accs:
        out["R_worst_group"] = float(min(accs.values()))

    # balanced accuracy of the groups with enough examples of both classes
    raw = {}
    for c in np.unique(g):
        c = int(c)
        if c in EXCLUDE_GROUPS:
            continue
        m = g == c
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.sum() < MIN_CELL or neg.sum() < MIN_CELL:
            continue
        raw[c] = 0.5 * (float((p[pos] == 1).mean()) + float((p[neg] == 0).mean()))
    shrunk = {}
    if raw:
        grand_tpr = float((p[y == 1] == 1).mean()) if (y == 1).any() else 0.5
        grand_tnr = float((p[y == 0] == 0).mean()) if (y == 0).any() else 0.5
        for c in raw:
            m = g == c
            pos, neg = m & (y == 1), m & (y == 0)
            tpr = (float((p[pos] == 1).sum()) + PRIOR_WEIGHT * grand_tpr) / (float(pos.sum()) + PRIOR_WEIGHT)
            tnr = (float((p[neg] == 0).sum()) + PRIOR_WEIGHT * grand_tnr) / (float(neg.sum()) + PRIOR_WEIGHT)
            shrunk[c] = 0.5 * (tpr + tnr)
        out["R_worst_gba"] = float(min(raw.values()))  # raw; replaced by the shrunk minimum for an admitted prompt
        out["R_gba_mean"] = float(np.mean(list(raw.values())))
        out["R_gba_mean_shrunk"] = float(np.mean(list(shrunk.values())))

    def rejected(reason):
        return {"fitness": REJECT, "combined_score": REJECT, "base_score": REJECT, "reject_reason": reason, **out}

    if out["invalid_rate"] > MAX_INVALID:
        return rejected("invalid_rate")
    if (set(np.unique(y).tolist()) <= {0, 1}
            and not (MIN_POS_RATE <= out["pred_pos_rate"] <= 1.0 - MIN_POS_RATE)):
        return rejected("degenerate")
    if not shrunk:
        return rejected("gba_empty")

    out["R_worst_gba"] = float(min(shrunk.values()))
    out["R_soft_min_gba"] = softmin(shrunk.values(), TAU)
    base = out["R_soft_min_gba"] + EPS_GLOBAL * out["R_gba_mean_shrunk"]
    return {"fitness": float(base), "combined_score": float(base), "base_score": float(base),
            "reject_reason": None, **out}

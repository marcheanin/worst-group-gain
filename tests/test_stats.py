"""The statistics core on small synthetic cases with known answers."""
import numpy as np

from wgg.stats import Bootstrap, boot_p, cell_index, delta_ci, gba_by_group, holm, metric, valid_rows


def test_holm_matches_the_textbook_procedure():
    p = np.array([0.001, 0.04, 0.03, 0.0004])
    # sorted: 0.0004 (<0.05/4), 0.001 (<0.05/3), 0.03 (>0.05/2): stops there
    assert holm(p).tolist() == [True, False, False, True]
    assert holm(np.array([0.2, 0.3])).sum() == 0
    assert holm(np.array([0.01, 0.02])).sum() == 2


def test_balanced_accuracy_and_worst_group():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    c = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    pred = np.array([1, 1, 0, 1, 1, 0, 0, 0])  # group 1: TPR 1, TNR .5 -> .75; group 2: TPR .5, TNR 1 -> .75
    g = gba_by_group(pred, y, c, groups=(1, 2))
    assert g == {1: 0.75, 2: 0.75}
    pred2 = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # group 1: .5; group 2: 1.0
    assert metric("hard_min", pred2, y, c, groups=(1, 2)) == 0.5
    assert metric("mean_gba", pred2, y, c, groups=(1, 2)) == 0.75
    assert metric("worst_class", pred2, y, c, groups=(1, 2)) == 0.5  # TNR = 2 of 4


def test_unparsable_rows_are_dropped_for_every_prompt():
    preds = {"seed": np.array([1, 0, 1, 0]), "a": np.array([1, -1, 1, 0])}
    keep = valid_rows(preds, 4, verbose=False)
    assert keep.tolist() == [True, False, True, True]


def test_bootstrap_contrast_of_identical_prompts_is_exactly_zero():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 200)
    c = np.tile(np.repeat([1, 2], 100), 2)
    pred = np.where(rng.random(400) < 0.8, y, 1 - y)
    bs = Bootstrap(y, c, 500, np.random.default_rng(1), groups=(1, 2))
    o, b = bs.metrics(pred)
    d, lo, hi = delta_ci(b, b, o, o, "hard_min")
    assert d == lo == hi == 0.0
    assert boot_p(b, b, "hard_min") >= 1.0     # no evidence against zero (the p-value is not capped at 1 here)


def test_bootstrap_is_paired_and_detects_a_real_difference():
    rng = np.random.default_rng(0)
    n = 1000
    y = np.repeat([0, 1], n // 2)
    c = np.tile(np.repeat([1, 2], n // 4), 2)
    base = np.where(rng.random(n) < 0.7, y, 1 - y)
    better = base.copy()
    flip = (rng.random(n) < 0.15) & (better != y)   # repairs 15% of the errors on every group
    better[flip] = y[flip]
    bs = Bootstrap(y, c, 1000, np.random.default_rng(2), groups=(1, 2))
    ob, bb = bs.metrics(base)
    oc, bc = bs.metrics(better)
    d, lo, hi = delta_ci(bc, bb, oc, ob, "mean_gba")
    assert lo > 0 and d > 0


def test_cells_are_resampled_inside_group_label_cells():
    y = np.array([0, 0, 1, 1, 0, 1])
    c = np.array([1, 1, 1, 1, 2, 2])
    cells = cell_index(y, c)
    assert {k: v.tolist() for k, v in cells.items()} == {(1, 0): [0, 1], (1, 1): [2, 3], (2, 0): [4], (2, 1): [5]}

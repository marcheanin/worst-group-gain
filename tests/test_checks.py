"""The seven checks on synthetic predictions with planted effects."""
import numpy as np

from wgg.checks import format_report, run_checks


def make(n_per_cell=250, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.repeat([0, 1], n_per_cell), k)
    g = np.repeat(np.arange(1, k + 1), 2 * n_per_cell)
    base_acc = np.array([0.8, 0.85, 0.9, 0.7])[g - 1]                 # group 4 is the worst group
    easy = rng.random(len(y))
    seed_pred = np.where(easy < base_acc, y, 1 - y)
    return y, g, easy, base_acc, seed_pred


def test_a_planted_gain_on_the_worst_group_is_found_and_the_minimum_understates_it():
    y, g, easy, base_acc, seed_pred = make()
    preds = {"seed": seed_pred}
    boost = np.where(g == 4, 0.2, 0.0)                                # helps only the worst group
    preds["helps"] = np.where(easy < base_acc + boost, y, 1 - y)
    rng = np.random.default_rng(5)
    for i in range(6):                                                # prompts that differ from the seed only by noise
        flip = rng.random(len(y)) < 0.02
        preds[f"noise{i}"] = np.where(flip, 1 - seed_pred, seed_pred)
    rep = run_checks(preds, y, g, n_boot=800, n_random=5, random_boot=300)
    assert rep["check6"]["target_group"] == 4
    assert "helps" in [r["prompt"] for r in rep["check6"]["gains"] if r["holm"]]
    assert rep["check3"]["best_prompt"] == "helps"
    # the gain of the target group is large, but the worst-group number reports only part of it: the minimum moves
    assert rep["check6"]["best_target_gain"] > 2 * rep["check6"]["best_worst_group_gain"]
    assert rep["check6"]["hidden"] == ["helps"]
    assert "Reality Check" in format_report(rep)


def test_without_any_effect_the_best_of_many_is_within_chance():
    y, g, easy, base_acc, seed_pred = make(seed=3)
    rng = np.random.default_rng(9)
    preds = {"seed": seed_pred}
    for i in range(20):
        preds[f"p{i}"] = np.where(rng.random(len(y)) < 0.03, 1 - seed_pred, seed_pred)
    rep = run_checks(preds, y, g, n_boot=800, n_random=5, random_boot=300)
    assert rep["check3"]["reality_check_p"] > 0.05
    assert rep["check3"]["chance_baseline"] > 0


def test_second_model_check_reports_a_reversed_gain():
    y, g, easy, base_acc, seed_pred = make(seed=1)
    boost = np.where(g == 4, 0.12, 0.0)
    helps = np.where(easy < base_acc + boost, y, 1 - y)
    hurts_elsewhere = np.where(easy < base_acc - boost, y, 1 - y)
    main = {"seed": seed_pred, "helps": helps}
    other = {"seed": seed_pred, "helps": hurts_elsewhere}
    rep = run_checks(main, y, g, second=other, n_boot=800, n_random=3, random_boot=300)
    assert rep["check7"]["significant_on_main_model"] == ["helps"]
    assert rep["check7"]["transfers"] == []
    assert rep["check7"]["reversed"] == ["helps"]

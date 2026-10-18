"""The fitness, the prompt contract, the parser and a mocked run of every method (no network, no key)."""
import numpy as np

from optimizers import profile
from optimizers.api import Budget, LabeledSet, Task, UnlabeledSet, score_prompt_on_set
from optimizers.contract import check_prompt_contract, ensure_prompt_contract
from optimizers.evoprompt import EvoPrompt
from optimizers.fitness import evaluate, softmin
from optimizers.gepa import GEPA
from optimizers.gpo import GPO
from optimizers.llm import MockCompleter, MockScorer
from optimizers.protegi import ProTeGi
from optimizers.ape import APE
from wgg.models import INVALID, parse_binary

SEED = ("<System><DynamicRules>\n4. When unsure, prefer 0 (non-toxic).\n</DynamicRules></System>\n"
        "Output Label: 0 or Label: 1.\nComment: {review}\n")


def toy_data(n=600, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    g = rng.randint(1, 5, size=n)
    return LabeledSet(texts=[f"text {i}" for i in range(n)], labels=y.tolist(), group_ids=g.tolist())


def test_softmin_is_between_minimum_and_mean():
    v = [0.6, 0.7, 0.9]
    assert min(v) < softmin(v, 0.1) < np.mean(v)
    assert abs(softmin(v, 1e-6) - min(v)) < 1e-5


def test_fitness_rejects_unparsable_and_degenerate_prompts():
    d = toy_data()
    y, g = np.array(d.labels), np.array(d.group_ids)
    good = np.where(np.random.RandomState(1).rand(len(y)) < 0.8, y, 1 - y)
    assert evaluate(good, y, g, "p")["reject_reason"] is None
    broken = good.copy()
    broken[:30] = INVALID                                  # 5% unparsable > 2%
    assert evaluate(broken, y, g, "p")["reject_reason"] == "invalid_rate"
    assert evaluate(np.zeros_like(y), y, g, "p")["reject_reason"] == "degenerate"
    assert evaluate(good, y, np.zeros_like(g), "p")["reject_reason"] == "gba_empty"


def test_fitness_value_on_a_hand_computed_case():
    # two groups with 10 examples of each class; group 1: TPR 1, TNR .8 -> .9; group 2: TPR .6, TNR 1 -> .8
    y = np.tile(np.repeat([1, 0], 10), 2)
    g = np.repeat([1, 2], 20)
    p = y.copy()
    p[10:12] = 1                      # group 1: two negatives called positive
    p[20 + 6:20 + 10] = 0             # group 2: four of ten positives missed
    r = evaluate(p, y, g, "p")
    assert r["reject_reason"] is None
    assert abs(r["R_gba_mean"] - 0.85) < 1e-9
    assert r["R_worst_gba"] < r["R_gba_mean"] and r["R_worst_gba"] > 0.7


def test_prompt_contract():
    assert check_prompt_contract(SEED).ok
    assert check_prompt_contract("no placeholder, output 0 or 1").reason == "missing_placeholder"
    assert check_prompt_contract("broken {review} {oops} 0 1 output").reason.startswith("format_error")
    wrapped = ensure_prompt_contract("classify carefully", seed_fallback=SEED)
    assert "classify carefully" in wrapped and "{review}" in wrapped and check_prompt_contract(wrapped).ok
    assert check_prompt_contract(ensure_prompt_contract("", seed_fallback=SEED)).ok
    assert ensure_prompt_contract("no label words {oops}") == profile.active().last_resort


def test_parser_reads_labels_and_never_defaults_to_zero():
    assert parse_binary("Label: 1") == 1 and parse_binary("0") == 0 and parse_binary("toxic") == 1
    assert parse_binary("I cannot say") == INVALID and parse_binary("") == INVALID


def test_every_method_runs_on_mock_models():
    profile.set_active(profile.CIVIL)
    train, dev = toy_data(360, 1), toy_data(300, 2)
    pool = UnlabeledSet(texts=[f"unlabelled {i}" for i in range(200)], example_ids=list(range(200)))

    def task(unlabeled=None):
        return Task(train=train, dev=dev, unlabeled=unlabeled or UnlabeledSet(texts=[]), scorer=MockScorer(),
                    optimizer_llm=MockCompleter(SEED), budget=Budget(), seed_prompt=SEED, meta={"seed": 42})

    for opt, pool_needed in ((APE(), False), (APE(n_unlabeled=12), True), (GPO(), True),
                             (ProTeGi(rounds=1, minibatch_size=32), False),
                             (EvoPrompt(mode="ga", population_size=3, generations=1, dev_subsample=100), False),
                             (EvoPrompt(mode="de", population_size=3, generations=1, dev_subsample=100), False),
                             (GEPA(reflections=2, dev_subsample=100), False)):
        r = opt.run(task(pool if pool_needed else None))
        assert check_prompt_contract(r.best_prompt).ok, type(opt).__name__
        assert r.all_candidates and r.scorer_calls > 0


def test_selection_statistic_can_be_swapped_by_wrapping_the_fitness():
    from optimizers import fitness
    from optimizers.run import ArchiveTap
    d = toy_data()
    tap = ArchiveTap("global")
    tap.install()
    try:
        m = score_prompt_on_set(MockScorer(), SEED, d)
    finally:
        tap.uninstall()
    assert m["R_soft_min_gba"] == m["R_global"]            # selected by plain accuracy
    assert len(tap.records) == 1 and fitness.evaluate(np.array([0, 1]), [0, 1], [1, 1], "x") is not None

"""The numbers printed in the paper's tables, read from `results/` (run `reproduce.py` to regenerate them).

Each assertion names the table or section it comes from and uses the rounding of the paper.
"""
import json

import pytest

from wgg import benchmarks


def res(key, name):
    p = benchmarks.results_root() / key / name
    if not p.is_file():
        pytest.skip(f"{p} not generated (run reproduce.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def top(name):
    p = benchmarks.results_root() / name
    if not p.is_file():
        pytest.skip(f"{p} not generated (run reproduce.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def r(x, n=3):
    return round(x, n)


# ---- Table 1 and Table 2: the reported gain against chance --------------------------------------------------------
def test_table1_test_examples_and_seed_worst_group():
    for key, used, seed_w in (("civil", 3584, 0.665), ("mnli", 4991, 0.838), ("toxlang", 2907, 0.721)):
        d = res(key, "power_summary.json")
        assert d["n_used"] == used
        assert r(d["truth_values"]["seed"]["hard_min"]) == seed_w


def test_table2_worst_group_block():
    expect = {"civil": dict(N=36, width=0.050, dist=0, sig=0, best=0.019, B=0.027, C95=0.050, p=0.74, fp=0.74),
              "mnli": dict(N=33, width=0.039, dist=13, sig=2, best=0.038, B=0.014, C95=0.030, p=0.014, fp=0.013),
              "toxlang": dict(N=20, width=0.060, dist=2, sig=0, best=0.034, B=0.019, C95=0.044, p=0.13, fp=0.12)}
    for key, e in expect.items():
        d = res(key, "power_summary.json")["resolvable"]["hard_min"]
        assert d["of"] == e["N"] and r(d["ci_width"]) == e["width"]
        assert d["resolved"] == e["dist"] and d["resolved_holm"] == e["sig"]
        allp = res(key, "best_of_n_ceiling_hard_min_all.json")
        fin = res(key, "best_of_n_ceiling_hard_min_finals.json")
        assert r(allp["best_delta"]) == e["best"] and r(allp["ceiling_mean"]) == e["B"] and r(allp["ceiling_p95"]) == e["C95"]
        assert r(allp["p_max"], 2 if e["p"] > 0.1 else 3) == e["p"]
        assert r(fin["p_max"], 2 if e["fp"] > 0.1 else 3) == e["fp"]


def test_table2_worst_class_and_target_group_blocks():
    for key, wc in (("civil", 31), ("mnli", 28), ("toxlang", 10)):
        assert res(key, "power_summary.json")["resolvable"]["worst_class"]["resolved_holm"] == wc
    for key, finals, sig, best in (("civil", 25, 0, 0.030), ("mnli", 22, 2, 0.038), ("toxlang", 9, 2, 0.061)):
        d = res(key, "target_group_contrast.json")
        assert d["n_finals"] == finals and d["n_holm_target"] == sig and r(d["best_final"]["delta"]) == best


# ---- Section 5: when the minimum moves -----------------------------------------------------------------------------
def test_section5_the_hidden_gain_on_amharic():
    d = res("toxlang", "target_group_contrast.json")["finals"]
    g43, g42 = d["s15:43_gepa__hard_min"], d["s15:42_gepa__hard_min"]
    assert r(g43["delta"]) == 0.061 and r(g43["hard_min"]["delta"]) == 0.024 and g43["holm"]
    assert r(g42["delta"]) == 0.055 and [r(v) for v in g42["ci"]] == [0.019, 0.095]
    assert r(g42["hard_min"]["delta"]) == 0.034 and not g42["hard_min"]["holm"]
    ops = res("toxlang", "target_group_operating_points.json")["models"]["gemma-3-12b-it"]
    assert r(ops["points"]["s15:43_gepa__hard_min"]["gba"]) - r(ops["summary"]["best_edit_gba"]) == pytest.approx(0.051, abs=1e-9)
    assert ops["summary"]["n_finals_above_curve"] == 4


# ---- Section 6 / Table 3: the stability of the worst group -----------------------------------------------------------
def test_table3_replication_and_stability():
    expect = {"civil": (60.3, 68.5, 89.0, 99.2), "mnli": (82.0, 92.9, 97.1, 98.4), "toxlang": (86.5, None, 94.8, 95.6)}
    for key, (w, cvar, mean, wc) in expect.items():
        s = res(key, "statistic_reproducibility_n100.json")["statistics"]
        assert r(s["hard_min"]["reproduced_share_pct"], 1) == w
        if cvar:
            assert r(s["cvar25"]["reproduced_share_pct"], 1) == cvar
        assert r(s["mean_gba"]["reproduced_share_pct"], 1) == mean and r(s["worst_class"]["reproduced_share_pct"], 1) == wc
    soft = [v["reproduced_share_pct"] for k, v in res("civil", "statistic_reproducibility_n100.json")["statistics"].items() if k.startswith("softmin")]
    assert (r(min(soft), 1), r(max(soft), 1)) == (86.8, 89.1)
    for key, stay, bias, groups in (("civil", 0.47, 0.0131, 4), ("mnli", 0.91, 0.0012, 1), ("toxlang", 0.93, 0.0006, 2)):
        d = res(key, "argmin.json")
        assert r(d["argmin_stay_median"], 2) == stay and r(d["argmin_bias_median"], 4) == bias and len(d["argmin_groups"]) == groups


# ---- Section 6: the number of groups --------------------------------------------------------------------------------
def test_number_of_groups():
    mnli = {k: res("mnli", f"regroup_pseudo_hard_min_{k}g_{reps}rep.json") for k, reps in ((2, 50), (4, 50), (10, 200))}
    import numpy as np
    assert [float(np.median([x["holm"] for x in mnli[k]["per_rep"]])) for k in (2, 4, 10)] == [12, 7, 0]
    for key, ks in (("civil", ((2, 50), (4, 50), (8, 200))), ("toxlang", ((2, 50), (3, 50), (6, 200)))):
        for k, reps in ks:
            d = res(key, f"regroup_pseudo_hard_min_{k}g_{reps}rep.json")
            assert float(np.median([x["holm"] for x in d["per_rep"]])) == 0


# ---- Section 7 and Appendix J: other task models -----------------------------------------------------------------------
def test_transfer_of_the_four_gains():
    g43, g42 = "s15:43_gepa__hard_min", "s15:42_gepa__hard_min"
    a43, a42 = "s13:43_ape__global", "s13:42_gepa__hard_min"
    expect = {
        ("toxlang", "gpt-4o-mini"): {g43: -0.034, g42: -0.002},
        ("toxlang", "qwen3-235b"): {g43: -0.020, g42: -0.042},
        ("mnli", "gpt-4o-mini"): {a43: -0.034, a42: -0.024},
        ("mnli", "qwen3-235b"): {a43: -0.034, a42: 0.000},
    }
    for (key, model), gains in expect.items():
        d = res(key, f"transfer_{model}.json")
        for prompt, delta in gains.items():
            row = d["other"][prompt] if key == "toxlang" else d["other"][prompt]["target"]
            assert r(row["delta"]) == delta
    assert res("mnli", "transfer_gpt-4o-mini.json")["other"][a43]["target"]["holm"]
    assert res("mnli", "transfer_qwen3-235b.json")["other"][a43]["target"]["holm"]
    assert not res("mnli", "transfer_qwen3-235b.json")["other"][a42]["target"]["holm"]
    assert not any(v["holm"] for v in res("toxlang", "transfer_gpt-4o-mini.json")["other"].values())


def test_accuracy_gate_of_the_further_models():
    q = top("transfer_gate_qwen3-235b.json")
    assert q["toxlang"]["pass"] and q["mnli"]["pass"]
    assert r(q["toxlang"]["mean_ba_third"]) == 0.833 and r(q["mnli"]["mean_ba_third"]) == 0.933
    assert r(q["toxlang"]["spearman_group_order"], 2) == 1.0 and r(q["mnli"]["spearman_group_order"], 2) == 0.94
    s = top("transfer_gate_qwen3-30b.json")
    assert s["mnli"]["pass"] and not s["toxlang"]["pass"] and r(s["toxlang"]["target_ba_third"]) == 0.623


def test_table_of_the_two_models_under_the_seed_prompt():
    d = top("second_model_seed_table.json")["benchmarks"]
    expect = {"toxlang": (0.835, 0.838, 0.721, 0.705, 1.0), "mnli": (0.897, 0.908, 0.838, 0.888, 0.55),
              "civil": (0.717, 0.721, 0.665, 0.688, 0.74)}
    for key, (m1, m2, t1, t2, rho) in expect.items():
        s = d[key]["main_rows"]["summary"]
        assert (r(s["mean_ba_gemma"]), r(s["mean_ba_gpt"]), r(s["target_ba_gemma"]), r(s["target_ba_gpt"])) == (m1, m2, t1, t2)
        assert r(s["spearman_rho_tie_robust"], 2) == rho


def test_second_model_on_civilcomments():
    g = res("civil", "gpt-4o-mini/best_of_n_ceiling_hard_min_all.json")
    assert abs(g["best_delta"] - 0.0125) < 1e-9 and r(g["ceiling_mean"]) == 0.024 and r(g["p_max"], 2) == 0.81
    s = res("civil", "gpt-4o-mini/statistic_reproducibility_n100.json")["statistics"]
    assert r(s["hard_min"]["reproduced_share_pct"], 0) == 37 and abs(s["mean_gba"]["reproduced_share_pct"] - 87.5) < 0.1
    assert r(res("civil", "gpt-4o-mini/argmin.json")["argmin_stay_median"], 2) == 0.52
    rel = res("civil", "target_rho_reliability.json")["rows_by"]
    assert (r(rel["target"]["rho_cross"], 2), r(rel["target"]["rel_gemma"], 2), r(rel["target"]["rel_gpt"], 2)) == (0.04, 0.38, 0.0)


# ---- Section 8 and Appendices B, E, G, H, I, M ------------------------------------------------------------------------
def test_length_penalty_decides_the_mnli_protocol_lottery():
    d = res("mnli", "lottery_summary_hard_min.json")["protocols"]
    off = [v["win_holm"] for k, v in d.items() if k.split("|")[2] == "0.0"]
    on = [v["win_holm"] for k, v in d.items() if k.split("|")[2] == "0.00025"]
    assert len(off) == len(on) == 36 and r(sum(off) / 36, 2) == 0.75 and sum(on) == 0
    dec = res("mnli", "lottery_decompose_hard_min.json")["win_holm"]
    assert r(100 * dec["length penalty"]["share"], 1) == 92.2


def test_wilds_metric_on_civilcomments():
    d = res("civil", "wilds_wga16.json")
    assert r(d["seed_wga16"]) == 0.530 and d["seed_worst_cell"] == "black|0"
    p = d["pool"]
    assert r(p["mean_ci_width"]) == 0.079 and p["resolved_95"] == 19 and p["holm"] == 11 and p["wins_holm"] == 0
    assert r(p["best_delta"]) == 0.065 and r(p["ceiling"]["observed"]["mean"]) == 0.042 and r(p["ceiling"]["observed"]["p95"]) == 0.080
    assert r(p["ceiling"]["observed"]["p_noise"]) == 0.146 and r(d["spearman_wga16_vs_share_toxic"], 2) == -0.90


def test_selection_inside_the_optimizer_excludes_a_regret_above_0_011():
    d = res("toxlang", "archive_regret.json")["summary"]["hard_min"]
    assert r(d["mean_naive_regret"]) == 0.021 and d["n_holm"] == 0 and d["crossfit_ci"][1] < 0.0115
    assert abs(d["mean_crossfit_regret"]) < 0.002


def test_leverage_and_the_ceiling_calibrated_on_edits():
    for key, real, rand in (("civil", 5.26, 5.97), ("mnli", 2.29, 3.64)):
        d = res(key, "predictability.json")["leverage_main_rows"]
        assert (r(d["leverage_real"], 2), r(d["leverage_pseudo"], 2)) == (real, rand)
    t = res("toxlang", "predictability.json")
    assert (r(t["leverage_before_search"]["leverage_real"], 2), r(t["leverage_main_rows"]["leverage_real"], 2)) == (2.51, 2.54)
    assert r(t["edits_ceiling"]["extrapolated"]["p95"]) == 0.031
    for key, ratio in (("civil", 1.61), ("mnli", 1.28), ("toxlang", 1.35)):
        ev = res(key, "predictability.json")["edits_vs_finals"]
        assert r(ev["finals"]["half_width"] / ev["edits"]["half_width"], 2) == ratio


def test_live_loops():
    c, m = res("civil", "live_loops.json"), res("mnli", "live_loops.json")
    assert c["returned_seed"] == 8 and m["returned_seed"] == 5
    assert m["significant_after_holm"]["hard_min"] == 2 and m["significant_after_holm"]["cvar25"] == 11
    assert r(100 * m["design"]["share"]["method"], 1) == 52.4 and r(100 * c["design"]["share"]["method"], 1) == 20.2
    assert m["ranking_cvar25"]["ape"]["holm_win"] == 9 and c["ranking_cvar25"]["ape"]["point_win"] == 0
    assert (c["reselection"]["winner_changes"], m["reselection"]["winner_changes"]) == (13, 14)
    t = res("toxlang", "live_loops_ranking.json")
    assert [r(t[k]["mean_gain"]) for k in ("ape", "apo", "evoprompt_de", "gepa")] == [0.010, -0.009, -0.001, 0.019]


def test_dropped_rows():
    expect = {"civil": (16, 0, 0.020, 0.725), "mnli": (9, 2, 0.038, 0.010), "toxlang": (93, 0, 0.032, 0.172)}
    for key, (dropped, holm_n, best, p) in expect.items():
        d = res(key, "dropped_rows.json")
        assert d["dropped"] == dropped and d["recovered"]["holm"] == holm_n
        assert r(d["recovered"]["best_gain"]) == best and r(d["recovered"]["rc_p"]) == p
    assert r(res("toxlang", "dropped_rows.json")["all_kept"]["rc_p"]) == 0.169

# worst-group-gain

Code, data and experiments for the paper *Inflated, Deflated, Model-Specific: What a Worst-Group Gain from Prompt
Optimization Measures*.

Prompt optimizers are compared by the worst-group gain of their best prompt. The paper shows that this number rests on
two selections, the best of many prompts and the worst of several groups, and that it errs in both directions: on
CivilComments the best of 36 prompts gains less than chance gives the best of 36, and on multilingual toxicity the
number hides a significant gain of the group that was worst before the search. The gains that survive belong to the
task model: none of the four holds on two other models. This repository holds everything behind those results, the
seven checks of Section 8 as a small library, and the optimizers and queries needed to run the experiments again.

All predictions of every prompt on every test example are saved, so every table and figure is recomputed from them with
no API key and no query.

## Install

```
pip install -e .            # numpy, scipy, matplotlib
pip install -e ".[query]"   # to rebuild the texts, query models or run optimizers: openai, datasets
pip install -e ".[dev]"     # pytest
```

## Check a worst-group gain of your own

```python
from wgg.checks import run_checks, format_report

# preds: {prompt name: int array, 0/1, -1 for an unparsable answer}, aligned with the labels y and group ids g
report = run_checks(preds, y, g, seed="seed", finals=optimizer_finals, second=preds_on_another_model)
print(format_report(report))
```

or on files: `python -m wgg.checks --preds-dir DIR --set set.json` (a directory of `<prompt>.npy` and a JSON with
`labels` and `cluster_ids`). On the released benchmarks: `python -m wgg.checks --benchmark toxlang --second-model gpt-4o-mini`.
The seven checks are listed in `wgg/checks.py` and in Table 7 of the paper.

## Reproduce the paper

```
python reproduce.py --quick     # a few minutes; everything but the slowest analyses
python reproduce.py             # all of it, about two hours (the split-half replication takes most)
python reproduce.py --list
pytest                          # unit tests, and the numbers of the paper's tables against results/
```

| in the paper | command (`python -m ...`) | output in `results/` |
|---|---|---|
| Table 1, Table 2 (contrasts in W, WC; counts) | `analysis.compare_with_seed` | `<benchmark>/power_summary.json` |
| Table 2, Figure 1, Figure 2, App. E (chance baseline, Reality Check) | `analysis.chance_baseline [--pool finals]` | `<benchmark>/best_of_n_ceiling_hard_min_{all,finals}.json` |
| Table 2 target-group block, Section 5, Figure 3, Figure 6, App. F | `analysis.target_group [--high-resolution]` | `<benchmark>/target_group_contrast*.json`, `target_group_operating_points.json` |
| Table 3 (replication of declared differences), App. E | `analysis.replication` | `<benchmark>/statistic_reproducibility_n100.json`, `statistic_faithfulness.json` |
| Table 3 (stability of the minimum) | `analysis.argmin` | `<benchmark>/argmin.json` |
| Figure 4 (number of groups), App. G | `analysis.regroup random` | `<benchmark>/regroup_pseudo_hard_min_*.json` |
| App. G (groupings with a meaning), App. M (gap), App. B (difficulty groups, cell size) | `analysis.regroup meaningful / features / cell-size` | `<benchmark>/semantic_regroup_*.json`, `regroup_features.json`, `cell_size_h6.json` |
| App. G (72 protocols, length penalty, pool composition), Sections 3 and 8 | `analysis.protocols run / decompose / composition` | `<benchmark>/lottery_summary_*.json`, `lottery_decompose_*.json`, `pool_composition.json` |
| Figure 5, App. G (test draw) | `analysis.test_draw halves / split-half` | `<benchmark>/draw_lottery_halves.json`, `civil/split_half_signs.json` |
| App. E (WILDS metric) | `analysis.wilds_metric` | `civil/wilds_wga16.json` |
| Section 7, Figure 7, App. J (transfer to other models) | `analysis.transfer gate / mnli / toxicity --model M` | `transfer_gate_M.json`, `<benchmark>/transfer_M.json` |
| App. J (two models under the seed prompt, reliability; GPT-4o-mini on CivilComments) | `analysis.second_model table / reliability`, and `chance_baseline`, `argmin`, `replication` with `--model gpt-4o-mini` | `second_model_seed_table.json`, `civil/target_rho_reliability.json`, `civil/gpt-4o-mini/` |
| App. H (live loops, ranking of methods) | `analysis.live_loops` | `<benchmark>/live_loops.json`, `toxlang/live_loops_ranking.json` |
| App. I (selection inside the optimizer) | `analysis.optimizer_selection [--invalid-as-error]` | `toxlang/archive_regret*.json`, `broken_candidates.json` |
| App. C (dropped examples and verdicts) | `analysis.dropped_rows` | `<benchmark>/dropped_rows.json` |
| App. M (can measurability be predicted?) | `analysis.predictability` | `<benchmark>/predictability.json` |
| All figures | `analysis.figures` | `figures/` |

`WGG_RESULTS=DIR` makes the analyses write to `DIR` instead of `results/`.

## Data

```
data/<benchmark>/            civil (CivilComments), mnli (MultiNLI), toxlang (multilingual toxicity)
  sets/{test,dev}.json       indices into the public dataset, labels, group ids, fingerprint; no text
  prompts.json               every prompt compared with the seed: the seed, the one-line edits (r15:*) and the finals
  predictions/<model>/<set>/<prompt>.npy   int16 labels of a prompt on every example, -1 = unparsable
  live_loop/, matrix_runs/   artifacts of the optimizer runs: best prompt, archive of every evaluated prompt, budgets
  requery.json               the retry of the unparsable answers (App. C)
  optimizer_data.json        (civil) the training and validation data of the optimizers
  archive/                   (toxlang) archives of the live loops scored on the test set (App. I)
preregistration/             the six plans and one amendment, with SHA256SUMS
```

Prompt names: `seed`; `r15:thr_p20`, ... the eleven one-line edits; `s9:42_ape`, `s13:42_gepa__hard_min`,
`s15:43_gepa__hard_min`, ... optimizer finals (`<prefix>:<seed>_<method>[__<selection statistic>]`).
Models: `gemma-3-12b` (main), `gpt-4o-mini`, `qwen3-235b`, `qwen3-30b`.

The texts of the public datasets are not redistributed. `python -m query.fetch_texts` rebuilds them from the public
sources and checks them against the released labels, groups and fingerprints.

## Run the models and the optimizers again

This costs money (the whole project cost $89.82 on OpenRouter). Put the key in `OPENROUTER_API_KEY` or a `.env` file.

```
python -m query.score --benchmark toxlang --model gpt-4o-mini --dry-run      # count the queries first
python -m query.score --benchmark toxlang --model gpt-4o-mini --prompts seed,s15:42_gepa__hard_min --max-spend 2
python -m optimizers.run matrix --seed 42 --methods ape,gpo --mock           # whole run on stand-in models, no cost
python -m optimizers.run live --benchmark mnli --methods ape,gepa --protocols hard_min --seeds 42
```

Scoring and live loops can be stopped at any moment (Ctrl-C or a `STOP` file) and continue with the same command.
`query/requery.py` repeats the unparsable answers. The optimizers are APE, ProTeGi, GPO, EvoPrompt (genetic and
differential evolution) and GEPA, with one optimizer LLM (DeepSeek-V4-Pro) and one task model per run; budgets per seed
are in `optimizers/run.py`. PRIME, the method of Appendix K, is not included; its two finals are in the pool.

"""The three benchmarks: where their data live and how their groups and validation draws are set up.

Each benchmark has a fixed test set and a validation universe (`data/<key>/sets/{test,dev}.json`),
predictions of every prompt by every task model (`data/<key>/predictions/<model>/<set>/<prompt>.npy`,
one int16 vector per prompt, -1 for an unparsable answer) and the prompt texts
(`data/<key>/prompts.json`). Analyses write to `results/<key>/`.

Prompt names: `seed` is the seed prompt, `r15:<edit>` the one-line edits, and optimizer finals are
`<prefix>:<seed>_<method>` with the prefix `s9` (CivilComments), `s13` (MultiNLI) or `s15`
(multilingual toxicity); on MultiNLI and toxicity a suffix `__<statistic>` names the selection
statistic of the run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODEL = "gemma-3-12b"
KEYS = ("civil", "mnli", "toxlang")
TITLES = {"civil": "CivilComments", "mnli": "MultiNLI", "toxlang": "Multilingual toxicity"}

CIVIL_GROUP_NAMES = ("none", "male", "female", "LGBTQ", "christian", "muslim", "other_religions",
                     "black", "white")


@dataclass(frozen=True)
class Benchmark:
    key: str
    ids: tuple                 # groups the worst-group metrics are computed over
    sample_groups: tuple       # groups validation draws are sampled from (CivilComments adds group 0,
                               # comments that mention no identity, which no group metric uses)
    uniform_per_cell: int      # validation rows per group x label cell in the uniform allocation
    pilot_per_cell: int        # rows per cell in the pilot that finds the worst groups
    dev_budget: int            # total validation rows an allocation may use
    final_prefix: str          # optimizer finals are named `<prefix>:<seed>_<method>...`
    method_sep: Optional[str]  # cut the selection statistic off the method name at this separator
    names: Optional[tuple]     # fixed group names indexed by id; None = read them from the set JSON
    test_set: str = "test"
    dev_set: str = "dev"

    @property
    def title(self) -> str:
        return TITLES[self.key]

    @property
    def data(self) -> Path:
        return ROOT / "data" / self.key

    @property
    def results(self) -> Path:
        """Where analyses write (`results/<key>/`; the environment variable WGG_RESULTS replaces `results`)."""
        return results_root() / self.key

    @property
    def pools(self) -> list:
        """Values of --pool: every prompt, the optimizer finals (+ seed), the one-line edits (+ seed)."""
        return ["all", self.final_prefix, "r15"]

    def out_dir(self, model: str = MAIN_MODEL) -> Path:
        """Where an analysis writes: results/<key>/ for the main model, results/<key>/<model>/ otherwise."""
        d = self.results if model == MAIN_MODEL else self.results / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def set_path(self, name: str) -> Path:
        return self.data / "sets" / f"{name}.json"

    def preds_dir(self, set_name: str, model: str = MAIN_MODEL) -> Path:
        return self.data / "predictions" / model / set_name

    def prompts(self) -> dict:
        return json.loads((self.data / "prompts.json").read_text(encoding="utf-8"))

    def prompt_words(self) -> dict:
        """{prompt name: word count}, the input of the length penalty."""
        return {n: len(t.split()) for n, t in self.prompts().items()}

    def group_names(self, rec: Optional[dict] = None) -> dict:
        """{group id: name}; from the set JSON on benchmarks without a fixed list."""
        if self.names is not None:
            return {i: n for i, n in enumerate(self.names)}
        rec = rec or json.loads(self.set_path(self.test_set).read_text(encoding="utf-8"))
        return {int(k): v for k, v in rec["group_names"].items()}

    def method_of(self, name: str) -> Optional[str]:
        """Optimizer of a final: `s9:42_evoprompt_ga` -> `evoprompt_ga`, `s13:42_ape__soft_min` -> `ape`.
        Anything else (seed, one-line edits) -> None."""
        if not (name.startswith(self.final_prefix + ":") and "_" in name):
            return None
        tail = name.split(":", 1)[1].split("_", 1)[1]
        return tail if self.method_sep is None else tail.split(self.method_sep, 1)[0]

    def is_final(self, name: str) -> bool:
        return name.startswith(self.final_prefix + ":")


def results_root() -> Path:
    """`results/`, or the directory named by the environment variable WGG_RESULTS."""
    return Path(os.environ.get("WGG_RESULTS") or ROOT / "results")


_SPECS = {
    "civil": dict(ids=tuple(range(1, 9)), sample_groups=tuple(range(9)), uniform_per_cell=50,
                  pilot_per_cell=20, dev_budget=900, final_prefix="s9", method_sep=None,
                  names=CIVIL_GROUP_NAMES),
    "mnli": dict(ids=tuple(range(1, 11)), sample_groups=tuple(range(1, 11)), uniform_per_cell=45,
                 pilot_per_cell=20, dev_budget=900, final_prefix="s13", method_sep="__", names=None),
    "toxlang": dict(ids=tuple(range(1, 7)), sample_groups=tuple(range(1, 7)), uniform_per_cell=45,
                    pilot_per_cell=20, dev_budget=900, final_prefix="s15", method_sep="__", names=None),
}


def load(key: str) -> Benchmark:
    if key not in _SPECS:
        raise SystemExit(f"unknown benchmark {key!r}; choose one of {', '.join(KEYS)}")
    return Benchmark(key=key, **_SPECS[key])


def add_argument(parser, default: str = "civil", choices=KEYS) -> None:
    parser.add_argument("--benchmark", default=default, choices=list(choices))

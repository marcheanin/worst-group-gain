"""The seven figures of the paper, drawn from the saved predictions and results.

Figures 4, 5 and 7 read the outputs of `regroup.py`, `test_draw.py` and `transfer.py`/`target_group.py`
(run those first, or use `reproduce.py`); the others are computed here from the predictions.

Colours belong to the benchmark, not to a rank: CivilComments blue, MultiNLI teal, multilingual toxicity
orange. Every mark carries a text label, since the two light slots have a contrast below 3:1 with the
background.

Two modes. The default is for the paper: no title and no explanatory text (they belong in the caption),
sized to the ACL template (6.3 in for a figure* across two columns, 3.03 in for one column), PNG at 300
dpi and PDF in `figures/`. `--standalone` adds a title and notes, is larger, and writes PNG to
`figures/standalone/` for slides.

    python -m analysis.figures [--standalone] [--only N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from wgg import benchmarks  # noqa: E402
from wgg.benchmarks import ROOT  # noqa: E402
from wgg.stats import Bootstrap, gba_by_group, load_preds, load_set, valid_rows  # noqa: E402

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
STAND_COLOR = {"civil": "#2a78d6", "mnli": "#1baf7a", "toxlang": "#eb6834"}
STAND_LABEL = benchmarks.TITLES
ACCENT = "#eda100"
W_FULL, W_ONE = 6.3, 3.03

STANDALONE = False
SURFACE = "#ffffff"


def set_mode(standalone: bool):
    global STANDALONE, SURFACE
    STANDALONE = standalone
    SURFACE = "#fcfcfb" if standalone else "#ffffff"
    base = 9 if standalone else 7.5
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": base, "font.family": "DejaVu Sans",
        "axes.edgecolor": INK3, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": base,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "axes.titlesize": base + 1 if standalone else base, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": base - 1,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
    })


def size(standalone: tuple, paper: tuple) -> tuple:
    return standalone if STANDALONE else paper


def fs(standalone: float, paper: float) -> float:
    return standalone if STANDALONE else paper


def finish(fig, out, name, title=None, note=None, title_y=1.04, note_y=-0.09):
    """A title and a note only in the standalone mode; in the paper they are in the caption."""
    if STANDALONE and title:
        fig.suptitle(title, fontsize=11, fontweight="bold", color=INK, y=title_y)
    fig.tight_layout()
    if STANDALONE and note:
        fig.text(0.5, note_y, note, ha="center", fontsize=8, color=INK2)
    if STANDALONE:
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    else:
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def tidy(ax, grid="y"):
    """Recessive axes: two lines instead of a frame, grid along one axis only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, zorder=0)
    ax.set_axisbelow(True)


def load_stand(key):
    b = benchmarks.load(key)
    y, c, rec = load_set(b, b.test_set)
    P = load_preds(b, b.test_set, len(y))
    keep = valid_rows(P, len(y), sorted(n for n in P if not n.startswith("CONTROL:")), verbose=False)
    return y, c, keep, P, b.group_names(rec), b, b.ids


def noise_max(key):
    """Distribution of the best of N gains under the null, and the observed best gain."""
    y, c, keep, P, gn, b, ids = load_stand(key)
    names = [n for n in sorted(P) if not n.startswith("CONTROL:") and n != "seed"]
    bs = Bootstrap(y[keep], c[keep], 4000, np.random.default_rng(0), groups=ids)
    o_s, b_s = bs.metrics(P["seed"][keep])
    obs, boot = [], []
    for n in names:
        o, bt = bs.metrics(P[n][keep])
        obs.append(o["hard_min"] - o_s["hard_min"])
        boot.append(bt["hard_min"] - b_s["hard_min"])
    obs = np.array(obs)
    boot = np.stack(boot, 0)
    return (boot - obs[:, None]).max(axis=0), float(obs.max()), len(names)


def fig1(out):
    fig, axes = plt.subplots(1, 3, figsize=size((10.5, 3.2), (W_FULL, 2.3)), sharey=True)
    stands = ("civil", "mnli", "toxlang")
    data = []
    for ax, stand in zip(axes, stands):
        mx, best, N = noise_max(stand)
        ax.hist(mx, bins=45, color=STAND_COLOR[stand], alpha=0.30, edgecolor="none", zorder=2)
        data.append((mx, best, N))
    top = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, top)
    under = dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=0.5)
    for ax, stand, (mx, best, N) in zip(axes, stands, data):
        col = STAND_COLOR[stand]
        c95 = float(np.percentile(mx, 95))
        chance = float(mx.mean())
        p = float((mx >= best).mean())
        ax.axvline(c95, color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.axvline(best, color=col, lw=2.2 if STANDALONE else 1.6, zorder=4)
        ax.axvline(chance, ymax=0.60, color=INK2, lw=1.1, ls=(0, (1, 2)), zorder=3)
        name = "chance baseline" if STANDALONE else "chance"
        ch_right = best < chance
        chance_txt = f"  {name}\n  {chance:+.3f}" if ch_right else f"{name}  \n{chance:+.3f}  "
        ax.text(chance, top * 0.62, chance_txt, color=INK2, fontsize=fs(7.5, 6), va="bottom",
                ha="left" if ch_right else "right", bbox=under, zorder=2.5)
        left_of_c95 = best < c95
        x_lo, x_hi = ax.get_xlim()
        obs_right = (not left_of_c95) or (c95 - best) / (x_hi - x_lo) > 0.2
        obs_txt = f"  observed\n  {best:+.3f}" if obs_right else f"observed  \n{best:+.3f}  "
        ax.text(best, top * 0.97, obs_txt, color=col, fontsize=fs(8, 6.5), va="top",
                ha="left" if obs_right else "right", fontweight="bold", bbox=under, zorder=2.5)
        c95_txt = f"  C95\n  {c95:+.3f}" if left_of_c95 else f"C95  \n{c95:+.3f}  "
        if STANDALONE:
            c95_txt = c95_txt.replace("C95", "noise ceiling C95")
        ax.text(c95, top * 0.50, c95_txt, color=INK2, fontsize=fs(7.5, 6), va="top",
                ha="left" if left_of_c95 else "right", bbox=under, zorder=2.5)
        verdict = "within noise" if left_of_c95 else "above noise"
        p_txt = f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
        if STANDALONE:
            ax.set_title(f"{STAND_LABEL[stand]}\nN = {N},  p = {p_txt} — {verdict}", fontsize=9, pad=8)
            ax.set_xlabel("gain in worst-group balanced accuracy")
        else:
            ax.set_title(f"{STAND_LABEL[stand]} (N = {N})\np = {p_txt}, {verdict}", pad=5)
        span = np.subtract(*ax.get_xlim()[::-1])
        ax.xaxis.set_major_locator(MultipleLocator(0.02 if span < 0.1 else 0.04))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.02f}"))
        tidy(ax)
        print(f"  {stand}: best {best:+.4f}, chance {chance:+.4f}, C95 {c95:+.4f}, p {p:.4f}")
    axes[0].set_ylabel("bootstrap resamples")
    if not STANDALONE:
        axes[1].set_xlabel("gain in worst-group balanced accuracy")
    finish(fig, out, "fig1_winners_curse", "Observed best gain against what the best of N gains by chance",
           "Histogram: the gain the best of N prompts reaches by chance alone, when no prompt beats the "
           "seed.\np: share of resamples in which chance alone gives at least the observed gain.")


def fig2(out):
    fig, ax = plt.subplots(figsize=size((6.2, 3.6), (W_ONE, 2.6)))
    for stand in ("civil", "mnli", "toxlang"):
        _, _, N = noise_max(stand)
        y, c, keep, P, gn, b, ids = load_stand(stand)
        names = [n for n in sorted(P) if not n.startswith("CONTROL:") and n != "seed"]
        bs = Bootstrap(y[keep], c[keep], 2000, np.random.default_rng(0), groups=ids)
        o_s, b_s = bs.metrics(P["seed"][keep])
        obs, boot = [], []
        for n in names:
            o, bt = bs.metrics(P[n][keep])
            obs.append(o["hard_min"] - o_s["hard_min"])
            boot.append(bt["hard_min"] - b_s["hard_min"])
        noise = np.stack(boot, 0) - np.array(obs)[:, None]
        rng = np.random.default_rng(7)
        ks = [k for k in (2, 3, 5, 8, 12, 16, 20, 25, 30, N) if k <= N]
        ys = [np.mean([noise[rng.choice(N, size=k, replace=False)].max(axis=0).mean() for _ in range(80)]) for k in ks]
        col = STAND_COLOR[stand]
        ax.plot(ks, ys, color=col, lw=2 if STANDALONE else 1.6, marker="o", ms=fs(4, 3), zorder=3)
        ax.annotate(STAND_LABEL[stand], (ks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=col,
                    fontsize=fs(8, 6.5), va="center", fontweight="bold")
    ax.set_xlabel("pool size N (number of prompts compared)")
    ax.set_ylabel("expected best-of-N gain\nunder the null" if not STANDALONE else "expected best-of-N gain under the null")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.03f}"))
    ax.set_xlim(0, 46 if STANDALONE else 60)
    if not STANDALONE:
        ax.set_xticks([0, 10, 20, 30, 40])
    else:
        ax.set_title("The more prompts are compared, the larger the gain from noise alone", fontsize=10, pad=10)
    tidy(ax)
    finish(fig, out, "fig2_ceiling_vs_N")


def fig3(out):
    """What a threshold shift alone achieves on the target group, and where the finals lie."""
    fig, axes = plt.subplots(1, 3, figsize=size((11, 3.6), (W_FULL, 2.5)))
    for ax, stand in zip(axes, ("civil", "mnli", "toxlang")):
        y, c, keep, P, gn, b, ids = load_stand(stand)
        gs = gba_by_group(P["seed"], y, c, keep, ids)
        w = min(gs, key=gs.get)
        pos, neg = (c == w) & (y == 1) & keep, (c == w) & (y == 0) & keep

        def point(n):
            return float((P[n][neg] == 1).mean()), float(0.5 * ((P[n][pos] == 1).mean() + (P[n][neg] == 0).mean()))

        edits = sorted(n for n in P if n.startswith("r15:"))
        finals = sorted(n for n in P if b.is_final(n))
        ep = sorted([point(n) for n in edits] + [point("seed")])
        ex, ey = zip(*ep)
        lo, hi = min(ex), max(ex)
        col = STAND_COLOR[stand]
        allx = list(ex) + [point(n)[0] for n in finals]
        pad = 0.06 * (max(allx) - min(allx) + 1e-9)
        x0, x1 = min(allx) - pad, max(allx) + pad
        ax.axvspan(lo, hi, color=col, alpha=0.06, zorder=1)
        for xb in (lo, hi):
            ax.axvline(xb, color=INK3, lw=0.8, ls=(0, (2, 3)), zorder=2)
        ax.plot(ex, ey, color=INK3, lw=fs(0.8, 0.55), alpha=0.7, marker="o", ms=fs(3.5, 2.5),
                markerfacecolor=INK2, markeredgecolor=INK2, zorder=3,
                label="one-line strictness edits (threshold shift only)")
        fx, fy = zip(*[point(n) for n in finals])
        ax.scatter(fx, fy, s=fs(26, 14), color=col, edgecolor=SURFACE, linewidth=0.6, zorder=4, label="optimizer finals")
        sx, sy = point("seed")
        ax.scatter([sx], [sy], s=fs(70, 34), marker="D", color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=5,
                   label="seed prompt")
        best_edit = max(ey)
        ax.axhline(best_edit, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=2)
        above = sum(1 for v in fy if v > best_edit)
        ax.set_xlim(x0, x1)
        ax.text(lo + (hi - lo) / 2, ax.get_ylim()[0], "range of the edits", fontsize=fs(6.5, 5.5), color=INK3,
                ha="center", va="bottom")
        if stand == "toxlang":
            bi = int(np.argmax(fy))
            ann = (f"GEPA: +{fy[bi] - best_edit:.3f} above the threshold" if STANDALONE
                   else f"GEPA: +{fy[bi] - best_edit:.3f}\nabove threshold")
            ax.annotate(ann, (fx[bi], fy[bi]), xytext=(-8, -14 if STANDALONE else -4), textcoords="offset points",
                        fontsize=fs(7.5, 6.5), color=col, fontweight="bold", ha="right", va="top")
        if STANDALONE:
            ax.set_title(f"{STAND_LABEL[stand]}\ntarget group '{gn[w]}': {above} of {len(finals)} above the line",
                         fontsize=9, pad=8)
            ax.set_xlabel("strictness: false-positive rate")
        else:
            ax.set_title(f"{STAND_LABEL[stand]}, group '{gn[w]}'\n{above} of {len(finals)} finals above the line", pad=5)
        tidy(ax)
    axes[0].set_ylabel("target-group balanced accuracy")
    if not STANDALONE:
        axes[1].set_xlabel("strictness: false-positive rate on the target group")
    h, l = axes[0].get_legend_handles_labels()
    if STANDALONE:
        fig.legend(h, l, loc="center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    else:
        fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=6.5, handletextpad=0.4,
                   columnspacing=1.2)
    finish(fig, out, "fig3_strictness_curves", "The strictness curve and where the prompts lie on it",
           "Grey points: one-line edits that by construction change only strictness; the thin line only joins them "
           "in order of strictness. Dashed line: the best they achieve on this group.\nA point above the dashed line "
           "means the prompt improved discrimination rather than merely shifting the threshold. Shaded: strictness "
           "range covered by the edits; outside it is extrapolation.",
           title_y=1.09, note_y=-0.13)


def fig4(out):
    """Number of groups as a lever: median over the random partitions of `regroup.py random`."""
    spec = {"civil": ([2, 4, 8], 50, "o"), "toxlang": ([2, 3, 6], 50, "s"), "mnli": ([2, 4, 10], 50, "o")}
    reps = {("civil", 8): 200, ("toxlang", 6): 200, ("mnli", 10): 200}
    label_off = {"mnli": ((7, 0), "center"), "civil": ((6, 5), "baseline"), "toxlang": ((6, 13), "baseline")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=size((9, 3.4), (W_FULL, 2.3)))
    for stand, (ks, _, mk) in spec.items():
        b = benchmarks.load(stand)
        hits, widths, tot = [], [], None
        for k in ks:
            d = json.loads((b.results / f"regroup_pseudo_hard_min_{k}g_{reps.get((stand, k), 50)}rep.json").read_text(encoding="utf-8"))
            tot = d["contrasts"]
            hits.append(float(np.median([r["holm"] for r in d["per_rep"]])))
            widths.append(round(float(np.median([r["mean_width"] for r in d["per_rep"]])), 4))
        col = STAND_COLOR[stand]
        lw, ms = fs(2, 1.6), fs(5, 3.5) * (0.95 if mk == "s" else 1.0)
        a1.plot(ks, [100 * h / tot for h in hits], color=col, lw=lw, marker=mk, ms=ms, zorder=3)
        off, va = label_off[stand]
        a1.annotate(STAND_LABEL[stand], (ks[0], 100 * hits[0] / tot), xytext=off, va=va, textcoords="offset points",
                    color=col, fontsize=fs(8, 6.5), fontweight="bold")
        a2.plot(ks, widths, color=col, lw=lw, marker=mk, ms=ms, zorder=3)
    a1.set_xlabel("number of groups k")
    a1.set_ylabel("significant findings,\n% of contrasts")
    a1.set_title("Findings after multiplicity correction", fontsize=fs(9.5, 7.5), pad=fs(8, 5))
    a2.set_xlabel("number of groups k")
    a2.set_ylabel("confidence interval width")
    a2.set_title("Interval width grows roughly as √k", fontsize=fs(9.5, 7.5), pad=fs(8, 5))
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.03f}"))
    for ax in (a1, a2):
        ax.set_xticks([2, 4, 6, 8, 10])
        tidy(ax)
    finish(fig, out, "fig4_group_count", "The effect stays the same; only the number of groups changes",
           "Random groups of equal size: the data, predictions and effect are unchanged. "
           "On MultiNLI, k = 10 gives no findings and k = 2 gives twelve.", title_y=1.03, note_y=-0.10)


def fig5(out):
    fig, ax = plt.subplots(figsize=size((6.4, 3.8), (W_ONE, 2.8)))
    for stand in ("civil", "mnli"):
        d = json.loads((benchmarks.load(stand).results / "draw_lottery_halves.json").read_text(encoding="utf-8"))
        w = np.array([round(h["width_mean"], 4) for h in d["halves"]])  # as printed in the tables
        f = np.array([round(100 * h["share"], 1) for h in d["halves"]])
        col = STAND_COLOR[stand]
        ax.scatter(w, f, s=fs(40, 18), color=col, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        label = f"{STAND_LABEL[stand]}\nwidth {w.min():.4f}–{w.max():.4f}\nfindings {f.min():.1f}–{f.max():.1f}%"
        if stand == "mnli":
            ax.annotate(label, (w.max(), f.mean()), xytext=(fs(14, 10), 0), textcoords="offset points", color=col,
                        fontsize=fs(8, 6.5), ha="left", va="center", fontweight="bold")
        else:
            ax.annotate(label, (w.max(), f.max()), xytext=(0, fs(14, 8)), textcoords="offset points", color=col,
                        fontsize=fs(8, 6.5), ha="right", va="bottom", fontweight="bold")
    ax.set_xlabel("mean confidence interval width")
    ax.set_ylabel("distinguishable prompt pairs, %")
    if STANDALONE:
        ax.set_title("Interval width barely changes; the number of findings changes several-fold", fontsize=10, pad=10)
    tidy(ax, grid="both")
    finish(fig, out, "fig5_draw_lottery", note=(
        "Each point is an independent half of the test of the same size, 12 halves per benchmark. "
        "The intervals are honest;\nwhat varies is which differences end up in the table."))


def fig6(out):
    y, c, keep, P, gn, b, ids = load_stand("toxlang")
    gs = gba_by_group(P["seed"], y, c, keep, ids)
    gb = gba_by_group(P["s15:43_gepa__hard_min"], y, c, keep, ids)
    order = sorted(gs, key=lambda g: gs[g])
    labels = [gn[g] for g in order]
    a = [gs[g] for g in order]
    bb = [gb[g] for g in order]
    fig, ax = plt.subplots(figsize=size((6.8, 3.8), (W_ONE, 2.9)))
    for i, (u, v) in enumerate(zip(a, bb)):
        colr = STAND_COLOR["toxlang"] if v >= u else INK3
        ax.plot([0, 1], [u, v], color=colr, lw=fs(1.8, 1.4), zorder=3, alpha=0.9)
        ax.scatter([0, 1], [u, v], s=fs(28, 14), color=colr, edgecolor=SURFACE, linewidth=0.6, zorder=4)
        ax.annotate(labels[i], (0, u), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=fs(8, 6.5), color=INK2)
        ax.annotate(f"{v:.3f}", (1, v), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=fs(8, 6.5), color=INK2)
    mn_a, mn_b = min(a), min(bb)
    ring = fs(150, 70)
    ax.scatter([0], [mn_a], s=ring, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
    ax.scatter([1], [mn_b], s=ring, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
    ax.annotate("minimum", (0, mn_a), xytext=(fs(16, 10), 4), textcoords="offset points", ha="left", va="bottom",
                fontsize=fs(8, 6.5), color=INK, fontweight="bold")
    if STANDALONE:
        ax.annotate("the minimum moved here", (1, mn_b), xytext=(-40, -16), textcoords="offset points", ha="right",
                    va="top", fontsize=8, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.9, shrinkA=2, shrinkB=9))
    else:
        ax.annotate("new minimum", (1, mn_b), xytext=(0, -9), textcoords="offset points", ha="center", va="top",
                    fontsize=6.5, color=INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["seed prompt", "best final (GEPA)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("balanced accuracy per language")
    if STANDALONE:
        ax.set_title("The target group gains +0.061, yet worst-group accuracy reports noise", fontsize=10, pad=10)
    tidy(ax)
    finish(fig, out, "fig6_minimum_moved", note=(
        "Orange: languages that improved. Amharic (am), the worst under the seed prompt, rose from "
        "0.721 to 0.782 (p < 0.001).\nGerman (de) dropped within noise and became the new minimum, so "
        "worst-group accuracy shows only +0.024."), note_y=-0.07)


def fig7(out):
    """The four target-group gains significant on the main model, re-measured on GPT-4o-mini and Qwen3-235B.
    Filled marker: significant after Holm; hollow: not."""
    tj = json.loads((benchmarks.load("toxlang").results / "target_group_contrast.json").read_text(encoding="utf-8"))
    tg = json.loads((benchmarks.load("toxlang").results / "transfer_gpt-4o-mini.json").read_text(encoding="utf-8"))
    tq = json.loads((benchmarks.load("toxlang").results / "transfer_qwen3-235b.json").read_text(encoding="utf-8"))
    mg = json.loads((benchmarks.load("mnli").results / "transfer_gpt-4o-mini.json").read_text(encoding="utf-8"))
    mq = json.loads((benchmarks.load("mnli").results / "transfer_qwen3-235b.json").read_text(encoding="utf-8"))
    g43, g42 = "s15:43_gepa__hard_min", "s15:42_gepa__hard_min"
    a43, a42 = "s13:43_ape__global", "s13:42_gepa__hard_min"

    def tox(f):
        return tj["finals"][f], tg["other"][f], tq["other"][f]

    def mnli(f):
        return mg["gemma_same_rows"][f]["target"], mg["other"][f]["target"], mq["other"][f]["target"]

    rows = [("Toxicity · GEPA (seed 43)", "group: am", *tox(g43)),
            ("Toxicity · GEPA (seed 42)", "group: am", *tox(g42)),
            ("MultiNLI · APE (seed 43)", "group: verbatim", *mnli(a43)),
            ("MultiNLI · GEPA (seed 42)", "group: verbatim", *mnli(a42))]
    QWEN = "#2a78d6"
    cols = (INK, ACCENT, QWEN)
    ms = fs(6.5, 4.4)
    offs = (-0.25, 0.0, 0.25)
    fig, ax = plt.subplots(figsize=size((6.4, 4.2), (W_ONE, 2.95)))
    ax.axvline(0, color=INK3, lw=0.9, zorder=2)
    for i, (lab, grp, g, q, w) in enumerate(rows):
        pts = []
        for k, (r, col, mk) in enumerate(zip((g, q, w), cols, ("o", "s", "D"))):
            yy = i + offs[k]
            ax.plot(r["ci"], [yy, yy], color=col, lw=fs(1.8, 1.2), solid_capstyle="butt", zorder=3)
            ax.plot([r["delta"]], [yy], marker=mk, ms=ms * (0.9 if mk == "D" else 1.0), color=col,
                    markerfacecolor=col if r["holm"] else SURFACE, markeredgewidth=fs(1.5, 1.0), zorder=5, ls="none")
            pts.append((r["delta"], yy))
        for j in (1, 2):
            ax.annotate("", xy=pts[j], xytext=pts[0], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=fs(0.9, 0.6), shrinkA=fs(5, 3.5),
                                        shrinkB=fs(5, 3.5), mutation_scale=fs(8, 6.0)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}\n{r[1]}" for r in rows], fontsize=fs(8, 6.0))
    ax.set_ylim(len(rows) - 0.55, -0.55)
    ax.set_xlim(-0.08, 0.108)
    ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}" if abs(v) > 1e-9 else "0"))
    ax.set_xlabel("change in balanced accuracy of\nthe pre-search worst group vs. seed" if not STANDALONE
                  else "change in balanced accuracy of the pre-search worst group vs. seed")
    ax.tick_params(axis="y", length=0)
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="none", color=INK, ms=ms, markeredgewidth=1.1, label="gemma-3-12b-it"),
               Line2D([], [], marker="s", ls="none", color=ACCENT, ms=ms, markeredgewidth=1.1, label="gpt-4o-mini"),
               Line2D([], [], marker="D", ls="none", color=QWEN, ms=ms * 0.9, markeredgewidth=1.1, label="qwen3-235b"),
               Line2D([], [], marker="o", ls="none", color=INK3, ms=ms, markeredgewidth=1.1, markerfacecolor=INK3,
                      label="significant after Holm"),
               Line2D([], [], marker="o", ls="none", color=INK3, ms=ms, markeredgewidth=1.1, markerfacecolor=SURFACE,
                      label="not significant")]
    if STANDALONE:
        fig.legend(handles=handles, loc="center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.995), fontsize=6.0,
                   handletextpad=0.3, columnspacing=1.0, labelspacing=0.3)
    finish(fig, out, "fig7_model_transfer",
           "Gains significant after Holm on gemma-3-12b-it, re-measured on gpt-4o-mini and qwen3-235b",
           "Point and 95% bootstrap interval per model; arrows from gemma-3-12b-it to the other two models. Filled: "
           "significant after Holm correction within the benchmark; hollow: not.\nMultiNLI rows are read on the "
           "5000 rows valid for all models.", title_y=1.08, note_y=-0.10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--only", type=int, default=0)
    args = ap.parse_args()
    set_mode(args.standalone)
    from pathlib import Path
    out = Path(args.out) if args.out else ROOT / "figures" / ("standalone" if args.standalone else "")
    out.mkdir(parents=True, exist_ok=True)
    for i, fn in enumerate((fig1, fig2, fig3, fig4, fig5, fig6, fig7), 1):
        if args.only and args.only != i:
            continue
        print(f"figure {i}")
        fn(out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()

"""
Build the paper's figures from the aggregator's JSON export.

    python -m src.aggregate_results --root results/hpc_final --format json \
        --out paper_data.json
    python -m src.make_paper_figures                       # -> ../overleaf/figures/
    python -m src.make_paper_figures --data paper_data.json --out some/dir

Everything is read from the export (schema `fdist-aggregate-v1`), never from the
results tree, so the figures regenerate from one committed file and can never
disagree with the tables -- `src/aggregate_results.py` builds both from the same
per-seed curves.

Palette
-------
Six categorical series is at the edge of what stays separable under colour-vision
deficiency. The Okabe-Ito subset below is the only candidate that cleared every
check in the palette validator (lightness band, chroma floor, all-pairs CVD,
normal-vision floor); it carries two dischargeable warnings, and both are
discharged here rather than ignored:

* all-pairs CVD worst case is dE 7.6 (deutan), inside the 6-8 band that is legal
  only WITH a secondary encoding -> every scheme also carries its own line style
  and marker, so identity never rests on hue alone;
* three hues sit under 3:1 contrast on white, which obliges a table view -> the
  paper's main results table lists every one of these series as a row.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate_results import SCHEME_LABEL, SCHEME_ORDER

# (colour, linestyle, marker), in SCHEME_ORDER. Assigned in fixed order and never
# cycled, so a scheme keeps its colour across every figure and panel.
# Hue assignment is not arbitrary: the validator's closest normal-vision pair is
# #D55E00 / #E69F00, so those two go to magnitude and f_dist_one_shot -- curves a
# reader rarely needs to tell apart -- while magnitude and the exact scheme, the
# contrast the paper turns on, sit at opposite ends of the palette.
STYLE = {
    "magnitude":        ("#D55E00", (0, (5, 2)),       "o"),
    "fim":              ("#CC79A7", (0, (4, 1, 1, 1)), "s"),
    "f_dist_one_shot":  ("#E69F00", (0, (1, 1.6)),     "^"),
    "f_dist_iterative": ("#009E73", "-",               "D"),
    "f_dist_global":    ("#56B4E9", "-",               "v"),
    "f_dist":           ("#0072B2", "-",               "*"),
}
# The exact scheme is the ground truth the cheap ones are measured against, so it
# is drawn first under a wide translucent halo: where a cheap scheme reproduces
# it, that curve visibly sits inside the corridor.
HALO = "f_dist"

MODEL_LABEL = {"SimpleNN_h2_n64": "SimpleNN", "SimpleViT_d4_e128_h4_p4": "SimpleViT"}
DATA_LABEL = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
CELLS = [("SimpleNN_h2_n64", "mnist"), ("SimpleNN_h2_n64", "cifar10"),
         ("SimpleViT_d4_e128_h4_p4", "mnist"), ("SimpleViT_d4_e128_h4_p4", "cifar10")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#555555",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(path):
    with open(path) as fh:
        return json.load(fh)


def pick(groups, model, dataset, scheme, k="main"):
    """One group. k='main' takes the table entry (K=3 where K applies)."""
    out = [g for g in groups if g["model"] == model and g["dataset"] == dataset
           and g["scheme"] == scheme and (g["K"] in (None, 3) if k == "main" else g["K"] == k)]
    return out[0] if out else None


def panel_title(model, dataset):
    return f"{MODEL_LABEL[model]} / {DATA_LABEL[dataset]}"


def _tidy(ax):
    ax.grid(True, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_curves(groups, outdir, metric="acc_norm", stem="fig_curves"):
    """2x2 grid: normalised metric vs pruning ratio, mean +/- std over seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True, sharey=True)
    handles = {}
    for ax, (model, dataset) in zip(axes.flat, CELLS):
        _tidy(ax)
        for scheme in SCHEME_ORDER:
            g = pick(groups, model, dataset, scheme)
            if g is None:
                continue
            colour, ls, marker = STYLE[scheme]
            x = np.array(g["pruning_ratio"])
            mean = np.array(g["curves"][metric]["mean"])
            std = np.array(g["curves"][metric]["std"])
            if scheme == HALO:
                ax.plot(x, mean, color=colour, lw=4.5, alpha=0.25, solid_capstyle="round", zorder=2)
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.13, lw=0, zorder=1)
            (line,) = ax.plot(x, mean, color=colour, ls=ls, lw=1.5, marker=marker,
                              ms=3.6, mew=0, zorder=3)
            handles.setdefault(scheme, line)
        ax.set_title(panel_title(model, dataset), pad=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.06)
    for ax in axes[1]:
        ax.set_xlabel("pruning ratio")
    ylab = "normalised accuracy" if metric == "acc_norm" else "normalised MCC"
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab)
    fig.legend([handles[s] for s in SCHEME_ORDER if s in handles],
               [SCHEME_LABEL[s] for s in SCHEME_ORDER if s in handles],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.075))
    fig.tight_layout()
    save(fig, outdir, stem)


def fig_pareto(groups, outdir, stem="fig_pareto"):
    """2x2 grid: accuracy AUC vs compute cost. One axis per panel, log x."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    handles = {}
    for ax, (model, dataset) in zip(axes.flat, CELLS):
        _tidy(ax)
        for scheme in SCHEME_ORDER:
            g = pick(groups, model, dataset, scheme)
            if g is None:
                continue
            colour, _, marker = STYLE[scheme]
            # Median, not mean: one ViT/MNIST seed ran on a contended node and is
            # ~2x the other four, which would drag the mean by 18%.
            t = float(np.median(g["compute_seconds"]["per_seed"]))
            auc = g["auc"]["acc_norm"]
            pt = ax.errorbar(t, auc["mean"], yerr=auc["std"], color=colour, marker=marker,
                             ms=8, mew=0, capsize=2, elinewidth=1, lw=0, zorder=3)
            handles.setdefault(scheme, pt)
        ax.set_xscale("log")
        ax.set_title(panel_title(model, dataset), pad=4)

        # The paper's central claim is a horizontal distance on this axis: the
        # cheap iterative scheme and the exact one land at the same height. Draw
        # it, so the reader does not have to divide two tick labels in their head.
        cheap = pick(groups, model, dataset, "f_dist_iterative")
        exact = pick(groups, model, dataset, "f_dist")
        if cheap and exact:
            t_c = float(np.median(cheap["compute_seconds"]["per_seed"]))
            t_e = float(np.median(exact["compute_seconds"]["per_seed"]))
            y = 0.5 * (cheap["auc"]["acc_norm"]["mean"] + exact["auc"]["acc_norm"]["mean"])
            ax.annotate("", xy=(t_e, y), xytext=(t_c, y),
                        arrowprops=dict(arrowstyle="<->", color="#777777",
                                        lw=0.8, shrinkA=5, shrinkB=5), zorder=2)
            # Whole label inside one math group: a thin-space \, outside $...$
            # is not mathtext and renders as a literal backslash.
            ax.text(np.sqrt(t_c * t_e), y, f"$\\times {t_e / t_c:,.0f}$".replace(",", "\\,"),
                    ha="center", va="bottom", fontsize=7.5, color="#444444",
                    bbox=dict(fc="white", ec="none", pad=0.8), zorder=4)
    for ax in axes[1]:
        ax.set_xlabel("compute per sweep [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC, normalised accuracy")
    fig.legend([handles[s] for s in SCHEME_ORDER if s in handles],
               [SCHEME_LABEL[s] for s in SCHEME_ORDER if s in handles],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.075))
    fig.tight_layout()
    save(fig, outdir, stem)


def fig_kladder(groups, outdir, stem="fig_kladder"):
    """Two stacked panels sharing K: AUC above, compute below. Never a dual axis."""
    ks = sorted(g["K"] for g in groups
                if g["model"] == "SimpleNN_h2_n64" and g["dataset"] == "mnist"
                and g["scheme"] == "f_dist" and g["K"] is not None)
    rows = [pick(groups, "SimpleNN_h2_n64", "mnist", "f_dist", k=k) for k in ks]
    auc = np.array([g["auc"]["acc_norm"]["mean"] for g in rows])
    err = np.array([g["auc"]["acc_norm"]["std"] for g in rows])
    sec = np.array([np.median(g["compute_seconds"]["per_seed"]) for g in rows])
    colour = STYLE["f_dist"][0]

    # Equal spacing, not linear in K: the ladder is a sequence of refinements
    # whose cost doubles (probes/coordinate 1,2,4,8,16), so linear K would bunch
    # the cheap end and stretch the expensive one for no reason.
    xs = np.arange(len(ks))

    fig, (top, bot) = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for ax in (top, bot):
        _tidy(ax)
    # Without a reference band the axis zoom would dress seed noise up as a
    # trend. This is the K=2 result +/- its own spread: every richer grid lands
    # inside it, which is the actual finding.
    top.axhspan(auc[0] - err[0], auc[0] + err[0], color=colour, alpha=0.12, lw=0, zorder=1)
    top.axhline(auc[0], color=colour, lw=0.8, ls=(0, (4, 3)), alpha=0.6, zorder=2)
    top.errorbar(xs, auc, yerr=err, color=colour, marker="*", ms=9, mew=0,
                 capsize=2.5, elinewidth=1, lw=1.2, zorder=3)
    top.set_ylabel("AUC, norm. accuracy")
    top.text(0.97, 0.06, "$K\\!=\\!2$ result $\\pm\\,1\\sigma$", transform=top.transAxes,
             ha="right", va="bottom", fontsize=7, color="#444444")
    bot.plot(xs, sec, color=colour, marker="*", ms=9, mew=0, lw=1.2, zorder=3)
    bot.set_yscale("log")
    bot.set_ylabel("compute [s]")
    bot.set_xlabel("path resolution $K$")
    bot.set_xticks(xs)
    bot.set_xticklabels([str(k) for k in ks])
    fig.tight_layout()
    save(fig, outdir, stem)


def save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {os.path.join(outdir, stem)}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="paper_data.json")
    ap.add_argument("--out", default=os.path.join("..", "overleaf", "figures"))
    args = ap.parse_args()

    blob = load(args.data)
    groups = blob["groups"]
    print(f"{len(groups)} groups from {args.data!r}")
    fig_curves(groups, args.out, "acc_norm", "fig_curves")
    fig_curves(groups, args.out, "mcc_norm", "fig_curves_mcc")
    fig_pareto(groups, args.out)
    fig_kladder(groups, args.out)


if __name__ == "__main__":
    main()

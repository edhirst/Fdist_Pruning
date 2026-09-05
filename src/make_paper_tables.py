"""
Emit the paper's LaTeX tables from the aggregator's JSON export.

    python -m src.aggregate_results --root results/hpc_final --format json \
        --out paper_data.json
    python -m src.make_paper_tables > ../overleaf/tables.tex

`src/aggregate_results.py` renders one table per (cell, metric), which is eight
tables for this grid -- too many for a paper. This consolidates them into the
three the paper actually prints, reading the same per-seed export so a number can
never drift between a figure, a table and the prose.

Compute times are reported as the MEDIAN over seeds, not the mean. One
SimpleViT/MNIST exact-f_dist seed landed on a contended node and ran 6.41 d
against 3.36-3.38 d for the other four; the mean would then overstate the cost of
the algorithm by 18% on the strength of one queue artefact. Any group whose
spread exceeds 1.5x is flagged in a footnote rather than quietly averaged.
"""
import argparse
import json

import numpy as np

from .aggregate_results import SCHEME_LABEL, SCHEME_ORDER, fmt_time

CELLS = [("SimpleNN_h2_n64", "mnist"), ("SimpleNN_h2_n64", "cifar10"),
         ("SimpleViT_d4_e128_h4_p4", "mnist"), ("SimpleViT_d4_e128_h4_p4", "cifar10")]
MODEL_LABEL = {"SimpleNN_h2_n64": "SimpleNN", "SimpleViT_d4_e128_h4_p4": "SimpleViT"}
DATA_LABEL = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
# From the dense-model banner in outputs/hpc_final/ch_*.o* (mean +/- std, 5 seeds).
SETUP = {
    ("SimpleNN_h2_n64", "mnist"):        (55_050, 50, "97.59", "0.27"),
    ("SimpleNN_h2_n64", "cifar10"):      (201_482, 50, "50.46", "0.63"),
    ("SimpleViT_d4_e128_h4_p4", "mnist"):   (539_914, 50, "98.70", "0.10"),
    ("SimpleViT_d4_e128_h4_p4", "cifar10"): (545_930, 50, "81.22", "0.47"),
}
SPREAD_FLAG = 1.5      # max/min over seeds above which a time gets a footnote


def pick(groups, model, dataset, scheme, k="main"):
    hits = [g for g in groups if g["model"] == model and g["dataset"] == dataset
            and g["scheme"] == scheme
            and (g["K"] in (None, 3) if k == "main" else g["K"] == k)]
    return hits[0] if hits else None


def compute_cell(g):
    """Median wall-clock, plus whether this group's seeds disagree badly."""
    t = np.array(g["compute_seconds"]["per_seed"], dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return "--", False
    spread = float(t.max() / t.min()) if t.min() > 0 else 1.0
    return fmt_time(float(np.median(t)), np.nan, seeds=1), spread > SPREAD_FLAG


def table_setup():
    out = [r"\begin{tabular}{llrrr}", r"\toprule",
           r"Architecture & Dataset & Parameters & Epochs & Dense accuracy [\%] \\",
           r"\midrule"]
    for model, dataset in CELLS:
        n, epochs, acc, std = SETUP[(model, dataset)]
        out.append(f"{MODEL_LABEL[model]} & {DATA_LABEL[dataset]} & "
                   f"{n:,} & {epochs} & ${acc} \\pm {std}$ \\\\".replace(",", "\\,"))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def table_main(groups):
    out = [r"\begin{tabular}{lcccc c}", r"\toprule",
           r"Scheme & AUC (acc.) & floor-corr. & AUC (MCC) & "
           r"vs.\ mag.\ (acc.\,/\,MCC) & Compute \\"]
    flagged = False
    for model, dataset in CELLS:
        out += [r"\midrule",
                rf"\multicolumn{{6}}{{l}}{{\textit{{{MODEL_LABEL[model]} "
                rf"--- {DATA_LABEL[dataset]}}}}} \\", r"\midrule"]
        base = pick(groups, model, dataset, "magnitude")
        b_acc = base["auc"]["acc_norm"]["mean"]
        b_mcc = base["auc"]["mcc_norm"]["mean"]
        for scheme in SCHEME_ORDER:
            g = pick(groups, model, dataset, scheme)
            if g is None:
                continue
            a, m = g["auc"]["acc_norm"], g["auc"]["mcc_norm"]
            c = g["auc_floor_corrected"]["acc_norm"]
            t, flag = compute_cell(g)
            flagged |= flag
            # Hoisted: an f-string expression cannot contain a backslash on 3.10.
            dagger = "$^\\dagger$" if flag else ""
            out.append(
                f"{SCHEME_LABEL[scheme]} & "
                f"${a['mean']:.3f} \\pm {a['std']:.3f}$ & "
                f"${c['mean']:.3f} \\pm {c['std']:.3f}$ & "
                f"${m['mean']:.3f} \\pm {m['std']:.3f}$ & "
                f"${a['mean'] / b_acc:.2f}\\times$\\,/\\,${m['mean'] / b_mcc:.2f}\\times$ & "
                f"{t}{dagger} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if flagged:
        out.append("% dagger: seeds disagree by >1.5x -- queue contention, see caption")
    return "\n".join(out)


def table_kladder(groups):
    ks = sorted(g["K"] for g in groups
                if g["model"] == "SimpleNN_h2_n64" and g["dataset"] == "mnist"
                and g["scheme"] == "f_dist" and g["K"] is not None)
    rows = [pick(groups, "SimpleNN_h2_n64", "mnist", "f_dist", k=k) for k in ks]
    ref = rows[0]["auc"]["acc_norm"]["mean"]
    out = [r"\begin{tabular}{ccccc}", r"\toprule",
           r"$K$ & probes/coord. & AUC (acc.) & $\Delta$ vs.\ $K\!=\!2$ & Compute \\",
           r"\midrule"]
    for k, g in zip(ks, rows):
        a = g["auc"]["acc_norm"]
        t, _ = compute_cell(g)
        out.append(f"{k} & {k - 1} & ${a['mean']:.4f} \\pm {a['std']:.4f}$ & "
                   f"${a['mean'] - ref:+.4f}$ & {t} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="paper_data.json")
    args = ap.parse_args()
    groups = json.load(open(args.data))["groups"]

    print("% Auto-generated by src/make_paper_tables.py -- do not hand-edit.")
    print("% Regenerate: python -m src.make_paper_tables > ../overleaf/tables.tex\n")
    for name, body in (("tabSetup", table_setup()),
                       ("tabMain", table_main(groups)),
                       ("tabK", table_kladder(groups))):
        print(f"\\newcommand{{\\{name}}}{{%\n{body}%\n}}\n")


if __name__ == "__main__":
    main()

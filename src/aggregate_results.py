"""
Aggregate multi-seed pruning runs into the paper's results tables.

Reads every `*_results.json` under a results root, groups by
(dataset, architecture, scheme, K), and reports each metric as mean +/- std over
seeds, plus mean +/- std compute time.

    python -m src.aggregate_results                       # accuracy + MCC, markdown
    python -m src.aggregate_results --metrics acc_norm --format latex
    python -m src.aggregate_results --root results --out paper_tables.md

Notes
-----
* One table is rendered per metric. The default pair is accuracy and MCC:
  accuracy is the headline number, MCC is the chance-corrected check that a
  scheme is not just riding the majority class. Precision and F1 are still
  recorded by `run_pruning.py` and exported by `--format json`; they are simply
  not tabulated.
* AUC is re-integrated here from the stored curves rather than trusting each
  run's own `auc` field, so every entry uses one quadrature rule even if runs
  were produced with different sweep grids. A warning is printed if the grids
  within a group disagree, because AUCs from different grids are not comparable.
* "floor" is the normalised metric of a fully destroyed model (chance /
  baseline). It differs between architectures, so the floor-corrected column
  `(AUC - floor) / (1 - floor)` is the one to use for cross-architecture
  comparison. MCC needs no such correction -- a collapsed one-class predictor
  scores exactly 0 -- so that column is omitted where the floor is 0.
* Runs with no seed recorded are treated as a single unnamed seed and flagged.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

SCHEME_ORDER = ["magnitude", "fim", "f_dist_one_shot", "f_dist_iterative",
                "f_dist_global", "f_dist"]
SCHEME_LABEL = {
    "magnitude": "Magnitude",
    "fim": "FIM",
    "f_dist_one_shot": "F-dist (one-shot)",
    "f_dist_iterative": "F-dist (iterative)",
    "f_dist_global": "F-dist (global)",
    "f_dist": "F-dist (exact)",
}
# Every metric recorded by run_pruning.py, in the order it exports them.
METRICS = ["acc_norm", "precision_norm", "f1_norm", "mcc_norm"]
# The metrics the paper tabulates: one table each.
TABLE_METRICS = ["acc_norm", "mcc_norm"]
METRIC_LABEL = {
    "acc_norm": "normalised accuracy",
    "precision_norm": "normalised precision",
    "f1_norm": "normalised F1",
    "mcc_norm": "normalised MCC",
}
# Below this, the floor correction (AUC-floor)/(1-floor) is the identity.
FLOOR_TOL = 1e-9


def load_runs(root):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*_results.json"), recursive=True)):
        if f"{os.sep}old{os.sep}" in path:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping unreadable {path}: {exc}")
            continue
        if "results" not in d or "metadata" not in d:
            continue
        d["_path"] = path
        runs.append(d)
    if not runs:
        raise SystemExit(f"No *_results.json found under {root!r}")
    return runs


def group_runs(runs):
    """
    Group runs by (dataset, model, scheme, K). K is part of the identity of an
    f_dist run -- averaging across K would silently mix different estimators.
    """
    groups = defaultdict(list)
    for d in runs:
        md = d["metadata"]
        groups[(md.get("dataset", "?"), md.get("model", "?"), md.get("scheme", "?"),
                md.get("f_dist_avg_points"))].append(d)
    return groups


def audit(groups):
    """Integrity checks on the run set; independent of which metric is tabulated."""
    noseed, gridwarn, dupes = set(), set(), {}
    for (dataset, model, scheme, kval), ds in groups.items():
        seeds = [d["metadata"].get("seed") for d in ds]
        if any(s is None for s in seeds):
            noseed.add((dataset, model, scheme))
        # A repeated seed within a group means the same run landed twice, which
        # would inflate n and shrink the std. Never silently average it.
        rep = sorted({s for s in seeds if s is not None and seeds.count(s) > 1})
        if rep:
            dupes[(dataset, model, scheme, kval)] = rep
        if len({tuple(np.round(d["results"]["pruning_ratio"], 6)) for d in ds}) > 1:
            gridwarn.add((dataset, model, scheme))
    return noseed, gridwarn, dupes


def auc(ratios, values):
    r = np.asarray(ratios, dtype=float)
    v = np.asarray(values, dtype=float)
    span = r[-1] - r[0]
    if span <= 0:
        return float("nan")
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(v, r) / span)


def fmt(mean, std, places=3, seeds=1):
    if np.isnan(mean):
        return "--"
    if seeds < 2 or np.isnan(std):
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} ± {std:.{places}f}"


def fmt_time(mean, std, seeds=1):
    if np.isnan(mean):
        return "--"

    def unit(x):
        if x < 90:
            return f"{x:.1f} s"
        if x < 5400:
            return f"{x / 60:.1f} min"
        if x < 172800:
            return f"{x / 3600:.1f} h"
        return f"{x / 86400:.2f} d"

    if seeds < 2 or np.isnan(std) or std == 0:
        return unit(mean)
    # express the spread in the same unit as the mean
    scale = 1.0 if mean < 90 else (60.0 if mean < 5400 else (3600.0 if mean < 172800 else 86400.0))
    return f"{unit(mean)} ± {std / scale:.1f}"


def build(runs, metric):
    """
    (dataset, model) -> (scheme, K) -> summary cells, for one metric.

    Returns `(table, missing)`; `missing` names the groups that did not record
    this metric, which happens when a run trimmed `evaluation.metrics`.
    """
    table = defaultdict(dict)
    missing = set()
    for (dataset, model, scheme, kval), ds in sorted(group_runs(runs).items(),
                                                     key=lambda kv: str(kv[0])):
        if any(metric not in d["results"] for d in ds):
            missing.add((dataset, model, scheme))
            continue
        n = len(ds)
        aucs = np.array([auc(d["results"]["pruning_ratio"], d["results"][metric]) for d in ds])
        floors = np.array([np.asarray(d["results"][metric], dtype=float)[-1] for d in ds])
        corr = (aucs - floors) / (1.0 - floors)
        times = np.array([d.get("timing", {}).get("sweep_seconds", np.nan) for d in ds],
                         dtype=float)
        # pre-instrumentation runs carry no timing at all; report "--" rather than
        # letting nanmean warn on an all-NaN slice
        has_t = np.isfinite(times).any()

        table[(dataset, model)][(scheme, kval)] = {
            "n": n,
            "auc": (aucs.mean(), aucs.std(ddof=1) if n > 1 else np.nan),
            "corr": (corr.mean(), corr.std(ddof=1) if n > 1 else np.nan),
            "floor": float(floors.mean()),
            "time": ((np.nanmean(times), np.nanstd(times, ddof=1) if n > 1 else np.nan)
                     if has_t else (np.nan, np.nan)),
        }
    return table, missing


def _rows_for(schemes):
    """Order rows by scheme, then by K. Rows are keyed (scheme, K)."""
    def rank(key):
        sch, k = key
        return (SCHEME_ORDER.index(sch) if sch in SCHEME_ORDER else len(SCHEME_ORDER),
                sch, -1 if k is None else k)
    return sorted(schemes, key=rank)


def _label(key, show_k):
    sch, k = key
    name = SCHEME_LABEL.get(sch, sch)
    return f"{name} (K={k})" if show_k and k is not None else name


def build_json(runs, root):
    """
    Full aggregated dataset for plotting: every metric curve, per seed and as
    mean/std across seeds, plus AUC and compute time.

    Written as JSON so figures can be produced later without re-reading the whole
    results tree. Per-seed arrays are kept alongside the summaries so error bands,
    individual traces and significance tests are all still possible downstream.
    All four recorded metrics are exported, not just the two that are tabulated.
    """
    out = []
    for (dataset, model, scheme, kval), ds in sorted(group_runs(runs).items(),
                                                     key=lambda kv: str(kv[0])):
        ds = sorted(ds, key=lambda d: (d["metadata"].get("seed") is None,
                                       d["metadata"].get("seed", 0)))
        seeds = [d["metadata"].get("seed") for d in ds]
        n = len(ds)
        grids = {tuple(np.round(d["results"]["pruning_ratio"], 6)) for d in ds}
        consistent = len(grids) == 1
        ratios = list(ds[0]["results"]["pruning_ratio"])

        entry = {
            "dataset": dataset,
            "model": model,
            "scheme": scheme,
            "K": kval,
            "n_seeds": n,
            "seeds": seeds,
            "pruning_ratio": ratios,
            "sweep_grid_consistent": consistent,
            "curves": {},
            "auc": {},
            "auc_floor_corrected": {},
            "warm_start": ds[0]["metadata"].get("warm_start"),
            "fim_subset_size": ds[0]["metadata"].get("fim_subset_size"),
            "source_files": [d["_path"] for d in ds],
        }

        # A run may have trimmed evaluation.metrics; only export what every run
        # in the group actually recorded.
        entry["metrics"] = [m for m in METRICS if all(m in d["results"] for d in ds)]
        for m in entry["metrics"]:
            per_seed = [list(map(float, d["results"][m])) for d in ds]
            aucs = np.array([auc(d["results"]["pruning_ratio"], d["results"][m]) for d in ds])
            floors = np.array([np.asarray(d["results"][m], dtype=float)[-1] for d in ds])
            corr = (aucs - floors) / (1.0 - floors)
            cur = {"per_seed": per_seed}
            if consistent:
                # mean/std curves only make sense on a common grid
                A = np.asarray(per_seed, dtype=float)
                cur["mean"] = A.mean(axis=0).tolist()
                cur["std"] = (A.std(axis=0, ddof=1).tolist() if n > 1
                              else [0.0] * A.shape[1])
            entry["curves"][m] = cur
            entry["auc"][m] = {
                "per_seed": aucs.tolist(),
                "mean": float(aucs.mean()),
                "std": float(aucs.std(ddof=1)) if n > 1 else None,
            }
            entry["auc_floor_corrected"][m] = {
                "per_seed": corr.tolist(),
                "mean": float(corr.mean()),
                "std": float(corr.std(ddof=1)) if n > 1 else None,
            }

        times = np.array([d.get("timing", {}).get("sweep_seconds", np.nan) for d in ds],
                         dtype=float)
        entry["compute_seconds"] = {
            "per_seed": [None if not np.isfinite(t) else float(t) for t in times],
            "mean": float(np.nanmean(times)) if np.isfinite(times).any() else None,
            "std": (float(np.nanstd(times, ddof=1))
                    if np.isfinite(times).sum() > 1 else None),
            "device": ds[0].get("timing", {}).get("device"),
            "workers": ds[0].get("timing", {}).get("probe_workers"),
        }
        out.append(entry)

    return {
        "schema": "fdist-aggregate-v1",
        "root": root,
        "metrics": METRICS,
        "table_metrics": TABLE_METRICS,
        "note": ("AUC is the mean normalised metric over the swept pruning range "
                 "(trapezoid / range). floor = metric at 100% pruning "
                 "(chance/baseline); floor-corrected = (AUC-floor)/(1-floor), which "
                 "is the fair cross-architecture number. For mcc_norm the floor is 0 "
                 "(a collapsed one-class predictor scores 0), so the correction is "
                 "the identity there."),
        "groups": out,
    }


def render(table, metric, style):
    label = METRIC_LABEL.get(metric, metric)
    out = []
    for (dataset, model), schemes in sorted(table.items()):
        keys = _rows_for(schemes)
        # Tag a row with K only when THAT scheme appears at more than one K.
        # Tagging every row because the table as a whole mixes K would put a
        # meaningless "(K=3)" next to magnitude.
        kcount = defaultdict(set)
        for sch, k in keys:
            if k is not None:
                kcount[sch].add(k)
        multi_k = {sch for sch, ks in kcount.items() if len(ks) > 1}
        base = next((schemes[k]["auc"][0] for k in keys if k[0] == "magnitude"), float("nan"))
        nseeds = sorted({c["n"] for c in schemes.values()})
        # (AUC-floor)/(1-floor) is the identity at floor 0, which is exactly what
        # MCC gives: a collapsed one-class predictor scores 0. A column that
        # duplicates AUC is noise, so only show it where the floor is real.
        show_floor = any(abs(schemes[key]["floor"]) > FLOOR_TOL for key in keys)
        head = (f"{model} — {dataset}   (AUC of {label} over the pruning range; "
                f"seeds: {'/'.join(map(str, nseeds))})")
        note = (None if show_floor else
                f"floor is 0 for {label}, so the floor correction is the identity "
                f"and its column is omitted")

        def row(key, pm="±", times="×"):
            c = schemes[key]
            cells = [_label(key, key[0] in multi_k), fmt(*c["auc"], seeds=c["n"])]
            if show_floor:
                cells.append(fmt(*c["corr"], seeds=c["n"]))
            cells.append("--" if np.isnan(base) or base == 0
                         else f"{c['auc'][0] / base:.2f}{times}")
            cells.append(fmt_time(*c["time"], seeds=c["n"]))
            return [x.replace("±", pm) for x in cells]

        if style == "latex":
            hdr = ["Scheme", "AUC"] + (["floor-corr."] if show_floor else []) \
                + [r"vs.\ magnitude", "Compute"]
            out.append(f"% {head}")
            if note:
                out.append(f"% {note}")
            out.append(r"\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}")
            out.append(r"\toprule")
            out.append(" & ".join(hdr) + r" \\")
            out.append(r"\midrule")
            for key in keys:
                out.append(" & ".join(row(key, pm=r"$\pm$", times=r"$\times$")) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append("")
        else:
            hdr = ["Scheme", "AUC"] + (["floor-corr."] if show_floor else []) \
                + ["vs. magnitude", "Compute", "seeds"]
            out.append(f"### {head}\n")
            out.append("| " + " | ".join(hdr) + " |")
            out.append("|" + "---|" * len(hdr))
            for key in keys:
                out.append("| " + " | ".join(row(key) + [str(schemes[key]["n"])]) + " |")
            if note:
                out.append(f"\n_{note[0].upper() + note[1:]}._")
            out.append("")
    return "\n".join(out)


def render_by_k(table, metric, style, scheme="f_dist"):
    """
    K-sweep table: for one scheme, how AUC and compute scale with the number of
    path-average points K. Cost is proportional to (K-1) probes per coordinate,
    so the compute column is the price of the accuracy column.
    """
    label = METRIC_LABEL.get(metric, metric)
    out = []
    for (dataset, model), schemes in sorted(table.items()):
        rows = sorted(((k, c) for (s, k), c in schemes.items() if s == scheme and k is not None),
                      key=lambda kc: kc[0])
        if len(rows) < 2:
            continue
        ref = rows[0][1]["auc"][0]
        nseeds = sorted({c["n"] for _, c in rows})
        head = (f"{model} — {dataset}: exact f_dist vs K "
                f"(AUC of {label}; seeds: {'/'.join(map(str, nseeds))})")
        if style == "latex":
            pm = r"$\pm$"
            out.append(f"% {head}")
            out.append(r"\begin{tabular}{rrrrr}")
            out.append(r"\toprule")
            out.append(r"$K$ & probes/coord & AUC & $\Delta$ vs $K{=}$"
                       + str(rows[0][0]) + r" & Compute \\")
            out.append(r"\midrule")
            for k, c in rows:
                a = fmt(*c["auc"], places=4, seeds=c["n"]).replace("±", pm)
                t = fmt_time(*c["time"], seeds=c["n"]).replace("±", pm)
                out.append(f"{k} & {k - 1} & {a} & {c['auc'][0] - ref:+.4f} & {t} " + r"\\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append("")
        else:
            out.append(f"### {head}\n")
            out.append(f"| K | probes/coord | AUC | Δ vs K={rows[0][0]} | Compute | seeds |")
            out.append("|---|---|---|---|---|---|")
            for k, c in rows:
                out.append(f"| {k} | {k - 1} | {fmt(*c['auc'], places=4, seeds=c['n'])} | "
                           f"{c['auc'][0] - ref:+.4f} | "
                           f"{fmt_time(*c['time'], seeds=c['n'])} | {c['n']} |")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate multi-seed pruning results into the paper tables.")
    ap.add_argument("--root", default=os.environ.get("FDIST_RESULTS_DIR", "results"))
    ap.add_argument("--metrics", "--metric", dest="metrics", nargs="+", metavar="METRIC",
                    default=TABLE_METRICS, choices=METRICS,
                    help="render one table per metric (default: acc_norm mcc_norm)")
    ap.add_argument("--format", dest="style", default="markdown",
                    choices=["markdown", "latex", "json"])
    ap.add_argument("--out", default=None, help="also write the rendered tables here")
    ap.add_argument("--by-k", dest="by_k", action="store_true",
                    help="render the K-sweep table (AUC and compute vs f_dist_avg_points)")
    ap.add_argument("--k-scheme", default="f_dist", help="scheme to use for --by-k")
    args = ap.parse_args()

    runs = load_runs(args.root)

    if args.style == "json":
        blob = build_json(runs, args.root)
        text = json.dumps(blob, indent=2)
        print(f"{len(blob['groups'])} groups (dataset x model x scheme x K); "
              f"{sum(g['n_seeds'] for g in blob['groups'])} runs")
        for g in blob["groups"]:
            k = "" if g["K"] is None else f" K={g['K']}"
            print(f"  {g['model']}/{g['dataset']}/{g['scheme']}{k}: "
                  f"{g['n_seeds']} seeds {g['seeds']}")
        if not args.out:
            print("\n(pass --out FILE to write the JSON)")
            return
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nWrote {args.out}")
        return

    sections, absent = [], []
    for metric in args.metrics:
        table, missing = build(runs, metric)
        for d, m, sch in sorted(missing):
            absent.append(f"  ! {m}/{d}/{sch}: no {metric} recorded "
                          "(evaluation.metrics was trimmed for that run) — row omitted")
        sections.append(render_by_k(table, metric, args.style, args.k_scheme) if args.by_k
                        else render(table, metric, args.style))
    text = "\n".join(s for s in sections if s.strip())
    if not text.strip():
        text = "(no scheme with more than one K value found)"
    print(text)

    for line in absent:
        print(line)

    # Integrity warnings depend on the run set, not the metric, so report once.
    noseed, gridwarn, dupes = audit(group_runs(runs))
    for d, m, s in sorted(noseed):
        print(f"  ! {m}/{d}/{s}: at least one run has no seed recorded (pre-seeding run?)")
    for d, m, s in sorted(gridwarn):
        print(f"  ! {m}/{d}/{s}: runs use DIFFERENT sweep grids — AUCs are not comparable")
    for (d, m, s, k), rep in sorted(dupes.items(), key=lambda kv: str(kv[0])):
        kt = "" if k is None else f" K={k}"
        print(f"  !! {m}/{d}/{s}{kt}: DUPLICATE runs for seed(s) {rep} — these are being "
              f"averaged as if independent. Delete the extra run directories.")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

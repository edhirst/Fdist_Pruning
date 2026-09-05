"""
Run one pruning scheme over a sparsity sweep and record what it costs.

    python -m src.run_pruning [config.yaml]      # defaults to src/config.yaml

Loads the checkpoint named by the config (or trained by src/train_model.py),
then walks `pruning.sweep` from `start` to `end` in steps of `step`, pruning to
each ratio and evaluating every metric in `evaluation.metrics` on the test set.
`magnitude`, `fim` and `f_dist_one_shot` re-prune a fresh copy of the dense model
to each absolute ratio; `f_dist_iterative`, `f_dist_global` and `f_dist` advance
one running model by `step` and recompute the Fisher on the current pruned state
(see ITERATIVE_SCHEMES below).

Everything for one run lands in

    {FDIST_RESULTS_DIR | paths.results_dir}/{arch}_{dataset}[_seed{N}]/{scheme}_..._{timestamp}/

as `{scheme}_results.json` (per-ratio metrics, their normalised forms, AUCs,
wall-clock and the resolved config) plus one plot per metric. The timestamp and
the seed suffix are what let the whole HPC grid run concurrently without
collisions.

Environment overrides, used by the cluster scripts: FDIST_SEED
(experiment.seed), FDIST_K (pruning.f_dist_avg_points), FDIST_RESULTS_DIR
(the output root), FDIST_WORKERS (exact-f_dist probe sharding).
"""
import os
import sys
import yaml
import copy
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe (HPC compute nodes have no display)
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from .utils.data_loader import build_dataloaders
from .utils.model_builder import build_model_from_config, get_model_type, resolve_checkpoint_path
from .utils.evaluation import (
    evaluate_metrics, get_model_size_kb, count_nonzero_params, calculate_auc
)

# Pruners
from .pruning.magnitude_pruning import MagnitudePruner
from .pruning.fim_pruning import FIMPruner
from .pruning.f_dist_one_shot import FDistOneShotPruner
from .pruning.f_dist_iterative import FDistIterativePruner
from .pruning.f_dist import FDistPruner
from .pruning.f_dist_global import FDistGlobalPruner
from .pruning.prunable import resolve_prunable_exclude
from .utils.seeding import resolve_seed, seed_everything
from .utils.fisher_parallel import resolve_workers

# Schemes whose scoring needs Fisher information (and hence a train_loader)
FIM_SCHEMES = {"fim", "f_dist_one_shot", "f_dist_iterative", "f_dist", "f_dist_global"}

# evaluation.metrics -> (results-JSON key, plot y-label, plot filename stem).
# Insertion order is the canonical order: it fixes the column order in the printed
# sweep and the key order in the results JSON. The values themselves come from
# evaluation.evaluate_metrics, which derives all of them from one forward pass.
METRIC_REGISTRY = {
    "accuracy":  ("acc_norm", "Normalized Accuracy", "accuracy"),
    "precision": ("precision_norm", "Normalized Precision", "precision"),
    "f1":        ("f1_norm", "Normalized F1-score", "f1"),
    "mcc":       ("mcc_norm", "Normalized MCC", "mcc"),
}
METRIC_ALIASES = {"acc": "accuracy", "f1-score": "f1", "f1_score": "f1", "f-score": "f1",
                  "matthews": "mcc", "matthews_corrcoef": "mcc"}

# Schemes the magnitude warm-start applies to
FDIST_FAMILY = ("f_dist", "f_dist_iterative", "f_dist_global", "f_dist_one_shot")

# Schemes that step a running model by a delta each sweep point, rather than
# re-pruning a fresh copy of the baseline to an absolute ratio
ITERATIVE_SCHEMES = ("f_dist_iterative", "f_dist_global", "f_dist")


# load config from yaml
def load_config(config_path: str):
    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config


# Use GPU if available else CPU
def get_device():
    # CUDA first, then MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def make_sweep_ratios(start: float, end: float, step: float):
    """
    Ensure end is included even if step doesn't land exactly on it.
    E.g., step=0.03 -> ... 0.99, 1.00
    """
    if step <= 0:
        raise ValueError("sweep.step must be > 0")

    ratios = []
    x = float(start)
    end = float(end)
    step = float(step)

    # accumulate with tolerance
    while x < end - 1e-12:
        ratios.append(x)
        x += step

    if not ratios or abs(ratios[-1] - end) > 1e-12:
        ratios.append(end)

    # clamp to [0, 1]
    ratios = [min(1.0, max(0.0, r)) for r in ratios]
    # unique & sorted
    ratios = sorted(set(ratios))
    return ratios


def make_fim_loader_subset(train_loader, subset_size: int, seed: int, shuffle: bool):
    """
    For FIM-based methods, optionally use a subset of training data for speed/reproducibility.
    """
    dataset = train_loader.dataset
    n = len(dataset)
    subset_size = min(int(subset_size), n)

    g = torch.Generator()
    g.manual_seed(int(seed))

    if shuffle:
        idx = torch.randperm(n, generator=g)[:subset_size].tolist()
    else:
        idx = list(range(subset_size))

    subset = Subset(dataset, idx)
    nw = train_loader.num_workers
    pm = train_loader.pin_memory
    return DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=pm, persistent_workers=(nw > 0))


def resolve_metrics(config):
    """
    Which metrics to evaluate at every sweep point (`evaluation.metrics`).

    Defaults to all of them, which is what the paper runs. They are all derived
    from a single forward pass (see evaluation.evaluate_metrics), so trimming the
    list changes what is recorded rather than what it costs. Accuracy cannot be
    dropped: the plots, the AUC and src/aggregate_results.py are keyed on
    acc_norm.
    """
    raw = (config.get("evaluation", {}) or {}).get("metrics")
    if raw is None:
        return list(METRIC_REGISTRY)
    if isinstance(raw, str):
        raw = [raw]

    chosen = set()
    for item in raw:
        key = str(item).strip().lower()
        key = METRIC_ALIASES.get(key, key)
        if key not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown evaluation.metrics entry {item!r}. Supported: "
                f"{', '.join(METRIC_REGISTRY)} (aliases: {', '.join(METRIC_ALIASES)})."
            )
        chosen.add(key)
    if "accuracy" not in chosen:
        raise ValueError("evaluation.metrics must include 'accuracy': the pruning "
                         "curves, the AUC and the results tables are all keyed on it.")
    return [m for m in METRIC_REGISTRY if m in chosen]


def build_pruner(config, scheme: str):
    """
    Return a pruner instance. We will always call:
        pruner.set_parameters({...})
        pruner.apply_pruning(model, train_loader=..., device=...)
    """
    p_cfg = config.get("pruning", {})
    prunable_exclude = resolve_prunable_exclude(p_cfg, get_model_type(config))

    if scheme == "magnitude":
        pruner = MagnitudePruner(threshold=0.0)
        pruner.set_parameters({"pruning_threshold": 0.0, "prunable_exclude": prunable_exclude})
        return pruner

    if scheme == "fim":
        return FIMPruner(parameters={
            "pruning_threshold": 0.0,
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "nngeometry"),
            "prunable_exclude": prunable_exclude,
        })

    if scheme == "f_dist_one_shot":
        return FDistOneShotPruner(parameters={
            "pruning_threshold": 0.0,
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "nngeometry"),
            "prunable_exclude": prunable_exclude,
        })

    if scheme == "f_dist_iterative":
        # Your iterative pruner currently takes params dict at init in your snippet.
        return FDistIterativePruner(parameters={
            "pruning_step": 0.0,
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "nngeometry"),
            "prunable_exclude": prunable_exclude,
        })

    if scheme == "f_dist":
        return FDistPruner(parameters={
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "nngeometry"),
            "freeze_all_zero_tensors": True,
            "f_dist_avg_points": p_cfg.get("f_dist_avg_points", 2),
            "prunable_exclude": prunable_exclude,
            # forward-mode (JVP) fast path; identical output, ~9-325x per probe.
            # See tests/test_fdist_fast_equivalence.py.
            "f_dist_fast": p_cfg.get("f_dist_fast", True),
            "f_dist_probe_batch": p_cfg.get("f_dist_probe_batch", 32),
        })

    if scheme == "f_dist_global":
        return FDistGlobalPruner(parameters={
            "pruning_step": 0.0,
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "backprop"),
            "f_dist_avg_points": p_cfg.get("f_dist_avg_points", 3),
            "prunable_exclude": prunable_exclude,
        })

    raise ValueError(f"Unknown pruning_scheme: {scheme}")


def _magnitude_pruned_copy(baseline_model, target_ratio, start, prunable_exclude, device):
    """
    Return a fresh copy of baseline_model magnitude-pruned to ABSOLUTE sparsity
    target_ratio (fraction of all params). baseline_model may already be pruned to
    `start`, so we only prune the additional (target_ratio - start).

    Used by the magnitude warm-start: cheap magnitude pruning covers the low-sparsity
    part of the curve so the expensive f_dist scheme only runs for the tail.
    """
    m = copy.deepcopy(baseline_model).to(device)
    delta = max(0.0, float(target_ratio) - float(start))
    if delta > 0.0:
        wp = MagnitudePruner(threshold=0.0)
        wp.set_parameters({"pruning_threshold": delta, "prunable_exclude": prunable_exclude})
        m = wp.apply_pruning(m, train_loader=None, device=device)
    m.eval()
    return m


def plot_metric(ratios, values, ylabel, save_path=None, dpi=300):
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, values, marker="o")
    plt.grid(True, alpha=0.5)
    plt.xlabel("Pruning ratio", fontdict={"fontsize": 14})
    plt.ylabel(ylabel, fontdict={"fontsize": 14})

    # normalized metric range
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("src", "config.yaml")
    config = load_config(config_path)

    p_cfg = config.get("pruning", {}) or {}
    if not p_cfg.get("enable_pruning", False):
        print("pruning.enable_pruning is False. Exiting.")
        return

    scheme = str(p_cfg.get("pruning_scheme", "magnitude")).lower()
    metric_names = resolve_metrics(config)

    # FDIST_K overrides pruning.f_dist_avg_points, so one config drives the whole
    # K-sweep (how the exact-f_dist path-average converges, and what it costs).
    _k_env = os.environ.get("FDIST_K")
    k_override = None
    if _k_env is not None and _k_env.strip() != "":
        k_override = int(_k_env)
        if k_override < 2:
            raise ValueError(f"FDIST_K must be >= 2, got {k_override}")
        p_cfg["f_dist_avg_points"] = k_override
        print(f"f_dist_avg_points (K): {k_override}  [FDIST_K]")

    run_seed = resolve_seed(config)
    if run_seed is not None:
        seed_everything(run_seed)
        print(f"seed: {run_seed}")

    device = get_device()
    model_type = get_model_type(config)
    prunable_exclude = resolve_prunable_exclude(p_cfg, model_type)

    # Fail fast: KFAC (nngeometry) cannot represent transformer layers
    fim_method = str(p_cfg.get("fim_calculate_method", "nngeometry")).lower()
    if scheme in FIM_SCHEMES and model_type == "Transformer" and fim_method in ("nngeometry", "nngeo"):
        raise ValueError(
            "fim_calculate_method='nngeometry' is not supported for transformer models "
            "(PMatKFAC lacks LayerNorm/attention support). "
            "Set pruning.fim_calculate_method: 'backprop'."
        )

    # Optional escape hatch: run FIM-based sweeps on CPU (torch.func per-sample
    # gradients can be slower or unsupported on some accelerators)
    if scheme in FIM_SCHEMES and str(p_cfg.get("fim_device", "auto")).lower() == "cpu":
        device = torch.device("cpu")
        print("pruning.fim_device=cpu -> running this sweep on CPU")

    dataset_name, train_loader, test_loader = build_dataloaders(config, augment=False)

    # Build model architecture and load checkpoint
    model, arch_name = build_model_from_config(config, dataset_name=dataset_name)
    model = model.to(device)

    ckpt_path = resolve_checkpoint_path(config, arch_name, dataset_name, seed=run_seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train it first: python -m src.train_model"
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Sweep ratios
    sweep_cfg = p_cfg.get("sweep", {}) or {}
    start = float(sweep_cfg.get("start", 0.0))
    end = float(sweep_cfg.get("end", 1.0))
    step = float(sweep_cfg.get("step", 0.05))
    ratios = make_sweep_ratios(start, end, step)
    print(f"pruning method: {scheme}")
    print(f"Pruning Range: [{start}, {end}, {step}]")

    baseline_model = copy.deepcopy(model).to(device)
    if start > 0.0:
        warm_pruner = MagnitudePruner(threshold=0.0)
        warm_pruner.set_parameters({"pruning_threshold": start, "prunable_exclude": prunable_exclude})
        baseline_model = warm_pruner.apply_pruning(baseline_model, train_loader=None, device=device)
        baseline_model.eval()

    # Get plot configuration
    plot_cfg = config.get("plots", {}) or {}
    save_plots = bool(plot_cfg.get("save_plots", True))
    plot_formats = plot_cfg.get("plot_formats", ["png"])
    plot_dpi = plot_cfg.get("dpi", 300)
    if plot_dpi is None:
        plot_dpi = 300

    # Get fold info for directory naming
    cv_cfg = config.get("cross_validation", {}) or {}
    current_fold = cv_cfg.get("current_fold", None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_suffix = f"_fold{current_fold}" if current_fold is not None else ""

    # Output root: env override (handy for HPC) > paths.results_dir > "results".
    # Group every run under a per-(architecture, dataset) subdirectory so that
    # concurrent jobs for different models never collide on output files — the
    # scheme + timestamp alone are NOT unique across architectures.
    paths_cfg = config.get("paths", {}) or {}
    results_root = os.environ.get("FDIST_RESULTS_DIR") or paths_cfg.get("results_dir") or "results"
    run_group = f"{arch_name}_{dataset_name}"
    seed_suffix = "" if run_seed is None else f"_seed{run_seed}"
    k_suffix = f"_K{k_override}" if k_override is not None else ""
    results_dir = os.path.join(
        results_root, run_group,
        f"{scheme}_pruning_result_figure_{timestamp}{fold_suffix}{seed_suffix}{k_suffix}"
    )
    os.makedirs(results_dir, exist_ok=True)
    if save_plots:
        print(f"Saving result figures to: {results_dir}")
    else:
        print(f"Plot saving disabled (save_plots=False). Results JSON only: {results_dir}")

    # Baseline evaluation. Every curve is normalised by these, so 1.0 always means
    # "as good as the dense model"; eps guards a baseline that is itself ~0.
    eps = 1e-12
    baselines = {name: max(value, eps) for name, value
                 in evaluate_metrics(model, test_loader, metric_names, device=device).items()}
    base_nonzero, base_total = count_nonzero_params(model)
    base_size_kb = get_model_size_kb(model)


    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {arch_name}")
    print(f"Checkpoint: {ckpt_path}")
    print("Baseline | " + f"acc={baselines['accuracy']*100:.2f}%"
          + "".join(f" {n}={baselines[n]:.4f}" for n in metric_names if n != "accuracy"))
    print(f"Baseline | nonzero={base_nonzero}/{base_total} size={base_size_kb:.2f}KB")
    print("-" * 80)


    # For FIM-based schemes, optionally use subset loader
    use_fim_loader = scheme in FIM_SCHEMES
    if use_fim_loader:
        fim_subset_size = int(p_cfg.get("fim_subset_size", 0))
        fim_seed = int(p_cfg.get("fim_subset_seed", 42))
        if run_seed is not None:
            fim_seed += run_seed
        fim_shuffle = bool(p_cfg.get("fim_subset_shuffle", True))
        if fim_subset_size > 0:
            fim_loader = make_fim_loader_subset(train_loader, fim_subset_size, fim_seed, fim_shuffle)
        else:
            fim_loader = train_loader
    else:
        fim_loader = None

    # Magnitude warm-start config: for the f_dist family, prune 0->warm_ratio by
    # magnitude and only run the expensive f_dist scheme for the warm_ratio->end
    # tail (the full curve is still produced). Does not touch magnitude/fim.
    warm_cfg = p_cfg.get("warm_start", {}) or {}
    warm_enabled = bool(warm_cfg.get("enabled", True))
    warm_ratio = min(1.0, max(0.0, float(warm_cfg.get("ratio", 0.8))))
    use_warm = warm_enabled and scheme in FDIST_FAMILY and warm_ratio > start + 1e-12
    if use_warm:
        print(f"warm-start: magnitude 0->{warm_ratio:.2f}, then {scheme} for the {warm_ratio:.2f}->{end:.2f} tail")
        if warm_ratio >= end - 1e-12:
            print(f"  (note: warm_start.ratio {warm_ratio} >= sweep end {end}; the whole curve is magnitude)")

    # Storage for curves and output JSON file
    results_json = {
        "metadata": {
            "scheme": scheme,
            "dataset": dataset_name,
            "model": arch_name,
            "timestamp": timestamp,
            "warm_start": {"enabled": bool(use_warm), "ratio": warm_ratio} if scheme in FDIST_FAMILY else {"enabled": False},
        },
        "results": {
            "pruning_ratio": [],
            **{METRIC_REGISTRY[n][0]: [] for n in metric_names},
        }
    }

    # Wall-clock accounting. `sweep_seconds` is the number the paper's compute
    # column reports: everything from the first pruning step to the last
    # evaluation, excluding process start-up, data loading and baseline eval.
    sweep_t0 = time.perf_counter()
    step_times = []
    last_t = sweep_t0

    def _eval_and_record(m, r):
        """Evaluate model m at sparsity r and append normalized metrics + print."""
        vals = evaluate_metrics(m, test_loader, metric_names, device=device)
        results_json["results"]["pruning_ratio"].append(r)
        for name, v in vals.items():
            results_json["results"][METRIC_REGISTRY[name][0]].append(v / baselines[name])
        nonlocal last_t
        now = time.perf_counter()
        step_times.append(now - last_t)
        last_t = now
        print(f"pruning ratio: {r*100:>5.1f}%, accuracy: {vals['accuracy']*100:.2f}%"
              + "".join(f" | {n}: {vals[n]:.4f}" for n in metric_names if n != "accuracy")
              + f" | {step_times[-1]:.2f}s")


    # One-shot sweep: to guarantee exact prune ratio semantics (prune ratio of original),
    # we evaluate each ratio from a fresh copy of the baseline model.
    # This avoids ambiguity when zeros accumulate across steps.
    if scheme in ITERATIVE_SCHEMES:
        # Delta-stepped schemes: one running model advanced by `step` at each sweep
        # point, so the Fisher is always recomputed on the current pruned state.
        pruner = build_pruner(config, scheme)
        params = {
            "fim_calculate_method": fim_method,
            "pruning_step": step,
            "prunable_exclude": prunable_exclude,
        }
        if scheme != "f_dist":
            # Exact f_dist takes K from build_pruner (whose default differs); the
            # others are explicit here.
            params["f_dist_avg_points"] = p_cfg.get("f_dist_avg_points", 3)
        pruner.set_parameters(params)

        pruned_model = None        # f_dist working model, initialised lazily
        last_warm_ratio = start    # magnitude sparsity where f_dist takes over

        for r in ratios:
            if abs(r - start) < 1e-12:
                m = baseline_model
            elif use_warm and r <= warm_ratio + 1e-12:
                # warm phase: cheap magnitude point (no Fisher eval)
                m = _magnitude_pruned_copy(baseline_model, r, start, prunable_exclude, device)
                last_warm_ratio = r
            else:
                # f_dist tail: continue delta-stepping from the warm (or baseline) state
                if pruned_model is None:
                    pruned_model = (
                        _magnitude_pruned_copy(baseline_model, last_warm_ratio, start, prunable_exclude, device)
                        if use_warm else copy.deepcopy(baseline_model).to(device)
                    )
                pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device)
                m = pruned_model

            _eval_and_record(m, r)

    else:
        # One-shot schemes (magnitude, fim, f_dist_one_shot): each sweep point is
        # pruned from a fresh copy of the baseline to the absolute target ratio, so
        # zeros never accumulate ambiguously across steps.
        for r in ratios:
            pruned_model = copy.deepcopy(baseline_model).to(device)

            pruner = build_pruner(config, scheme)

            delta = max(0.0, float(r - start))

            # Unified dict-style parameter setting
            if scheme == "magnitude":
                pruner.set_parameters({"pruning_threshold": delta, "prunable_exclude": prunable_exclude})
                pruned_model = pruner.apply_pruning(pruned_model, train_loader=None, device=device)

            elif scheme == "fim":
                pruner.set_parameters({
                    "pruning_threshold": delta,
                    "fim_calculate_method": fim_method,
                    "prunable_exclude": prunable_exclude,
                })
                pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device)

            elif scheme == "f_dist_one_shot":
                if use_warm and r <= warm_ratio + 1e-12:
                    # warm phase: cheap magnitude point (no Fisher eval)
                    wp = MagnitudePruner(threshold=0.0)
                    wp.set_parameters({"pruning_threshold": delta, "prunable_exclude": prunable_exclude})
                    pruned_model = wp.apply_pruning(pruned_model, train_loader=None, device=device)
                else:
                    if use_warm:
                        # magnitude to warm_ratio, then one-shot f_dist for the remainder
                        wp = MagnitudePruner(threshold=0.0)
                        wp.set_parameters({
                            "pruning_threshold": max(0.0, warm_ratio - start),
                            "prunable_exclude": prunable_exclude,
                        })
                        pruned_model = wp.apply_pruning(pruned_model, train_loader=None, device=device)
                        os_delta = max(0.0, float(r - warm_ratio))
                    else:
                        os_delta = delta
                    pruner.set_parameters({
                        "pruning_threshold": os_delta,
                        "fim_calculate_method": fim_method,
                        "prunable_exclude": prunable_exclude,
                    })
                    pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device)

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            _eval_and_record(pruned_model, r)


    # Plots (4 separate figures)
    ratios = results_json["results"]["pruning_ratio"]

    if save_plots:
        for metric_key, ylabel, metric_name in (METRIC_REGISTRY[n] for n in metric_names):
            for fmt in plot_formats:
                save_path = os.path.join(results_dir, f"{scheme}_{metric_name}.{fmt}")
                plot_metric(ratios, results_json["results"][metric_key],
                           ylabel=ylabel, save_path=save_path, dpi=plot_dpi)


    # AUC of each normalized-metric curve over the swept range (higher = accuracy
    # retained longer under pruning)
    results_json["auc"] = {
        METRIC_REGISTRY[n][0]: calculate_auc(ratios, results_json["results"][METRIC_REGISTRY[n][0]])
        for n in metric_names
    }
    print("AUC:", {k: round(v, 4) for k, v in results_json["auc"].items()})

    # Compute cost of this run (the paper's "avg compute time" column) plus the
    # context needed to interpret it -- a time is meaningless without the device
    # and thread count it was measured on.
    results_json["timing"] = {
        "sweep_seconds": time.perf_counter() - sweep_t0,
        "step_seconds": step_times,
        "n_steps": len(step_times),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        # how the exact-f_dist probes were parallelised; a compute time is only
        # comparable against another measured the same way
        "probe_workers": (resolve_workers(p_cfg.get("f_dist_workers", 1))
                          if scheme == "f_dist" else 1),
    }
    results_json["metadata"]["seed"] = run_seed
    results_json["metadata"]["sweep"] = {"start": start, "end": end, "step": step}
    # K only means something for the schemes that actually average along the path;
    # recording it for magnitude/fim would split their groups on a phantom key.
    results_json["metadata"]["f_dist_avg_points"] = (
        p_cfg.get("f_dist_avg_points", 3) if scheme in ("f_dist", "f_dist_global") else None
    )
    results_json["metadata"]["fim_subset_size"] = int(p_cfg.get("fim_subset_size", 0))
    results_json["metadata"]["metrics"] = metric_names
    print(f"sweep wall-clock: {results_json['timing']['sweep_seconds']:.1f}s "
          f"on {results_json['timing']['device']}")

    json_path = os.path.join(results_dir, f"{scheme}_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Saved results JSON to: {json_path}")


    print("Done.")


if __name__ == "__main__":
    main()

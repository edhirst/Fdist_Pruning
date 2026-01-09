import os
import sys
import yaml
import numpy as np
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from .utils.data_loader import load_mnist, load_fashion_mnist
from .models.simple_cnn import SimpleCNN
from .models.simple_nn import SimpleNN
from .utils.evaluation import (
    evaluate_accuracy, evaluate_precision, evaluate_f1, evaluate_mcc,
    get_model_size_kb, count_nonzero_params
)

# Pruners
from .pruning.magnitude_pruning import MagnitudePruner
from .pruning.fim_pruning import FIMPruner
from .pruning.f_dist_one_shot import MagnitudeFIMOneShotPruner
from .pruning.magnitude_fim_iterative import MagnitudeFIMIterativePruner
from .pruning.sqrt_averaged_magnitude_fim import SqrtAveragedMagnitudeFIMPruner


# load config from yaml
def load_config(config_path: str):
    with open(config_path, "r") as file:
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



def build_dataloaders(config):
    dataset_name = config.get("dataset", {}).get("name", "mnist").lower()
    batch_size = config["training"]["batch_size"]

    if dataset_name == "mnist":
        train_loader = load_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_mnist(batch_size=batch_size, train=False, download=True)
    elif dataset_name == "fashion_mnist":
        train_loader = load_fashion_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_fashion_mnist(batch_size=batch_size, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_name, train_loader, test_loader


def build_model_from_config(config):
    model_cfg = config.get("model", {}) or {}
    common_cfg = model_cfg.get("common", {}) or {}
    model_type = common_cfg.get("model_type", "NN")
    num_classes = common_cfg.get("num_classes", 10)

    if model_type == "NN":
        nn_cfg = model_cfg.get("NN", {}) or {}
        hidden_size = nn_cfg.get("hidden_size", 32)
        hidden_layers = nn_cfg.get("hidden_layers", 2)
        model = SimpleNN(hidden_size=hidden_size, hidden_layers=hidden_layers, num_classes=num_classes)
        arch_name = f"SimpleNN_h{hidden_layers}_w{hidden_size}"
    elif model_type == "CNN":
        model = SimpleCNN(num_classes=num_classes)
        arch_name = f"SimpleCNN"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, arch_name



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
    return DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False, num_workers=0)


def build_pruner(config, scheme: str):
    """
    Return a pruner instance. We will always call:
        pruner.set_parameters({...})
        pruner.apply_pruning(model, train_loader=..., device=...)
    """
    p_cfg = config.get("pruning", {}) or {}

    if scheme == "magnitude":
        return MagnitudePruner(threshold=0.0)

    if scheme == "fim":
        return FIMPruner(parameters={
            "pruning_threshold": 0.0,
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "nngeometry"),
        })

    if scheme == "magnitude_fim_one_shot":
        return MagnitudeFIMOneShotPruner(parameters={
            "pruning_threshold": 0.0,
            "fim_calculate_method": p_cfg.get("fim_calculate_method", "nngeometry"),
        })

    if scheme == "magnitude_fim_iterative":
        # Your iterative pruner currently takes params dict at init in your snippet.
        return MagnitudeFIMIterativePruner(
            model=None,
            params={
                "pruning_threshold": 0.0,
                "iterations": int(p_cfg.get("iterative_pruning_steps", 5)),
            },
        )

    if scheme == "sqrt_averaged_magnitude_fim":
        return SqrtAveragedMagnitudeFIMPruner(
            magnitude_threshold=0.0,
            fim_threshold=float(p_cfg.get("fim_threshold", 0.8)),
        )

    raise ValueError(f"Unknown pruning_scheme: {scheme}")


def plot_metric(ratios_pct, values, title, ylabel, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(ratios_pct, values, marker="o")
    plt.grid(True)
    plt.xlabel("Pruning ratio (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("src", "config.yaml")
    config = load_config(config_path)

    p_cfg = config.get("pruning", {}) or {}
    if not p_cfg.get("enable_pruning", False):
        print("pruning.enable_pruning is False. Exiting.")
        return

    scheme = str(p_cfg.get("pruning_scheme", "magnitude")).lower()

    device = get_device()
    dataset_name, train_loader, test_loader = build_dataloaders(config)

    # Build model architecture and load checkpoint
    model, arch_name = build_model_from_config(config)
    model = model.to(device)

    ckpt_path = (config.get("paths", {}) or {}).get("pretrained_model_path", None)
    if not ckpt_path:
        raise ValueError("Missing config: paths.pretrained_model_path")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{scheme}_pruning_result_figure_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving result figures to: {results_dir}")

    # Baseline evaluation
    base_acc = evaluate_accuracy(model, test_loader, device=device)
    base_prec = evaluate_precision(model, test_loader, device=device)
    base_f1 = evaluate_f1(model, test_loader, device=device)
    base_mcc = evaluate_mcc(model, test_loader, device=device)
    base_nonzero, base_total = count_nonzero_params(model)
    base_size_kb = get_model_size_kb(model)

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {arch_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Baseline | acc={base_acc*100:.2f}% prec={base_prec:.4f} f1={base_f1:.4f} mcc={base_mcc:.4f}")
    print(f"Baseline | nonzero={base_nonzero}/{base_total} size={base_size_kb:.2f}KB")
    print("-" * 80)

    # Sweep ratios
    sweep_cfg = p_cfg.get("sweep", {}) or {}
    start = float(sweep_cfg.get("start", 0.0))
    end = float(sweep_cfg.get("end", 1.0))
    step = float(sweep_cfg.get("step", 0.05))
    ratios = make_sweep_ratios(start, end, step)
    print(f"pruning method: {scheme}")
    print(f"Pruning Range: [{start}, {end}, {step}]")

    # For FIM-based schemes, optionally use subset loader
    use_fim_loader = scheme in {
        "fim",
        "magnitude_fim_one_shot",
        "magnitude_fim_iterative",
        "sqrt_averaged_magnitude_fim",
    }
    if use_fim_loader:
        fim_subset_size = int(p_cfg.get("fim_subset_size", 0))
        fim_seed = int(p_cfg.get("fim_subset_seed", 42))
        fim_shuffle = bool(p_cfg.get("fim_subset_shuffle", True))
        if fim_subset_size > 0:
            fim_loader = make_fim_loader_subset(train_loader, fim_subset_size, fim_seed, fim_shuffle)
        else:
            fim_loader = train_loader
    else:
        fim_loader = None

    # Storage for curves
    ratios_pct = []
    accs = []
    precs = []
    f1s = []
    mccs = []

    # One-shot sweep: to guarantee exact prune ratio semantics (prune ratio of original),
    # we evaluate each ratio from a fresh copy of the baseline model.
    # This avoids ambiguity when zeros accumulate across steps.
    for r in ratios:
        pruned_model = copy.deepcopy(model).to(device)

        pruner = build_pruner(config, scheme)

        # Unified dict-style parameter setting
        if scheme == "magnitude":
            pruner.set_parameters({"pruning_threshold": r})
            pruned_model = pruner.apply_pruning(pruned_model, train_loader=None, device=device)

        elif scheme == "fim":
            pruner.set_parameters({
                "pruning_threshold": r,
                "fim_calculate_method": str(p_cfg.get("fim_calculate_method", "nngeometry")).lower(),
            })
            pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device)

        elif scheme == "magnitude_fim_one_shot":
            # assumes its apply_pruning uses its internal thresholds
            pruner.set_parameters({
                "pruning_threshold": r,
                "fim_calculate_method": str(p_cfg.get("fim_calculate_method", "nngeometry")).lower(),
            })
            pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device)

        elif scheme == "magnitude_fim_iterative":
            # NOTE: this pruner (as currently written) does iterative internally; it won't give per-step reuse for plotting.
            # We'll still plot by running one-shot-from-baseline at each ratio for correctness.
            pruner.set_parameters({
                "pruning_threshold": r,
                "iterations": int(p_cfg.get("iterative_pruning_steps", 5)),
            })
            pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device)

        elif scheme == "sqrt_averaged_magnitude_fim":
            pruner.set_parameters(magnitude_threshold=r, fim_threshold=float(p_cfg.get("fim_threshold", 0.8)))
            pruned_model = pruner.apply_pruning(pruned_model, train_loader=fim_loader, device=device, target_pruning_pct=r)

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Evaluate
        acc = evaluate_accuracy(pruned_model, test_loader, device=device)
        prec = evaluate_precision(pruned_model, test_loader, device=device)
        f1 = evaluate_f1(pruned_model, test_loader, device=device)
        mcc = evaluate_mcc(pruned_model, test_loader, device=device)

        ratios_pct.append(r * 100.0)
        accs.append(acc * 100.0)
        precs.append(prec)
        f1s.append(f1)
        mccs.append(mcc)

        # Print in the format you asked for (and keep extra metrics for debugging)
        print(f"pruning ratio: {r*100:>5.1f}%, accuracy: {acc*100:.2f}% | precision: {prec:.4f} | f1: {f1:.4f} | mcc: {mcc:.4f}")

    # Plots (4 separate figures)
    plot_metric(ratios_pct, accs,
                "Accuracy vs Pruning Ratio", "Accuracy (%)",
                save_path=os.path.join(results_dir, f"{scheme}_accuracy.png"))
    plot_metric(ratios_pct, precs,
                "Precision vs Pruning Ratio", "Precision (macro)",
                save_path=os.path.join(results_dir, f"{scheme}_precision.png"))
    plot_metric(ratios_pct, f1s,
                "F1-score vs Pruning Ratio", "F1-score (macro)",
                save_path=os.path.join(results_dir, f"{scheme}_f1.png"))
    plot_metric(ratios_pct, mccs,
                "MCC vs Pruning Ratio", "MCC",
                save_path=os.path.join(results_dir, f"{scheme}_mcc.png"))

    print("Done.")


if __name__ == "__main__":
    main()
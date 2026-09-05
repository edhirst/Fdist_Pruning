"""
Test-set metrics, AUC, and the comparison plots.

`evaluate_metrics` derives accuracy, precision, F1 and MCC from ONE forward pass
over the loader (collect_predictions, then the private _accuracy/_precision/
_f1/_mcc helpers), which is why listing more metrics in `evaluation.metrics`
changes what is recorded but not the runtime. The single-metric wrappers below
it each do their own pass and exist for interactive use;
tests/test_single_pass_metrics.py asserts the two agree.

`calculate_auc` integrates a metric curve over the pruning range with the
trapezoid rule. The curves passed to it are NORMALISED by the dense-model value,
so a scheme that held performance perfectly would score 1.0.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, precision_score, f1_score
from scipy.stats import ttest_rel
import os


@torch.no_grad()
def collect_predictions(model, dataloader, device="cpu"):
    """Argmax predictions and labels over the whole loader, as numpy arrays."""
    model.eval()
    all_preds = []
    all_labels = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds, all_labels


# Every metric below is a pure reduction of the SAME (preds, labels) pair, which
# is what lets evaluate_metrics() serve all of them from one forward pass.
def _accuracy(preds, labels):
    return float((preds == labels).mean())


def _precision(preds, labels, average="macro"):
    return float(precision_score(labels, preds, average=average, zero_division=0))


def _f1(preds, labels, average="macro"):
    return float(f1_score(labels, preds, average=average, zero_division=0))


def _mcc(preds, labels):
    return float(matthews_corrcoef(labels, preds))


METRIC_FNS = {
    "accuracy": _accuracy,
    "precision": _precision,
    "f1": _f1,
    "mcc": _mcc,
}


def evaluate_metrics(model, dataloader, names=None, device="cpu"):
    """
    Evaluate several metrics from a SINGLE pass over `dataloader`.

    The metrics differ only in how they reduce the same argmax predictions, so
    evaluating four of them separately costs four identical forward passes over
    the test set. During a pruning sweep that is the dominant cost of every step
    for the cheap schemes. Values are identical to the single-metric helpers.

    Returns {name: value}, ordered as `names` (default: all of them).
    """
    names = list(METRIC_FNS) if names is None else list(names)
    unknown = [n for n in names if n not in METRIC_FNS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(map(str, unknown))}. "
            f"Supported: {', '.join(METRIC_FNS)}."
        )
    preds, labels = collect_predictions(model, dataloader, device)
    return {n: METRIC_FNS[n](preds, labels) for n in names}


def evaluate_accuracy(model, dataloader, device="cpu"):
    return _accuracy(*collect_predictions(model, dataloader, device))


def evaluate_precision(model, dataloader, device="cpu", average="macro"):
    return _precision(*collect_predictions(model, dataloader, device), average=average)


def evaluate_f1(model, dataloader, device="cpu", average="macro"):
    return _f1(*collect_predictions(model, dataloader, device), average=average)


def evaluate_mcc(model, dataloader, device="cpu"):
    return _mcc(*collect_predictions(model, dataloader, device))



def calculate_auc(pruning_percentages, accuracies):
    """
    Calculates the Area Under the (Accuracy vs. Proportion Pruned) curve.
    Higher AUC indicates better pruning performance (accuracy is retained longer).
    """
    sorted_indices = np.argsort(pruning_percentages)
    x = np.array(pruning_percentages)[sorted_indices]
    y = np.array(accuracies)[sorted_indices]
    auc = np.trapz(y, x=x)
    return auc


def get_model_size_kb(model):
    """Calculate model size in KB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_kb = (param_size + buffer_size) / 1024.0
    return size_kb


def count_nonzero_params(model):
    """Count non-zero parameters in the model"""
    total = 0
    nonzero = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
            nonzero += torch.count_nonzero(param).item()
    return nonzero, total


# ----------------- plot -----------------


def plot_accuracy_comparison(prune_pcts, results_dict, title="Accuracy vs. Pruning Fraction", 
                             save_path=None, xlim=None, ylim=None):
    """
    Plot accuracy curves for multiple pruning methods
    
    Args:
        prune_pcts: List of pruning percentages
        results_dict: Dict with format {method_name: accuracy_list}
        title: Plot title
        save_path: Path to save figure (optional)
        xlim: X-axis limits tuple (optional)
        ylim: Y-axis limits tuple (optional)
    """
    plt.figure(figsize=(12, 7))

    colors = ['skyblue', 'orange', 'green', 'purple', 'red']
    markers = ['o', 'x', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':', '--']

    for idx, (method_name, accuracies) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        plt.plot(prune_pcts, accuracies, marker=marker, linestyle=linestyle,
                color=color, label=method_name)

    plt.title(title)
    plt.xlabel("Fraction of Weights Pruned Globally")
    plt.ylabel("Test Accuracy")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([-0.01, 1.01])
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_size_comparison(prune_pcts, results_dict, title="Model Size vs. Pruning Fraction",
                               save_path=None):
    """Plot model size curves for multiple pruning methods"""
    plt.figure(figsize=(12, 7))

    colors = ['skyblue', 'orange', 'green', 'purple', 'red']
    markers = ['o', 'x', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':', '--']

    for idx, (method_name, sizes) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        plt.plot(prune_pcts, sizes, marker=marker, linestyle=linestyle,
                color=color, label=method_name)

    plt.title(title)
    plt.xlabel("Fraction of Weights Pruned Globally")
    plt.ylabel("Model Size (KB)")
    plt.xlim([-0.01, 1.01])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metric_curves(all_metrics, metric_name, prune_pcts, save_path=None):
    """
    Plot metric curves with mean and std deviation
    
    Args:
        all_metrics: Dict with format {method_name: [curve1, curve2, ...]}
        metric_name: Name of the metric (e.g., "MCC", "Precision")
        prune_pcts: List of pruning percentages
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    colors = ['skyblue', 'orange', 'green', 'purple', 'red']

    for idx, (scheme, curves) in enumerate(all_metrics.items()):
        curves = np.array(curves)
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        color = colors[idx % len(colors)]

        plt.plot(prune_pcts, mean_curve, label=f"{scheme} (mean)", color=color)
        plt.fill_between(prune_pcts, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color=color)

    plt.xlabel("Fraction of Weights Pruned")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Pruning Fraction (Mean ± Std, Cross-Validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_auc_comparison(all_aucs, title="Mean AUC Comparison", save_path=None):
    """
    Plot bar chart comparing mean AUCs across pruning schemes
    
    Args:
        all_aucs: Dict with format {method_name: [auc1, auc2, ...]}
        title: Plot title
        save_path: Path to save figure (optional)
    """
    schemes = list(all_aucs.keys())
    mean_aucs = [np.mean(all_aucs[scheme]) for scheme in schemes]
    std_aucs = [np.std(all_aucs[scheme]) for scheme in schemes]

    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'orange', 'green', 'purple', 'red']
    plt.bar(schemes, mean_aucs, yerr=std_aucs, capsize=6,
           color=colors[:len(schemes)])
    plt.ylabel("Mean Normalized AUC")
    plt.title(title)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def statistical_comparison(all_aucs, baseline='magnitude'):
    """
    Perform paired t-tests comparing baseline to other methods
    
    Args:
        all_aucs: Dict with format {method_name: [auc1, auc2, ...]}
        baseline: Name of baseline method to compare against
    
    Returns:
        Dict with t-test results
    """
    results = {}
    baseline_aucs = all_aucs.get(baseline)

    if baseline_aucs is None:
        print(f"Warning: Baseline method '{baseline}' not found in results")
        return results

    print(f"\n{'='*60}")
    print(f"Statistical Comparison (Paired t-tests vs {baseline})")
    print(f"{'='*60}")

    for method, aucs in all_aucs.items():
        if method != baseline:
            t_stat, p_val = ttest_rel(baseline_aucs, aucs)
            results[method] = {'t_stat': t_stat, 'p_val': p_val}
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"{baseline} vs {method:30s}: t={t_stat:7.3f}, p={p_val:.4g} {significance}")

    print(f"{'='*60}")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print(f"{'='*60}\n")

    return results


def print_auc_summary(all_aucs):
    """Print mean ± std AUC for each pruning scheme"""
    print(f"\n{'='*60}")
    print("AUC Summary (Mean ± Std)")
    print(f"{'='*60}")

    for scheme, aucs in all_aucs.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"{scheme:30s}: {mean_auc:.4f} ± {std_auc:.4f}")

    print(f"{'='*60}\n")


def save_results_to_file(results_dict, filepath):
    """Save results dictionary to a text file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        for key, value in results_dict.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")

    print(f"Results saved to {filepath}")
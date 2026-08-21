"""
evaluate_metrics() must return EXACTLY what the single-metric helpers return.

run_pruning.py evaluates every sweep point through evaluate_metrics, which makes
one forward pass over the test set and reduces those predictions four ways. The
previous implementation made one pass per metric. That is only a valid
optimisation if the values are bit-identical -- the pruning curves, the AUC and
the paper's tables are all built from them -- so this asserts exact equality (==,
not allclose) rather than closeness.

    python tests/test_single_pass_metrics.py
"""
import sys
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simple_nn import SimpleNN
from src.models.simple_vit import SimpleViT
from src.utils.evaluation import (
    METRIC_FNS, collect_predictions, evaluate_metrics,
    evaluate_accuracy, evaluate_precision, evaluate_f1, evaluate_mcc,
)

SINGLE = {
    "accuracy": evaluate_accuracy,
    "precision": evaluate_precision,
    "f1": evaluate_f1,
    "mcc": evaluate_mcc,
}


def make_loader(shape, n=160, classes=10, batch_size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def prune_fraction(model, frac, seed=0):
    """Zero a deterministic fraction of the weights, to move the metrics around."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand(p.shape, generator=g) >= frac
            p.mul_(mask)
    return model


def collapse_to_one_class(model):
    """Drive every prediction to a single class: the degenerate case where
    precision/f1/mcc hit their zero_division / undefined branches."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def check(label, model, loader):
    batched = evaluate_metrics(model, loader, list(SINGLE))
    ok = True
    for name, fn in SINGLE.items():
        one = fn(model, loader)
        same = batched[name] == one          # exact, not approximate
        ok &= same
        flag = "" if same else "   <-- MISMATCH"
        print(f"  {label:26s} {name:10s} batched={batched[name]!r:24s} single={one!r:24s}"
              f"{flag}")
    assert ok, f"{label}: batched and single-pass metrics differ"

    # order of `names` must not change any value, and must be respected
    rev = evaluate_metrics(model, loader, list(SINGLE)[::-1])
    assert list(rev) == list(SINGLE)[::-1], "evaluate_metrics ignored the requested order"
    assert all(rev[n] == batched[n] for n in SINGLE), "value depends on requested order"

    # a subset must match the same metrics from the full call
    sub = evaluate_metrics(model, loader, ["accuracy", "mcc"])
    assert list(sub) == ["accuracy", "mcc"]
    assert all(sub[n] == batched[n] for n in sub), "subset differs from full evaluation"


def main():
    torch.manual_seed(0)
    cases = []

    nn_loader = make_loader((1, 28, 28))
    vit_loader = make_loader((3, 32, 32), batch_size=16)

    for frac, tag in ((0.0, "dense"), (0.5, "50%-pruned"), (0.9, "90%-pruned")):
        m = SimpleNN(input_size=784, hidden_size=16, num_classes=10, hidden_layers=2)
        cases.append((f"SimpleNN {tag}", prune_fraction(m, frac, seed=1), nn_loader))

    vit_kw = dict(img_size=32, in_channels=3, patch_size=8, embed_dim=32, depth=2,
                  num_heads=2, mlp_ratio=2.0, num_classes=10, dropout=0.0)
    cases.append(("SimpleViT dense", SimpleViT(**vit_kw), vit_loader))
    cases.append(("SimpleViT 70%-pruned",
                  prune_fraction(SimpleViT(**vit_kw), 0.7, seed=2), vit_loader))

    dead = SimpleNN(input_size=784, hidden_size=16, num_classes=10, hidden_layers=2)
    cases.append(("SimpleNN collapsed", collapse_to_one_class(dead), nn_loader))

    print("batched vs per-metric evaluation (must be EXACTLY equal)\n")
    for label, model, loader in cases:
        check(label, model, loader)

    # repeated passes over the same model/loader are themselves deterministic --
    # the premise the whole optimisation rests on
    model, loader = cases[1][1], cases[1][2]
    a = collect_predictions(model, loader)
    b = collect_predictions(model, loader)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all(), "collect_predictions is not deterministic"
    print("\n  repeated collect_predictions identical: PASS")

    for bad in (["nonsense"], ["accuracy", "loss"]):
        try:
            evaluate_metrics(model, loader, bad)
        except ValueError as exc:
            print(f"  rejects {bad}: {str(exc)[:60]}")
        else:
            raise AssertionError(f"evaluate_metrics accepted {bad}")

    assert list(METRIC_FNS) == ["accuracy", "precision", "f1", "mcc"], \
        "METRIC_FNS order defines the results-JSON key order; do not reorder casually"

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

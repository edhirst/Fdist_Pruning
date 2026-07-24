"""
Prunable-parameter policy shared by all pruners.

Transformers contain small-but-fragile parameter groups (LayerNorm scales/shifts,
positional embeddings) that unstructured pruning literature leaves untouched
(Wanda / SparseGPT / SViTE prune linear weights only). `get_prunable_mask`
returns a flat boolean mask (True = eligible for pruning) aligned with the
model's parameter flatten order, so every pruner can intersect its active mask
with it.

With `exclude=[]` (the NN default) everything is prunable and behavior is
bit-identical to the original pruners.
"""
import torch
import torch.nn as nn

NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)

VALID_EXCLUDE = {"norm", "pos_embed", "bias"}


def resolve_prunable_exclude(pruning_cfg, model_type):
    """
    Read pruning.prunable_exclude from config; null means arch default:
    [] for NN/CNN, ["norm", "pos_embed"] for Transformer.
    """
    ex = (pruning_cfg or {}).get("prunable_exclude", None)
    if ex is None:
        return ["norm", "pos_embed"] if model_type == "Transformer" else []
    ex = [str(e).strip().lower() for e in ex]
    unknown = set(ex) - VALID_EXCLUDE
    if unknown:
        raise ValueError(f"Unknown prunable_exclude entries: {sorted(unknown)}. Valid: {sorted(VALID_EXCLUDE)}")
    return ex


def _excluded_param_names(model, exclude):
    excluded = set()
    if "norm" in exclude:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, NORM_TYPES):
                for p_name, _ in mod.named_parameters(recurse=False):
                    excluded.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    for name, _ in model.named_parameters():
        if "pos_embed" in exclude and name.split(".")[-1] == "pos_embed":
            excluded.add(name)
        if "bias" in exclude and name.endswith(".bias"):
            excluded.add(name)
    return excluded


def get_prunable_mask(model, exclude=None, requires_grad_only=False):
    """
    Flat bool tensor (CPU), True = eligible for pruning.

    Aligned with the flatten order of model.named_parameters(); pass
    requires_grad_only=True to align with pruners that flatten only
    trainable parameters.
    """
    exclude = [str(e).strip().lower() for e in (exclude or [])]
    excluded_names = _excluded_param_names(model, exclude) if exclude else set()

    chunks = []
    for name, p in model.named_parameters():
        if requires_grad_only and not p.requires_grad:
            continue
        chunks.append(torch.full((p.numel(),), name not in excluded_names, dtype=torch.bool))

    if not chunks:
        return torch.zeros(0, dtype=torch.bool)
    return torch.cat(chunks, dim=0)

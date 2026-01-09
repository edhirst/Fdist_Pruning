import torch
import torch.nn as nn

try:
    from .base_pruner import BasePruner
except Exception:
    # Fallback BasePruner for environments where the relative import is unavailable
    class BasePruner:
        """Minimal fallback BasePruner used for linting/tests when the real BasePruner can't be imported."""
        def __init__(self):
            pass
import torch 

class MagnitudePruner(BasePruner):
    """
    Prunes weights based on their absolute magnitude.
    Removes weights with the lowest absolute values.
    """
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = float(threshold)

    def set_parameters(self, params):
        """
        Unified interface: params is a dict that must contain 'pruning_threshold'.
        """
        if isinstance(params, dict):
            self.threshold = float(params.get("pruning_threshold", self.threshold))
        else:
            # backward compatible: allow passing float directly
            self.threshold = float(params)

        # clamp to [0, 1]
        self.threshold = max(0.0, min(1.0, self.threshold))


    @staticmethod
    def _count_nonzero_and_total(model):
        nonzero = 0
        total = 0
        for p in model.parameters():
            if p.requires_grad:
                t = p.data
                total += t.numel()
                nonzero += (t != 0).sum().item()
        return nonzero, total

    def apply_pruning(self, model, train_loader=None, device="cpu"):
        """
        Prunes weights globally by magnitude using prune ratio self.threshold.

        Args:
            model: PyTorch model to prune

        Returns:
            model: pruned model
        """
        model.to(device)

        # Count before pruning (for robust pruned count)
        before_nz, total_params = self._count_nonzero_and_total(model)

        # Collect all trainable weights (flattened)
        weights_list = [p.data.view(-1) for p in model.parameters() if p.requires_grad]
        if not weights_list:
            return model

        all_weights = torch.cat(weights_list, dim=0)
        prune_ratio = float(self.threshold)

        # Edge cases
        prune_ratio = float(self.threshold)
        if prune_ratio <= 0.0:
            return model
        if prune_ratio >= 1.0:
            # prune everything
            offset = 0
            for p in model.parameters():
                if p.requires_grad:
                    p.data.zero_()
            return model

        # Magnitude scores
        magnitude_scores = torch.abs(all_weights)
        total = magnitude_scores.numel()
        k_prune = int(round(prune_ratio * total))

        # Build a global boolean mask: True=keep, False=prune
        if k_prune <= 0:
            mask_global = torch.ones_like(magnitude_scores, dtype=torch.bool)
        elif k_prune >= total:
            mask_global = torch.zeros_like(magnitude_scores, dtype=torch.bool)
        else:
            prune_idx = torch.topk(magnitude_scores, k=k_prune, largest=False).indices
            mask_global = torch.ones_like(magnitude_scores, dtype=torch.bool)
            mask_global[prune_idx] = False


        # Apply mask back to each parameter tensor
        offset = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param_mask = mask_global[offset:offset + numel].view_as(param)
                param.data.mul_(param_mask)
                offset += numel

        # Count pruned parameters robustly (what actually changed)
        after_nz, _ = self._count_nonzero_and_total(model)
        pruned_params = before_nz - after_nz
        pruning_rate = 100.0 * pruned_params / max(total_params, 1)

        print(f"Magnitude Pruning: Pruned {pruned_params}/{total_params} parameters ({pruning_rate:.2f}%)")

        return model

    def __str__(self):
        return f"MagnitudePruner(prune_ratio={self.threshold})"

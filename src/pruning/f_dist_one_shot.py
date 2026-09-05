import torch

from .base_pruner import BasePruner

from ..utils.fim_calculator import calculate_fim_nngeometry, calculate_fim_backprop
from .prunable import get_prunable_mask

class FDistOneShotPruner(BasePruner):
    """
    Fisher-distance pruning, one-shot: score = sqrt(I_kk(theta*)) * |w|.

    The cheapest member of the f_dist family, and the leading term of the
    distance derivation: the metric is evaluated at the trained point and never
    re-anchored as the model is pruned. Note the SQUARE ROOT on the Fisher
    entry, which is what makes the product a length rather than an ad hoc
    combination of the two baselines.

    run_pruning.py drives this scheme by re-pruning a fresh copy of the dense
    model to each absolute ratio, so the metric is always the one at theta*.

    Selection runs only over ACTIVE weights (w != 0) intersected with the
    eligibility mask from prunable.py.

    Parameters (dict) via set_parameters():
        pruning_threshold: float in [0,1], the fraction of ALL prunable
            parameters to remove
        fim_calculate_method: "backprop" or "nngeometry"
        prunable_exclude: list of groups to protect, see prunable.py
    """
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        # Quantile threshold: e.g. 0.1 means prune the bottom 10% (keep top 90%)
        self.threshold = float(self.parameters.get("pruning_threshold", 0.1))
        # Backend selection: "nngeometry" or "backprop"
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()
        self.prunable_exclude = self.parameters.get("prunable_exclude", None)
        self.epsilon = 1e-6

    def set_parameters(self, parameters):
        self.parameters = parameters or {}
        self.threshold = float(parameters.get("pruning_threshold", 0.1))
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"pruning_threshold must be in [0,1], got {self.threshold}")
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()
        self.prunable_exclude = self.parameters.get("prunable_exclude", self.prunable_exclude)

    def _calculate_fim(self, model, train_loader, device="cpu"):
        """
        Dispatch to the selected FIM backend.

        Returns:
            1D CPU tensor (FIM diagonal).
        """
        if self.fim_calculate_method in ("nngeometry", "nngeo"):
            return calculate_fim_nngeometry(model, train_loader, device=device)
        elif self.fim_calculate_method in ("backprop", "analytic"):
            return calculate_fim_backprop(model, train_loader, device=device)
        else:
            raise ValueError(
                f"Unknown fim_calculate_method: {self.fim_calculate_method}. "
                "Supported: 'nngeometry' or 'backprop'."
            )


    def apply_pruning(self, model, train_loader=None, device="cpu"):
        """
        Incremental one-shot pruning:
        - threshold means "prune ~threshold * TOTAL params in this call" (delta semantics)
        - selection is among active (non-zero) weights only
        """
        if train_loader is None:
            raise ValueError("One-shot FIM x Magnitude pruning requires train_loader")

        model.to(device)

        # Only trainable params
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            print("FIM x Magnitude (One-Shot): no trainable parameters found.")
            return model

        # Flatten weights
        flat_w = torch.cat([p.data.view(-1) for p in params], dim=0).to(device)
        total = int(flat_w.numel())

        # Count nonzero before (robust)
        before_nz = int((flat_w != 0).sum().item())

        # Determine how many to prune in this call (delta fraction of TOTAL)
        k_target = int(round(float(self.threshold) * total))
        if k_target <= 0:
            print(
                f"FIM x Magnitude (One-Shot): "
                f"Pruned 0/{total} parameters (0.00%), Remaining {before_nz}"
            )
            return model

        # Compute fim_diag aligned with the same flatten order
        fim_diag = self._calculate_fim(model, train_loader, device=device).to(device)
        if fim_diag.numel() != total:
            raise ValueError(
                f"FIM diag length mismatch: fim={fim_diag.numel()} vs weights={total}. "
                "Ensure fim_calculator flattens parameters in the same order as model.parameters() (requires_grad only)."
            )

        # Combined score
        combined_scores = torch.sqrt(torch.clamp(fim_diag, min=0.0)) * flat_w.abs()

        # Active-only selection to avoid wasting quota on already-zero weights;
        # restricted to prunable params
        prunable_mask = get_prunable_mask(
            model, exclude=self.prunable_exclude, requires_grad_only=True
        ).to(flat_w.device)
        active_mask = (flat_w != 0) & prunable_mask
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            print(
                f"FIM x Magnitude (One-Shot): "
                f"Pruned 0/{total} parameters (0.00%), Remaining 0"
            )
            return model

        k = min(k_target, active_count)

        active_scores = combined_scores[active_mask]
        prune_idx_in_active = torch.topk(active_scores, k=k, largest=False).indices
        active_global_idx = active_mask.nonzero(as_tuple=False).view(-1)
        prune_global_idx = active_global_idx[prune_idx_in_active]

        # Apply pruning: set selected weights to zero
        flat_w[prune_global_idx] = 0.0

        # Write back to parameters
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat_w[offset:offset + n].view_as(p).to(p.device))
            offset += n

        # Robust reporting: how many nonzero actually dropped
        after_flat = torch.cat([p.data.view(-1) for p in params], dim=0).to(device)
        after_nz = int((after_flat != 0).sum().item())

        pruned_params = before_nz - after_nz
        remaining = after_nz
        pruning_rate = 100.0 * pruned_params / max(total, 1)

        print(
            f"FIM x Magnitude (One-Shot): "
            f"Pruned {pruned_params}/{total} parameters ({pruning_rate:.2f}%), "
            f"Remaining {remaining}"
        )

        return model

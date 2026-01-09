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

from ..utils.fim_calculator import calculate_fim_nngeometry, calculate_fim_backprop

class MagnitudeFIMOneShotPruner(BasePruner):
    """
    One-shot pruning combining magnitude and FIM scores.
    Combined score = FIM_diagonal * |weight|
    """
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        # Quantile threshold: e.g. 0.1 means prune the bottom 10% (keep top 90%)
        self.threshold = float(self.parameters.get("pruning_threshold", 0.1))
        # Backend selection: "nngeometry" or "backprop"
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()

        self.epsilon = 1e-6

    def set_parameters(self, parameters):
        self.parameters = parameters or {}
        self.threshold = float(parameters.get("pruning_threshold", 0.1))
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"pruning_threshold must be in [0,1], got {self.threshold}")
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()

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
                f"Unknown fim_backend: {self.fim_calculate_method}. "
                "Supported: 'nngeometry' or 'backprop'."
            )

    def apply_pruning(self, model, train_loader=None, device='cpu'):
        """
        Apply one-shot FIM x Magnitude pruning.
        
        Args:
            model: PyTorch model to prune
            train_loader: DataLoader for FIM computation
            device: Device for computation
            
        Returns:
            model: Pruned model
        """
        if train_loader is None:
            raise ValueError("One-shot FIM x Magnitude pruning requires train_loader")

        # Flatten all trainable parameters in a stable order
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            print("FIM x Magnitude (One-Shot): no trainable parameters found.")
            return model

        all_weights = torch.cat([p.data.view(-1) for p in params], dim=0)
        fim_diag = self._calculate_fim(model, train_loader, device=device).to(all_weights.device)

        if fim_diag.numel() != all_weights.numel():
            raise ValueError(
                f"FIM diag length mismatch: fim={fim_diag.numel()} vs weights={all_weights.numel()}. "
                "Your FIM calculator must return a vector aligned with flattened model.parameters()."
            )

        # Combined score
        combined_scores = torch.sqrt(fim_diag) * all_weights.abs()

        total = combined_scores.numel()
        k_prune = int(round(self.threshold * total))

        # Build keep-mask: True=keep, False=prune (guarantee exact pruning count)
        if k_prune <= 0:
            mask_global = torch.ones_like(combined_scores, dtype=torch.bool)
        elif k_prune >= total:
            mask_global = torch.zeros_like(combined_scores, dtype=torch.bool)
        else:
            prune_idx = torch.topk(combined_scores, k=k_prune, largest=False).indices
            mask_global = torch.ones_like(combined_scores, dtype=torch.bool)
            mask_global[prune_idx] = False

        # Apply mask to each parameter tensor
        offset = 0
        for p in params:
            numel = p.numel()
            p_mask = mask_global[offset:offset + numel].view_as(p)
            p.data.mul_(p_mask.to(p.device))
            offset += numel

        # Count pruned parameters
        pruned_params = (~mask_global).sum().item()
        remaining = total - pruned_params
        pruning_rate = 100.0 * pruned_params / max(total, 1)

        print(
            f"FIM x Magnitude (One-Shot): "
            f"Pruned {pruned_params}/{total} parameters ({pruning_rate:.2f}%), "
            f"Remaining {remaining}"
        )
        return model

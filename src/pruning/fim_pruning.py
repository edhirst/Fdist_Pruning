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



class FIMPruner(BasePruner):
    """
    Prunes weights based on Fisher Information Matrix (FIM).
    Requires nngeometry library for FIM computation.
    """
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        # Quantile threshold: e.g. 0.1 means prune the bottom 10% (keep top 90%)
        self.threshold = float(self.parameters.get("pruning_threshold", 0.1))
        # Backend selection: "nngeometry" or "backprop"
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()

    def set_parameters(self, parameters):
        self.parameters = parameters or {}
        self.threshold = float(self.parameters.get("pruning_threshold", 0.1))
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
        Apply FIM-based pruning to the model.
        
        Args:
            model: PyTorch model to prune
            train_loader: DataLoader for FIM computation
            device: Device for computation
            
        Returns:
            model: Pruned model
        """
        if train_loader is None:
            raise ValueError("FIM pruning requires train_loader for FIM computation")

        # Calculate FIM diagonal
        fim_diag = self._calculate_fim(model, train_loader, device)
        
        total = fim_diag.numel()
        k_prune = int(round(self.threshold * total))

        if k_prune <= 0:
            mask_global = torch.ones_like(fim_diag, dtype=torch.bool)
        elif k_prune >= total:
            mask_global = torch.zeros_like(fim_diag, dtype=torch.bool)
        else:
            prune_idx = torch.topk(fim_diag, k=k_prune, largest=False).indices
            mask_global = torch.ones_like(fim_diag, dtype=torch.bool)
            mask_global[prune_idx] = False
        
        # Apply mask to each layer
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param_mask = mask_global[offset:offset + numel].view_as(param)
            param.data.mul_(param_mask.to(param.data.device))
            offset += numel
        
        # Count pruned parameters
        total_params = fim_diag.numel()
        pruned_params = (mask_global == 0).sum().item()
        pruning_rate = 100.0 * pruned_params / total_params
        
        print(f"FIM Pruning: Pruned {pruned_params}/{total_params} parameters ({pruning_rate:.2f}%)")
        
        return model

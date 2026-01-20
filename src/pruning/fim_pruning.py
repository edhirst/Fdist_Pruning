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
        
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return model

        # Flatten current weights (for active_mask) and compute total once from params
        flat_w = torch.cat([p.data.view(-1) for p in params], dim=0)
        total = int(flat_w.numel())        

        # How many to prune this call (delta fraction of TOTAL)
        k_target = int(round(self.threshold * total))
        if k_target <= 0:
            print(f"FIM Pruning: Pruned 0/{total} parameters (0.00%)")
            return model


        # Calculate FIM diagonal
        fim_diag = self._calculate_fim(model, train_loader, device=device).to(flat_w.device)
        
        
        if fim_diag.numel() != total:
            raise ValueError(
                f"FIM diag length mismatch: fim={fim_diag.numel()} vs params={total}. "
                "Ensure fim_calculator flattens parameters in the same order as model.parameters() (requires_grad only)."
            )

        # Active-only selection (exclude already-zero weights)
        active_mask = (flat_w != 0)
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            print(f"FIM Pruning: Pruned 0/{total} parameters (0.00%)")
            return model

        k = min(k_target, active_count)

        # Select k smallest FIM among active weights
        active_scores = fim_diag[active_mask]
        prune_idx_in_active = torch.topk(active_scores, k=k, largest=False).indices
        active_global_idx = active_mask.nonzero(as_tuple=False).view(-1)
        prune_global_idx = active_global_idx[prune_idx_in_active]

        # Build global boolean mask (True=keep, False=prune)
        mask_global = torch.ones(total, dtype=torch.bool, device=flat_w.device)
        mask_global[prune_global_idx] = False

        # Apply mask back to each parameter
        offset = 0
        for p in params:
            n = p.numel()
            p_mask = mask_global[offset:offset + n].view_as(p).to(dtype=p.data.dtype, device=p.data.device)
            p.data.mul_(p_mask)
            offset += n

        # Robust reporting: how many nonzero actually dropped
        before_nz = int((flat_w != 0).sum().item())
        after_flat = torch.cat([p.data.view(-1) for p in params], dim=0)
        after_nz = int((after_flat != 0).sum().item())

        pruned_params = before_nz - after_nz
        pruning_rate = 100.0 * pruned_params / max(total, 1)

        print(f"FIM Pruning: Pruned {pruned_params}/{total} parameters ({pruning_rate:.2f}%)")

        return model
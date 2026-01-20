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


class FDistIterativePruner(BasePruner):
    """
    Iterative (cumulative) pruning with per-step recomputation of f_dist score.

    Score definition (default):
        f_dist = FIM_diag * |w|

    IMPORTANT:
    - 'pruning_step' is interpreted as a DELTA pruning fraction per step
      relative to the TOTAL number of prunable parameters in the model.
      Example: pruning_step=0.05 => each apply_pruning() prunes ~5% of total params.
    - Selection is performed only among active (non-zero) weights: w != 0
      to avoid ties dominated by already-pruned weights.

    Parameters (dict) via set_parameters():
        pruning_step: float in [0,1]  (delta prune fraction per apply_pruning call)
        fim_calculate_method: "nngeometry" or "backprop"
        verbose: bool (optional, default True)
    """

    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        # delta prune fraction per call
        self.step = float(self.parameters.get("pruning_step", 0.0))
        # Backend selection: "nngeometry" or "backprop"
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()
        # cached total prunable params (trainable params) for delta->count conversion
        self._total_params = None

    def set_parameters(self, parameters):
        self.parameters = parameters or {}

        # delta per step
        self.step = float(self.parameters.get("pruning_step", 0.0))
        # clamp to [0,1]
        self.step = max(0.0, min(1.0, self.step))

        self.fim_calculate_method = str(
            self.parameters.get("fim_calculate_method", self.parameters.get("fim_backend", "nngeometry"))
        ).lower()

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

    @staticmethod
    def _get_prunable_params(model):
        # Here "prunable" == requires_grad parameters
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _flatten_params(params):
        return torch.cat([p.data.view(-1) for p in params], dim=0) if params else None

    @staticmethod
    def _write_back(params, flat_vec):
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat_vec[offset:offset + n].view_as(p))
            offset += n

    @staticmethod
    def _count_nonzero_and_total(params):
        nonzero = 0
        total = 0
        for p in params:
            t = p.data
            total += t.numel()
            nonzero += (t != 0).sum().item()
        return nonzero, total
            

    def apply_pruning(self, model, train_loader=None, device="cpu"):
        if train_loader is None:
            raise ValueError("FDistIterativePruner requires train_loader for FIM computation.")

        model.to(device)

        params = self._get_prunable_params(model)
        if not params:
            return model
        
        # Count before pruning (robust change-based reporting)
        before_nz, total_params_runtime = self._count_nonzero_and_total(params)

        flat_w = self._flatten_params(params)
        if flat_w is None:
            return model

        # cache total params once (defines what "5% of the model" means)
        if self._total_params is None:
            self._total_params = int(flat_w.numel())

        k_prune_target = int(round(float(self.step) * self._total_params))

        # recompute fim on current model state
        fim_diag = self._calculate_fim(model, train_loader, device=device).to(device)

        # refresh weights after potential earlier steps
        params = self._get_prunable_params(model)
        flat_w = self._flatten_params(params).to(device)

        if fim_diag.numel() != flat_w.numel():
            raise ValueError(
                f"FIM diag length mismatch: fim={fim_diag.numel()} vs weights={flat_w.numel()}. "
                "Ensure your FIM calculator returns a vector aligned with model.parameters() flatten order."
            )

        # active-only mask (exclude already-zero weights)
        active_mask = (flat_w != 0)
        active_count = int(active_mask.sum().item())

        # we can prune at most active_count
        k_prune = min(k_prune_target, active_count)

        # score definition: f_dist = fim * |w|
        scores = torch.sqrt(fim_diag) * flat_w.abs()
        active_scores = scores[active_mask]

        # select k smallest active scores to prune
        prune_idx_in_active = torch.topk(active_scores, k=k_prune, largest=False).indices
        active_global_idx = active_mask.nonzero(as_tuple=False).view(-1)
        prune_global_idx = active_global_idx[prune_idx_in_active]

        # apply pruning
        flat_w[prune_global_idx] = 0.0
        self._write_back(params, flat_w)

        # Count after pruning (robust reporting)
        after_nz, _ = self._count_nonzero_and_total(params)
        pruned_params = before_nz - after_nz
        remaining = after_nz
        pruning_rate = 100.0 * pruned_params / max(total_params_runtime, 1)

        print(
            f"FDist Iterative: "
            f"Pruned {pruned_params}/{total_params_runtime} parameters ({pruning_rate:.2f}%), "
            f"Remaining {remaining}"
        )

        return model

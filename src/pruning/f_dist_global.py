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
from .prunable import get_prunable_mask


class FDistGlobalPruner(BasePruner):
    """
    Global-path Fisher-distance pruning (batched alpha-scan).

    Instead of perturbing one coordinate at a time (FDistPruner, cost
    ~ #active x (K-1) full Fisher evaluations per step), the Fisher diagonal is
    evaluated at K interpolated models alpha * theta with ALL weights scaled
    together — K Fisher evaluations per step in total, independent of model size.

    Score (per-coordinate contribution to the Fisher–Rao length of the straight
    path theta -> 0 under the diagonal approximation):

        s_i = |w_i| * mean_alpha sqrt( F_ii(alpha * theta) )

    Note this is mean-of-sqrt (the correct path-length average).

    alphas = linspace(1.0, 1/K, K): alpha = 0 is deliberately EXCLUDED — at the
    all-zero model every activation and hence every gradient vanishes, so
    F(0) == 0 identically and including it only rescales the one-shot score.

    Zeros stay zero along the path (alpha * 0 = 0), so the current pruning mask
    is respected at every alpha.

    Pruning semantics are identical to FDistIterativePruner: 'pruning_step' is a
    DELTA fraction of TOTAL prunable params per apply_pruning() call, selection
    among active (non-zero) prunable weights only.

    Parameters (dict) via set_parameters():
        pruning_step: float in [0,1]      (delta prune fraction per call)
        fim_calculate_method: "nngeometry" or "backprop"
        f_dist_avg_points: int >= 2       (K, number of alpha points)
        prunable_exclude: optional list   (see prunable.py)
    """

    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.step = max(0.0, min(1.0, float(self.parameters.get("pruning_step", 0.0))))
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "backprop")).lower()
        self.f_dist_avg_points = int(self.parameters.get("f_dist_avg_points", 3))
        if self.f_dist_avg_points < 2:
            raise ValueError(f"f_dist_avg_points must be >= 2, got {self.f_dist_avg_points}")
        self.prunable_exclude = self.parameters.get("prunable_exclude", None)
        # cached total prunable params for delta->count conversion
        self._total_params = None

    def set_parameters(self, parameters):
        self.parameters = parameters or {}
        self.step = max(0.0, min(1.0, float(self.parameters.get("pruning_step", self.step))))
        self.fim_calculate_method = str(
            self.parameters.get("fim_calculate_method", self.fim_calculate_method)
        ).lower()
        self.f_dist_avg_points = int(self.parameters.get("f_dist_avg_points", self.f_dist_avg_points))
        if self.f_dist_avg_points < 2:
            raise ValueError(f"f_dist_avg_points must be >= 2, got {self.f_dist_avg_points}")
        self.prunable_exclude = self.parameters.get("prunable_exclude", self.prunable_exclude)

    def _calculate_fim(self, model, train_loader, device="cpu"):
        if self.fim_calculate_method in ("nngeometry", "nngeo"):
            return calculate_fim_nngeometry(model, train_loader, device=device)
        elif self.fim_calculate_method in ("backprop", "analytic"):
            return calculate_fim_backprop(model, train_loader, device=device)
        else:
            raise ValueError(
                f"Unknown fim_calculate_method: {self.fim_calculate_method}. "
                "Supported: 'nngeometry' or 'backprop'."
            )

    def _sqrt_fim_path_avg(self, model, train_loader, device):
        """
        mean over alphas of sqrt(F_diag(alpha * theta)), one Fisher eval per alpha.

        Returns:
            1D CPU tensor aligned with model.named_parameters() flatten order.
        """
        params = list(model.parameters())
        theta0 = [p.data.detach().clone() for p in params]

        K = self.f_dist_avg_points
        alphas = torch.linspace(1.0, 1.0 / K, steps=K).tolist()

        acc = None
        try:
            for a in alphas:
                if abs(a - 1.0) > 1e-12:
                    with torch.no_grad():
                        for p, w0 in zip(params, theta0):
                            p.data.copy_(w0 * a)
                F_a = self._calculate_fim(model, train_loader, device=device).detach().cpu()
                sqrt_F = torch.sqrt(torch.clamp(F_a, min=0.0))
                acc = sqrt_F if acc is None else acc + sqrt_F
        finally:
            with torch.no_grad():
                for p, w0 in zip(params, theta0):
                    p.data.copy_(w0)

        return acc / float(K)

    @staticmethod
    def _get_prunable_params(model):
        return [p for p in model.parameters() if p.requires_grad]

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
            raise ValueError("FDistGlobalPruner requires train_loader for FIM computation.")

        model.to(device)

        params = self._get_prunable_params(model)
        if not params:
            return model

        before_nz, total_params_runtime = self._count_nonzero_and_total(params)

        # cache total params once (defines what "x% of the model" means)
        if self._total_params is None:
            self._total_params = int(total_params_runtime)

        k_prune_target = int(round(float(self.step) * self._total_params))
        if k_prune_target <= 0:
            print(
                f"FDist Global: Pruned 0/{total_params_runtime} parameters (0.00%), "
                f"Remaining {before_nz}"
            )
            return model

        # K Fisher evals along the global shrinkage path, on the CURRENT model state
        sqrt_F_avg = self._sqrt_fim_path_avg(model, train_loader, device)

        flat_w = torch.cat([p.data.view(-1) for p in params], dim=0).cpu()
        if sqrt_F_avg.numel() != flat_w.numel():
            raise ValueError(
                f"FIM diag length mismatch: fim={sqrt_F_avg.numel()} vs weights={flat_w.numel()}. "
                "Ensure your FIM calculator returns a vector aligned with model.parameters() flatten order."
            )

        # score: |w| * mean_alpha sqrt(F_ii(alpha * theta))
        scores = sqrt_F_avg * flat_w.abs()

        # active-only selection among prunable params
        prunable_mask = get_prunable_mask(model, exclude=self.prunable_exclude, requires_grad_only=True)
        active_mask = (flat_w != 0) & prunable_mask
        active_count = int(active_mask.sum().item())

        k_prune = min(k_prune_target, active_count)
        if k_prune <= 0:
            print(
                f"FDist Global: Pruned 0/{total_params_runtime} parameters (0.00%), "
                f"Remaining {before_nz}"
            )
            return model

        active_scores = scores[active_mask]
        prune_idx_in_active = torch.topk(active_scores, k=k_prune, largest=False).indices
        active_global_idx = active_mask.nonzero(as_tuple=False).view(-1)
        prune_global_idx = active_global_idx[prune_idx_in_active]

        # apply pruning
        flat_w[prune_global_idx] = 0.0
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.data.copy_(flat_w[offset:offset + n].view_as(p).to(p.device))
                offset += n

        after_nz, _ = self._count_nonzero_and_total(params)
        pruned_params = before_nz - after_nz
        pruning_rate = 100.0 * pruned_params / max(total_params_runtime, 1)

        print(
            f"FDist Global: "
            f"Pruned {pruned_params}/{total_params_runtime} parameters ({pruning_rate:.2f}%), "
            f"Remaining {after_nz}"
        )

        return model

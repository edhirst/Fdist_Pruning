import torch
import torch.nn as nn
import numpy as np
import copy

try:
    from .base_pruner import BasePruner
except Exception:
    # Fallback BasePruner for environments where the relative import is unavailable
    class BasePruner:
        """Minimal fallback BasePruner used for linting/tests when the real BasePruner can't be imported."""
        def __init__(self):
            pass

from ..utils.fim_calculator import calculate_fim_nngeometry, calculate_fim_backprop


class FDistPruner(BasePruner):
    """
    - Each apply_pruning() recomputes F_sqrt_avg on CURRENT model state.
    - Pruning semantics are cumulative: prune model to target_pruning_pct in [0,1],
      but only prune the delta needed beyond current zeros.
    - Selection is performed only among ACTIVE weights (w != 0).

    Parameters via set_parameters():
        freeze_all_zero_tensors: bool, default True
        f_dist_avg_points: int, default 2
            - 2: {w, 0}
            - 5: {w, 0.75w, 0.5w, 0.25w, 0}
    """

    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.freeze_all_zero_tensors = bool(self.parameters.get("freeze_all_zero_tensors", True))
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()
        self.f_dist_avg_points = int(self.parameters.get("f_dist_avg_points", 2))
        if self.f_dist_avg_points < 2:
            raise ValueError(f"f_dist average points must be >= 2, got {self.f_dist_avg_points}")

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
    def _flatten_all_params(model):
        return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    @staticmethod
    def _flatten_requires_grad_mask(model):
        chunks = []
        for p in model.parameters():
            chunks.append(torch.full((p.numel(),), bool(p.requires_grad), dtype=torch.bool))
        return torch.cat(chunks, dim=0)

    def calculate_fim_avg(self, model, train_loader, device):
        model_cp = copy.deepcopy(model).to(device)

        if self.freeze_all_zero_tensors:
            for p in model_cp.parameters():
                if torch.all(p == 0):
                    p.requires_grad_(False)

        total_vec = self._flatten_all_params(model_cp).to(device)
        
        # idx -> (param_tensor, local_index) mapping in flatten order
        idx2tensor = {}
        offset = 0
        for _, p in model_cp.named_parameters():
            n = p.numel()
            for k in range(n):
                idx2tensor[offset + k] = (p, k)
            offset += n

        # baseline fisher at alpha=1.0
        F_base = self._calculate_fim(model_cp, train_loader, device).detach().cpu()
        if F_base.numel() != total_vec.numel():
            raise ValueError(
                f"fisher_diag_sqrt length mismatch: fisher={F_base.numel()} vs params={total_vec.numel()}. "
                "Your fisher_diag_sqrt must flatten exactly model.parameters() order."
            )

        F_avg = F_base.clone()

        nonzero_idx = (total_vec != 0).nonzero(as_tuple=False).view(-1).tolist()
        K = self.f_dist_avg_points
        alphas = torch.linspace(1.0, 0.0, steps=K, device=device)

        for idx in nonzero_idx:
            param, local_i = idx2tensor[idx]

            with torch.no_grad():
                w0 = param.view(-1)[local_i].item()

            acc = 0.0
            for a in alphas.tolist():
                if abs(a - 1.0) < 1e-12:
                    acc += float(F_base[idx].item())
                    continue

                with torch.no_grad():
                    param.view(-1)[local_i] = float(a) * w0

                F_tmp = torch.sqrt(self._calculate_fim(model_cp, train_loader, device).detach().cpu())
                acc += float(F_tmp[idx].item())

            with torch.no_grad():
                param.view(-1)[local_i] = w0

            F_avg[idx] = acc / float(K)

        return F_avg  # CPU 1D

    def apply_pruning(self, model, train_loader=None, device="cpu", target_pruning_pct=None):
        if train_loader is None:
            raise ValueError("FDistPruner requires train_loader")
        if target_pruning_pct is None:
            raise ValueError("FDistPruner requires target_pruning_pct (cumulative pruning ratio in [0,1]).")

        r = float(target_pruning_pct)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"target_pruning_pct must be in [0,1], got {r}")

        model.to(device)


        F_sqrt_avg_all = self.calculate_fim_avg(model, train_loader=train_loader, device=device)  

        # Flatten abs/weights over ALL params, build requires_grad mask
        all_abs, all_w = [], []
        for p in model.parameters():
            all_abs.append(p.data.abs().reshape(-1).detach().cpu())
            all_w.append(p.data.reshape(-1).detach().cpu())

        all_abs_vec = torch.cat(all_abs, dim=0)  # CPU
        all_w_vec = torch.cat(all_w, dim=0)      # CPU
        grad_mask_all = self._flatten_requires_grad_mask(model)  # CPU bool

        # Restrict to prunable (requires_grad) entries
        F = F_sqrt_avg_all[grad_mask_all]
        Wabs = all_abs_vec[grad_mask_all]
        w_flat = all_w_vec[grad_mask_all].clone()

        total = int(F.numel())
        if total == 0:
            print("f_dist: Pruned 0/0 parameters (0.00%), Remaining 0")
            return model

        current_zeros = int((w_flat == 0).sum().item())
        target_zeros = int(round(r * total))
        target_zeros = max(0, min(target_zeros, total))

        need_to_prune = target_zeros - current_zeros
        if need_to_prune <= 0:
            remaining = total - current_zeros
            print(f"f_dist: Pruned 0/{total} parameters (0.00%), Remaining {remaining}")
            return model

        # F_dist
        importance = F * Wabs

        # prune only among active weights
        active_mask = (w_flat != 0)
        active_count = int(active_mask.sum().item())
        k = min(int(need_to_prune), active_count)

        if k <= 0:
            remaining = total - current_zeros
            print(f"f_dist: Pruned 0/{total} parameters (0.00%), Remaining {remaining}")
            return model

        active_scores = importance[active_mask]
        prune_idx_in_active = torch.topk(active_scores, k=k, largest=False).indices
        active_global_idx = active_mask.nonzero(as_tuple=False).view(-1)
        prune_global_idx = active_global_idx[prune_idx_in_active]

        before_nz = int((w_flat != 0).sum().item())
        w_flat[prune_global_idx] = 0.0
        after_nz = int((w_flat != 0).sum().item())

        pruned_params = before_nz - after_nz
        remaining = after_nz
        pruning_rate = 100.0 * pruned_params / max(total, 1)

        # write back: scatter into full vector, then copy into parameters
        full_vec = all_w_vec.clone()
        full_vec[grad_mask_all] = w_flat

        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(full_vec[offset:offset + n].view_as(p).to(p.device))
                offset += n

        print(
            f"f_dist: Pruned {pruned_params}/{total} parameters ({pruning_rate:.2f}%), "
            f"Remaining {remaining}"
        )
        return model

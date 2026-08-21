import torch
import copy

from .base_pruner import BasePruner

from ..utils.fim_calculator import calculate_fim_nngeometry, calculate_fim_backprop
from ..utils.fisher_parallel import ProbePool, materialise_batches, resolve_workers
from .prunable import get_prunable_mask


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
        # delta per step
        self.step = float(self.parameters.get("pruning_step", 0.0))
        # clamp to [0,1]
        self.step = max(0.0, min(1.0, self.step))
        self.freeze_all_zero_tensors = bool(self.parameters.get("freeze_all_zero_tensors", True))
        self.fim_calculate_method = str(self.parameters.get("fim_calculate_method", "nngeometry")).lower()
        self.f_dist_avg_points = int(self.parameters.get("f_dist_avg_points", 2))
        if self.f_dist_avg_points < 2:
            raise ValueError(f"f_dist average points must be >= 2, got {self.f_dist_avg_points}")
        self.prunable_exclude = self.parameters.get("prunable_exclude", None)
        # forward-mode (JVP) fast path: one JVP per (coordinate, sample) instead of
        # a full reverse-mode Fisher per coordinate. Mathematically identical --
        # see tests/test_fisher_forward_equivalence.py (float64 rel err ~1e-16).
        self.fast = bool(self.parameters.get("f_dist_fast", True))
        self.probe_batch = int(self.parameters.get("f_dist_probe_batch", 32))
        # Probe worker processes. Coordinate probes are independent and the work
        # is compute-bound per core, so this is the knob that actually uses a big
        # CPU node; FDIST_WORKERS / PBS_NP override the config value.
        self.workers = resolve_workers(self.parameters.get("f_dist_workers", 1))
        self.sample_chunk = self.parameters.get("f_dist_sample_chunk", None)

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

        # preserve previously-set values when keys are absent (set_parameters used
        # to silently reset these to defaults)
        self.freeze_all_zero_tensors = bool(
            self.parameters.get("freeze_all_zero_tensors", self.freeze_all_zero_tensors)
        )
        self.f_dist_avg_points = int(self.parameters.get("f_dist_avg_points", self.f_dist_avg_points))
        if self.f_dist_avg_points < 2:
            raise ValueError(f"f_dist average points must be >= 2, got {self.f_dist_avg_points}")
        self.prunable_exclude = self.parameters.get("prunable_exclude", self.prunable_exclude)
        self.fast = bool(self.parameters.get("f_dist_fast", self.fast))
        self.probe_batch = int(self.parameters.get("f_dist_probe_batch", self.probe_batch))
        self.workers = resolve_workers(self.parameters.get("f_dist_workers", self.workers))
        self.sample_chunk = self.parameters.get("f_dist_sample_chunk", self.sample_chunk)

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

    @staticmethod
    def _count_nonzero_and_total_prunable(model):
        nonzero = 0
        total = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            t = p.data
            total += t.numel()
            nonzero += (t != 0).sum().item()
        return int(nonzero), int(total)

    def calculate_fim_avg(self, model, train_loader, device):
        """Average Fisher over the per-coordinate shrink path; dispatches to the
        forward-mode fast path unless f_dist_fast=False."""
        if self.fast:
            return self._calculate_fim_avg_forward(model, train_loader, device)
        return self._calculate_fim_avg_legacy(model, train_loader, device)

    def _calculate_fim_avg_forward(self, model, train_loader, device):
        """
        Same quantity as _calculate_fim_avg_legacy, via forward-mode AD.

            F_avg[i] = (1/K) * sum_alpha F_ii( theta with w_i -> alpha*w_i )

        The legacy path spends one FULL reverse-mode Fisher (C backward passes
        per sample over all P parameters) per (coordinate, alpha) and keeps a
        single entry. Here one JVP per (coordinate, sample) yields that entry
        directly, so the whole tensor's coordinates share one pass over the data.
        Measured 8.9x (SimpleViT/CIFAR-10) to 325x (SimpleNN/CIFAR-10) per probe.
        """
        model_cp = copy.deepcopy(model).to(device)

        if self.freeze_all_zero_tensors:
            for p in model_cp.parameters():
                if torch.all(p == 0):
                    p.requires_grad_(False)

        total_vec = self._flatten_all_params(model_cp).to(device)

        # baseline Fisher at alpha=1.0 -- one reverse pass gives every entry
        F_base = self._calculate_fim(model_cp, train_loader, device).detach().cpu()
        if F_base.numel() != total_vec.numel():
            raise ValueError(
                f"fisher_diag_sqrt length mismatch: fisher={F_base.numel()} vs params={total_vec.numel()}. "
                "Your fisher_diag_sqrt must flatten exactly model.parameters() order."
            )

        F_avg = F_base.clone()
        K = self.f_dist_avg_points
        alphas = [a for a in torch.linspace(1.0, 0.0, steps=K).tolist() if abs(a - 1.0) >= 1e-12]

        # Pull the Fisher subset into memory once. The serial path would otherwise
        # re-iterate the DataLoader for every (tensor, alpha) -- ~100 passes per
        # step on the ViT -- and forked workers must not share a live DataLoader.
        batches = materialise_batches(train_loader)

        # Sharding is CPU-only: forking a CUDA context is unsafe.
        workers = self.workers if str(device) == "cpu" else 1
        if self.workers > 1 and workers == 1:
            print(f"FDist (exact): f_dist_workers={self.workers} ignored on device={device} "
                  "(probe sharding is CPU-only)")

        nonzero_all = (total_vec != 0).cpu()
        with ProbePool(model_cp, batches, workers=workers,
                       probe_batch=self.probe_batch, sample_chunk=self.sample_chunk) as pool:
            if workers > 1:
                n_probes = int(nonzero_all.sum()) * len(alphas)
                print(f"FDist (exact): {n_probes:,} probes over {workers} worker processes")
            offset = 0
            for name, p in model_cp.named_parameters():
                n = p.numel()
                local_nz = nonzero_all[offset:offset + n].nonzero(as_tuple=False).view(-1)
                if local_nz.numel():
                    acc = F_base[offset:offset + n][local_nz].clone()   # the alpha=1 term
                    for a in alphas:
                        acc += pool.run(name, local_nz,
                                        torch.full((local_nz.numel(),), float(a)))
                    F_avg[offset + local_nz] = acc / float(K)
                offset += n

        return F_avg  # CPU 1D

    def _calculate_fim_avg_legacy(self, model, train_loader, device):
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

                F_tmp = self._calculate_fim(model_cp, train_loader, device).detach()
                acc += float(F_tmp[idx].item())

            with torch.no_grad():
                param.view(-1)[local_i] = w0

            F_avg[idx] = acc / float(K)

        return F_avg  # CPU 1D

    def apply_pruning(self, model, train_loader=None, device="cpu",):
        if train_loader is None:
            raise ValueError("FDistPruner requires train_loader")

        before_nz, total_params_runtime = self._count_nonzero_and_total_prunable(model)
        if total_params_runtime == 0:
            print("FDist (exact): Pruned 0/0 parameters (0.00%), Remaining 0")
            return model

        model.to(device)

        # cache total prunable params once
        if self._total_params is None:
            self._total_params = int(total_params_runtime)

        # each call prunes a fixed delta w.r.t. total prunable params
        k_prune_target = int(round(float(self.step) * self._total_params))

        if k_prune_target <= 0:
            print(
                f"FDist (exact): Pruned 0/{total_params_runtime} parameters (0.00%), "
                f"Remaining {before_nz}"
            )
            return model

        # recompute F_sqrt_avg on current model state (clamp guards against tiny
        # negative Fisher values; no-op for the non-negative backprop Fisher)
        F_avg_all = self.calculate_fim_avg(model, train_loader=train_loader, device=device)
        F_sqrt_avg_all = torch.sqrt(torch.clamp(F_avg_all, min=0.0))

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

        if F.numel() != w_flat.numel():
            raise ValueError(
                f"F_sqrt_avg length mismatch: fim={F.numel()} vs weights={w_flat.numel()}. "
                "Ensure calculate_fim_avg aligns with model.parameters() flatten order."
            )

        # prune only among active weights, restricted to prunable params
        prunable_all = get_prunable_mask(model, exclude=self.prunable_exclude, requires_grad_only=False)
        active_mask = (w_flat != 0) & prunable_all[grad_mask_all]
        active_count = int(active_mask.sum().item())
        k_prune = min(k_prune_target, active_count)

        if k_prune <= 0:
            print(
                f"FDist (exact): Pruned 0/{total_params_runtime} parameters (0.00%), "
                f"Remaining {before_nz}"
            )
            return model


        # F_dist
        importance = F * Wabs

        active_scores = importance[active_mask]
        prune_idx_in_active = torch.topk(active_scores, k=k_prune, largest=False).indices
        active_global_idx = active_mask.nonzero(as_tuple=False).view(-1)
        prune_global_idx = active_global_idx[prune_idx_in_active]

                # apply pruning
        w_flat[prune_global_idx] = 0.0

        # write back: scatter into full vector, then copy into parameters
        full_vec = all_w_vec.clone()
        full_vec[grad_mask_all] = w_flat

        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(full_vec[offset:offset + n].view_as(p).to(p.device))
                offset += n

        # Count after pruning
        after_nz, _ = self._count_nonzero_and_total_prunable(model)
        pruned_params = before_nz - after_nz
        remaining = after_nz
        pruning_rate = 100.0 * pruned_params / max(total_params_runtime, 1)

        print(
            f"FDist (exact): "
            f"Pruned {pruned_params}/{total_params_runtime} parameters ({pruning_rate:.2f}%), "
            f"Remaining {remaining}"
        )
        return model
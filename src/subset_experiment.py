import os
import sys
import json
import yaml
import copy
import torch
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr

from nngeometry import FIM
from nngeometry.object import PMatDiag

from .utils.data_loader import build_dataloaders
from .utils.model_builder import build_model_from_config, get_model_type, resolve_checkpoint_path
from .utils.fim_calculator import calculate_fim_backprop


# ------------------------- Config / Device ------------------------- #

def load_config(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_subset_loader(train_loader: DataLoader, subset_size: int, seed: int):
    """
    """
    dataset = train_loader.dataset
    n = len(dataset)
    subset_size = min(int(subset_size), n)

    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:subset_size].tolist()

    subset = Subset(dataset, idx)
    return DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=getattr(train_loader, "pin_memory", False),
    )


# ------------------------- FIM diag ------------------------- #

@torch.no_grad()
def freeze_all_zero_tensors(model: torch.nn.Module):
    """
    """
    for p in model.parameters():
        if p.requires_grad and torch.all(p == 0):
            p.requires_grad_(False)


def compute_fim_diag_pmatdiag(model, loader, device, freeze_zero_tensors: bool, method: str = "pmatdiag"):
    """
    Fisher diagonal for the convergence diagnostic.
    method='pmatdiag' uses nngeometry (NN/CNN only); method='backprop' uses the
    torch.func class-marginal Fisher (required for transformer models).
    """
    model = model.to(device)
    model.eval()

    if freeze_zero_tensors:
        freeze_all_zero_tensors(model)

    if method == "backprop":
        return calculate_fim_backprop(model, loader, device=device)

    fim_obj = FIM(
        model=model,
        loader=loader,
        representation=PMatDiag,
        device=device
    )
    diag = fim_obj.get_diag().detach().cpu()
    return diag


# ------------------------- Main ------------------------- #

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("src", "config.yaml")
    config = load_config(config_path)

    device = get_device()
    dataset_name, train_loader, _ = build_dataloaders(config, augment=False)

    # model + checkpoint
    model, arch_name = build_model_from_config(config, dataset_name=dataset_name)
    model = model.to(device)

    ckpt_path = resolve_checkpoint_path(config, arch_name, dataset_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train it first: python -m src.train_model")

    # PyTorch warning 的做法：如果你存的是 state_dict，這樣讀就好
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    diag_cfg = (config.get("fim_diagnostics", {}) or {})
    sample_sizes = diag_cfg.get("sample_sizes", [100, 500, 1000, 5000, 10000])
    seed = int(diag_cfg.get("seed", 42))
    freeze_zero_tensors = bool(diag_cfg.get("freeze_all_zero_tensors", False))

    # transformer models require the backprop Fisher (nngeometry lacks LayerNorm support)
    default_method = "backprop" if get_model_type(config) == "Transformer" else "pmatdiag"
    fim_method = str(diag_cfg.get("method", default_method)).lower()

    # reference size：預設用 train set 全量
    full_n = int(diag_cfg.get("full_n", len(train_loader.dataset)))
    full_n = min(full_n, len(train_loader.dataset))

    # 先算 reference
    ref_loader = make_subset_loader(train_loader, subset_size=full_n, seed=seed)
    ref_model = copy.deepcopy(model)
    F_ref = compute_fim_diag_pmatdiag(ref_model, ref_loader, device=device, freeze_zero_tensors=freeze_zero_tensors, method=fim_method)

    fisher_dict = {}
    for N in sample_sizes:
        N = int(N)
        sub_loader = make_subset_loader(train_loader, subset_size=N, seed=seed)
        sub_model = copy.deepcopy(model)
        diag = compute_fim_diag_pmatdiag(sub_model, sub_loader, device=device, freeze_zero_tensors=freeze_zero_tensors, method=fim_method)
        fisher_dict[N] = diag

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {arch_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Reference N(full_n): {full_n}")
    print(f"Freeze all-zero tensors: {freeze_zero_tensors}")
    print(f"Fisher method: {fim_method}")
    print()
    print(f"{'N':>8} | {'MSE':>12} | {'MaxAbsErr':>12} | {'Spearman ρ':>10}")
    print("-" * 52)

    rows = []
    for N in sample_sizes:
        N = int(N)
        diag = fisher_dict[N]
        if diag.numel() != F_ref.numel():
            raise ValueError(f"FIM diag length mismatch: N={N} diag={diag.numel()} vs ref={F_ref.numel()}")

        mse = torch.mean((diag - F_ref) ** 2).item()
        max_abs = torch.max(torch.abs(diag - F_ref)).item()
        rho = spearmanr(F_ref.numpy(), diag.numpy()).correlation

        print(f"{N:8d} | {mse:12.4e} | {max_abs:12.4e} | {rho:10.4f}")

        rows.append({
            "N": N,
            "mse": mse,
            "max_abs_err": max_abs,
            "spearman_rho": float(rho) if rho is not None else None,
        })

    # optional JSON output (方便你之後畫圖/比較)
    out_path = diag_cfg.get("output_json", None)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "metadata": {
                "dataset": dataset_name,
                "model": arch_name,
                "checkpoint": ckpt_path,
                "seed": seed,
                "full_n": full_n,
                "freeze_all_zero_tensors": freeze_zero_tensors,
            },
            "results": rows
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print()
        print(f"Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()

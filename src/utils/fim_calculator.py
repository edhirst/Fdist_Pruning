import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import vmap, grad, functional_call
from nngeometry import FIM
from nngeometry.object import PMatKFAC


def calculate_fim_nngeometry(model, loader, device="cpu"):
    """
    Calculate Fisher Information Matrix diagonal using nngeometry package for the model.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        device: Device to run computation on

    Returns:
        fim_diag: Diagonal of FIM as a flat tensor
    """

    # nngeometry's KFAC representation only covers Linear/Conv2d layers; transformer
    # blocks (LayerNorm, attention) are unsupported and silently miscounted.
    if any(isinstance(m, nn.LayerNorm) for m in model.modules()):
        raise ValueError(
            "fim_calculate_method='nngeometry' (PMatKFAC) does not support transformer "
            "models (LayerNorm/attention layers). Use fim_calculate_method='backprop'."
        )

    model = model.to(device)
    model.eval()

    fim_obj = FIM(
        model = model,
        loader = loader,
        representation = PMatKFAC,
        variant = "classif_logits",
        device = device,
    )

    return fim_obj.get_diag().detach().cpu()



def calculate_fim_backprop(model, loader, device="cpu", chunk_size=32):
    """
    Calculate the class-marginal empirical Fisher diagonal in a SINGLE pass over
    the data, for ALL parameter tensors at once:

        F_ii = E_x[ sum_c p(c|x) * (d log p(c|x) / d theta_i)^2 ]

    Per batch chunk and per class c, one vmap'd per-sample gradient of
    log p(c|x) w.r.t. the full parameter dict accumulates p-weighted squared
    gradients for every tensor simultaneously. This replaces the previous
    per-tensor implementation (kept below as calculate_fim_backprop_per_tensor),
    which re-iterated the full dataloader once per parameter tensor — identical
    math and output, ~#tensors times faster (6x for SimpleNN, ~60x for SimpleViT).

    Args:
        model: PyTorch model returning logits [B, C]
        loader: DataLoader for the Fisher subset
        device: Device to run computation on
        chunk_size: per-sample-gradient chunk to bound memory (chunk x P floats)

    Returns:
        1D CPU tensor whose flatten order matches model.named_parameters().
    """
    model = model.to(device)
    model.eval()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    if not params:
        return torch.empty(0)

    def logp_one(pr, x_single, c_long):
        logits = functional_call(model, (pr, buffers), (x_single.unsqueeze(0),))
        logp = F.log_softmax(logits, dim=1)  # [1, C]
        return logp[0, c_long]

    grad_one = grad(logp_one)                       # d log p(c|x) / d params (dict)
    grad_batch = vmap(grad_one, in_dims=(None, 0, None))

    acc = {k: torch.zeros_like(v, device=device) for k, v in params.items()}
    seen = 0

    for xb, _ in loader:
        xb = xb.to(device)
        for x_chunk in xb.split(chunk_size):
            b = x_chunk.size(0)
            with torch.no_grad():
                logits = functional_call(model, (params, buffers), (x_chunk,))
                probs = torch.softmax(logits, dim=1)  # [b, C]
            C = probs.size(1)

            for c in range(C):
                c_long = torch.tensor(c, device=device, dtype=torch.long)
                gtree = grad_batch(params, x_chunk, c_long)      # dict of [b, ...param_shape]
                w = probs[:, c]
                for k, g in gtree.items():
                    acc[k] += (w.reshape(b, *([1] * (g.ndim - 1))) * g ** 2).sum(dim=0)

            seen += b

    parts = []
    for name, _ in model.named_parameters():
        parts.append((acc[name] / float(max(seen, 1))).reshape(-1).to("cpu"))

    return torch.cat(parts, dim=0)



def calculate_fim_backprop_per_tensor(model, loader, device="cpu"):
    """
    Legacy per-tensor implementation (one full dataloader pass per parameter
    tensor). Kept for equivalence testing against calculate_fim_backprop.

    Returns:
        1D CPU tensor whose flatten order matches model.named_parameters().
    """
    model = model.to(device)
    model.eval()

    parts = []
    for name, p in model.named_parameters():
        # Compute E[g^2] for this tensor (same shape as p)
        Eg2_tensor = _fisher_diag_for_one_tensor(model, loader, device, name)
        # Flatten and append
        parts.append(Eg2_tensor.reshape(-1).to("cpu"))

    if not parts:
        return torch.empty(0)

    return torch.cat(parts, dim=0)







# ================================================================
# Backprop Fisher diagonal helper function
# ================================================================

def _switch_requires_grad(model, only_prefixes):
    """
    Temporarily set requires_grad=True only for parameters whose names
    start with one of the given prefixes.
    """
    old = {}
    for n, p in model.named_parameters():
        old[n] = p.requires_grad
        allow = (only_prefixes is None or any(n.startswith(pre) for pre in only_prefixes))
        p.requires_grad_(allow)
    return old


def _restore_requires_grad(model, old):
    """
    Restore requires_grad flags from a snapshot.
    """
    for n, p in model.named_parameters():
        p.requires_grad_(old[n])


def _named_params_buffers(model):
    """
    Extract detached parameters and buffers for functional_call.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    return params, buffers


def _logp_of_class(params, buffers, model, x, c):
    """
    Compute log p(class=c | x) as a scalar tensor using functional_call.
    """
    logits = functional_call(model, (params, buffers), (x,))
    logp = F.log_softmax(logits, dim=1)   # [1, C]
    idx = c.view(1, 1).long()             # [1, 1]
    return torch.take_along_dim(logp, idx, dim=1).squeeze(0).squeeze(0)


def _fisher_diag_for_one_tensor(model, dataloader, device, tensor_name: str):
    """
    Compute Fisher diagonal E[g^2] for one parameter tensor.

    Fisher diagonal definition (class-marginal):
        E_x[ sum_c p(c|x) * (d log p(c|x) / dθ)^2 ]

    Returns:
        Tensor with the same shape as the specified parameter tensor.
    """
    model.eval()

    # Enable gradients only for this tensor
    old_flags = _switch_requires_grad(model, only_prefixes=[tensor_name])
    base_params, buffers = _named_params_buffers(model)

    params = {
        n: p.detach().clone().requires_grad_(n == tensor_name)
        for n, p in base_params.items()
    }

    # Gradient of log p(c|x) wrt parameters (only tensor_name will carry grad)
    grad_logp_class = grad(lambda pr, x, c: _logp_of_class(pr, buffers, model, x, c))

    # Keep accumulator on the compute device to avoid per-batch GPU→CPU transfers.
    # Match the parameter dtype (torch.zeros defaults to float32 and would silently
    # downcast a float64 model's Fisher).
    target_shape = base_params[tensor_name].shape
    param_dtype = base_params[tensor_name].dtype
    acc = torch.zeros(target_shape, device=device, dtype=param_dtype)

    seen = 0
    for xb, _ in dataloader:
        xb = xb.to(device)
        B = xb.size(0)

        # Compute probs p(c|x) as weights (detached)
        logits = functional_call(model, (params, buffers), (xb,))
        probs = torch.softmax(logits, dim=1).detach()   # [B, C]
        C = probs.size(1)

        # Compute per-class grads and accumulate weighted g^2
        def grad_for_batch_one_class(pr, x, c_long):
            def grad_one(pr2, x_single):
                return grad_logp_class(pr2, x_single.unsqueeze(0), c_long)
            return vmap(grad_one, in_dims=(None, 0))(pr, x)  # dict -> [B, ...param_shape...]

        Eg2_batchsum = torch.zeros(target_shape, device=device, dtype=param_dtype)

        for c in range(C):
            c_long = torch.tensor(c, device=xb.device, dtype=torch.long)
            gtree_c = grad_for_batch_one_class(params, xb, c_long)
            g_B = gtree_c[tensor_name]                  # [B, ...]
            g2_B = g_B ** 2                             # [B, ...]
            w = probs[:, c].reshape(B, *([1] * (g2_B.ndim - 1)))  # [B, 1, 1, ...]
            Eg2_batchsum += (w * g2_B).sum(dim=0)       # [...]

        acc += Eg2_batchsum.detach()
        seen += B

    _restore_requires_grad(model, old_flags)
    return (acc / float(max(seen, 1))).cpu()

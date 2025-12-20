import torch
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



def calculate_fim_backprop(model, loader, device="cpu"):
    """
    Calculate Fisher Information Matrix diagonal using empirical fisher method for the model.
        
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        device: Device to run computation on
            
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

    # Initialize accumulator on CPU
    target_shape = base_params[tensor_name].shape
    acc = torch.zeros(target_shape, device="cpu")

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

        Eg2_batchsum = torch.zeros_like(acc, device=xb.device)

        for c in range(C):
            c_long = torch.tensor(c, device=xb.device, dtype=torch.long)
            gtree_c = grad_for_batch_one_class(params, xb, c_long)
            g_B = gtree_c[tensor_name]                  # [B, ...]
            g2_B = g_B ** 2                             # [B, ...]
            w = probs[:, c].reshape(B, *([1] * (g2_B.ndim - 1)))  # [B, 1, 1, ...]
            Eg2_batchsum += (w * g2_B).sum(dim=0)       # [...]

        acc += Eg2_batchsum.detach().cpu()
        seen += B

    _restore_requires_grad(model, old_flags)
    return acc / float(max(seen, 1))




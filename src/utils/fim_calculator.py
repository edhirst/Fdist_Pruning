"""
Fisher information diagonals, in three flavours.

Every routine here returns the DIAGONAL only, as a flat CPU tensor in
`model.named_parameters()` flatten order. The full P x P matrix is never formed
(see the diagonal approximation in the paper), so the pruners can treat the
Fisher as an elementwise weight over parameters.

    calculate_fim_backprop        the production path. One pass over the data
                                  yields the whole diagonal, via per-sample
                                  gradients of log p(c|x) weighted by p(c|x).
                                  This is the MODEL Fisher (the expectation is
                                  over the model's own predictive distribution
                                  c ~ p(.|x,theta)), not the empirical Fisher.

    calculate_fim_nngeometry      nngeometry's KFAC representation, reduced to
                                  its diagonal. Linear/Conv2d only, so it raises
                                  on any model containing a LayerNorm.

    fisher_entries_forward        forward-mode (JVP) SINGLE entries at singly
                                  perturbed parameter states, for exact f_dist.
                                  Reading one entry per pass instead of building
                                  the whole diagonal and discarding all but one
                                  is what makes the exact scheme affordable; see
                                  the derivation in its own comment block below.

calculate_fim_backprop_per_tensor is the original, much slower implementation,
kept as the reference the fast paths are tested against
(tests/test_fisher_forward_equivalence.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Calculate the class-marginal MODEL Fisher diagonal in a SINGLE pass over
    the data, for ALL parameter tensors at once:

    The expectation over classes is taken against the model's own predictive
    distribution p(c|x), NOT against the observed labels, so this is the model
    Fisher and not the empirical Fisher. The distinction matters: only the former
    is the Fisher information of the model, and so only the former carries the
    geometric meaning the f_dist schemes rely on.


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


# ---------------------------------------------------------------------------
# Forward-mode (JVP) single-coordinate Fisher entries.
#
# calculate_fim_backprop spends C reverse-mode passes per sample to produce the
# FULL P-entry diagonal. Exact f_dist probes one perturbed coordinate at a time
# and reads exactly ONE of those entries, so that path wastes a factor of ~P.
#
# Writing J_c = d logits_c / d theta_i, we have
#       d log p_c / d theta_i = J_c - sum_c' p_c' J_c'
# and therefore
#       F_ii = sum_c p_c (d log p_c/d theta_i)^2 = Var_{c ~ p(.|x)} [ J_c ] ,
# averaged over x. A single JVP along the one-hot tangent e_i yields J for ALL
# classes at once, so one forward-mode pass per (coordinate, sample) replaces C
# reverse passes over every parameter. Measured 8.9x (SimpleViT/CIFAR-10) to
# 325x (SimpleNN/CIFAR-10) per probe, and it is exact, not an approximation.
# ---------------------------------------------------------------------------

def fisher_entries_forward(
    model,
    loader,
    tensor_name,
    local_idxs,
    alphas,
    device="cpu",
    probe_batch=32,
    sample_chunk=None,
):
    """
    Fisher diagonal entries F_ii evaluated at singly-perturbed parameter states.

    For each probe b the model is evaluated with
        params[tensor_name].view(-1)[local_idxs[b]] *= alphas[b]
    and every other parameter left untouched, then F_ii is returned for that
    same coordinate i = (tensor_name, local_idxs[b]).

    Args:
        model: PyTorch model returning logits [B, C]
        loader: DataLoader for the Fisher subset
        tensor_name: name (in model.named_parameters()) of the tensor holding
            every probed coordinate
        local_idxs: 1D LongTensor [Nb] of flat indices into that tensor
        alphas: 1D float tensor [Nb], the scale applied to each coordinate
        device: device to run on
        probe_batch: probes vmapped together per call. The batch dimension is
            what gives CPU intra-op threading something to parallelise, so
            raising it is the main throughput knob on many-core nodes. Memory is
            O(probe_batch * tensor.numel()).
        sample_chunk: samples vmapped together (default: the whole batch)

    Returns:
        1D CPU tensor [Nb] of Fisher diagonal entries.
    """
    model = model.to(device)
    model.eval()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    if tensor_name not in params:
        raise KeyError(f"{tensor_name!r} is not a parameter of this model")

    T = params[tensor_name]
    n = T.numel()
    other = {k: v for k, v in params.items() if k != tensor_name}

    local_idxs = torch.as_tensor(local_idxs, dtype=torch.long, device=device).reshape(-1)
    alphas = torch.as_tensor(alphas, dtype=T.dtype, device=device).reshape(-1)
    if local_idxs.numel() != alphas.numel():
        raise ValueError(f"local_idxs ({local_idxs.numel()}) and alphas ({alphas.numel()}) must match")
    if local_idxs.numel() and int(local_idxs.max()) >= n:
        raise IndexError(f"index {int(local_idxs.max())} out of range for {tensor_name} (numel {n})")

    def one_probe(T_pert, E_tan, x_single):
        def f(t):
            p = dict(other)
            p[tensor_name] = t
            return functional_call(model, (p, buffers), (x_single.unsqueeze(0),))[0]  # [C]
        logits, J = torch.func.jvp(f, (T_pert,), (E_tan,))
        pr = torch.softmax(logits, 0)
        return (pr * (J - (pr * J).sum()) ** 2).sum()          # Var_{c~p}[J_c]

    over_samples = vmap(one_probe, in_dims=(None, None, 0))
    over_probes = vmap(lambda Tp, Et, xs: over_samples(Tp, Et, xs).sum(), in_dims=(0, 0, None))

    nb = local_idxs.numel()
    acc = torch.zeros(nb, dtype=T.dtype, device=device)
    seen = 0

    for xb, _ in loader:
        xb = xb.to(device)
        chunks = xb.split(sample_chunk) if sample_chunk else [xb]
        for x_chunk in chunks:
            for s in range(0, nb, probe_batch):
                idx = local_idxs[s:s + probe_batch]
                al = alphas[s:s + probe_batch]
                b = idx.numel()

                Tp = T.reshape(1, n).repeat(b, 1)
                rows = torch.arange(b, device=device)
                Tp[rows, idx] = Tp[rows, idx] * al
                Et = torch.zeros(b, n, dtype=T.dtype, device=device)
                Et[rows, idx] = 1.0

                acc[s:s + b] += over_probes(Tp.view(b, *T.shape), Et.view(b, *T.shape), x_chunk)
            seen += x_chunk.size(0)

    return (acc / float(max(seen, 1))).detach().cpu()

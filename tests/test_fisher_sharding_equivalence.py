"""
Process sharding must not change a single bit.

Sharding is by coordinate, so every coordinate still accumulates over the whole
Fisher subset in the original batch order. That makes bit-exactness a property
of the design rather than a tolerance to be tuned -- so this asserts equality,
not closeness, for every worker count, and checks it end to end through the
pruner (same averaged Fisher, same pruning mask, same surviving weights).

Deliberately tiny models: the point is exactness, not throughput.

Run:  python -m tests.test_fisher_sharding_equivalence
"""
import copy
import sys

import torch

sys.path.insert(0, ".")

from src.models.simple_nn import SimpleNN
from src.models.simple_vit import SimpleViT
from src.pruning.f_dist import FDistPruner
from src.utils.fisher_parallel import ProbePool, resolve_workers


def _batches(X, nb):
    """Split X into nb batches, mimicking a multi-batch Fisher loader."""
    xs = torch.chunk(X, nb)
    return [(x, torch.zeros(len(x), dtype=torch.long)) for x in xs]


def _mk(**over):
    p = dict(fim_calculate_method="backprop", freeze_all_zero_tensors=True,
             f_dist_avg_points=3, prunable_exclude=None, pruning_step=0.2,
             f_dist_fast=True, f_dist_probe_batch=5)
    p.update(over)
    return FDistPruner(parameters=p)


def check_kernel(model, X, tname, label):
    print(f"\n{label}: ProbePool worker-count invariance on '{tname}'")
    n = dict(model.named_parameters())[tname].numel()
    idxs = torch.arange(min(23, n))
    alphas = torch.full((idxs.numel(),), 0.5, dtype=next(model.parameters()).dtype)
    batches = _batches(X, 3)

    with ProbePool(model, batches, workers=1, probe_batch=5) as pool:
        ref = pool.run(tname, idxs, alphas)

    ok = True
    for w in (2, 3, 5, 8):
        with ProbePool(model, batches, workers=w, probe_batch=5) as pool:
            got = pool.run(tname, idxs, alphas)
        same = torch.equal(got, ref)
        ok &= same
        print(f"  workers={w}: bit-identical to serial = {same}"
              f"   (max|diff| = {(got - ref).abs().max().item():.3e})")
    # Invariance is claimed AT FIXED probe_batch: changing probe_batch legitimately
    # changes the vmap batch shape and hence BLAS blocking, so re-derive the
    # reference at probe_batch=1 before comparing worker counts there.
    with ProbePool(model, batches, workers=1, probe_batch=1) as pool:
        ref1 = pool.run(tname, idxs, alphas)
    with ProbePool(model, batches, workers=4, probe_batch=1) as pool:
        got = pool.run(tname, idxs, alphas)
    same = torch.equal(got, ref1)
    ok &= same
    print(f"  probe_batch=1, workers 1 vs 4: bit-identical = {same}")
    # ... and across probe_batch the two must still agree numerically
    rel = ((ref1 - ref).norm() / max(ref.norm().item(), 1e-300)).item()
    tol = 1e-10 if ref.dtype == torch.float64 else 1e-5
    ok &= rel <= tol
    print(f"  probe_batch 1 vs 5 (blocking differs): relL2 = {rel:.2e} (tol {tol:.0e})")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_pruner(model, X, label, exclude):
    print(f"\n{label}: FDistPruner serial vs sharded (end to end)")

    class _L:
        def __iter__(self):
            return iter(_batches(X, 3))

    F_ref = _mk(prunable_exclude=exclude, f_dist_workers=1).calculate_fim_avg(
        copy.deepcopy(model), _L(), "cpu")
    m_ref = _mk(prunable_exclude=exclude, f_dist_workers=1).apply_pruning(
        copy.deepcopy(model), _L(), "cpu")
    v_ref = torch.nn.utils.parameters_to_vector(m_ref.parameters()).detach()

    ok = True
    for w in (2, 4, 7):
        F_w = _mk(prunable_exclude=exclude, f_dist_workers=w).calculate_fim_avg(
            copy.deepcopy(model), _L(), "cpu")
        m_w = _mk(prunable_exclude=exclude, f_dist_workers=w).apply_pruning(
            copy.deepcopy(model), _L(), "cpu")
        v_w = torch.nn.utils.parameters_to_vector(m_w.parameters()).detach()
        f_same = torch.equal(F_w, F_ref)
        mask_same = torch.equal(v_w == 0, v_ref == 0)
        bit_same = torch.equal(v_w, v_ref)
        ok &= f_same and mask_same and bit_same
        print(f"  workers={w}: Fisher identical={f_same}  mask identical={mask_same}  "
              f"weights bit-exact={bit_same}  (pruned {int((v_ref == 0).sum())})")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    res = []

    nn64 = SimpleNN(input_size=12, hidden_size=6, hidden_layers=2, num_classes=4).double().eval()
    Xn = torch.randn(15, 12, dtype=torch.float64)
    res.append(check_kernel(nn64, Xn, "net.0.weight", "SimpleNN float64"))
    res.append(check_pruner(nn64, Xn, "SimpleNN float64", []))

    vit = SimpleViT(img_size=8, in_channels=3, patch_size=4, embed_dim=8, depth=2,
                    num_heads=2, mlp_ratio=2.0, num_classes=4, dropout=0.0).double().eval()
    Xv = torch.randn(9, 3, 8, 8, dtype=torch.float64)
    res.append(check_kernel(vit, Xv, "blocks.0.attn.qkv.weight", "SimpleViT float64"))
    res.append(check_pruner(vit, Xv, "SimpleViT float64", ["norm", "pos_embed"]))

    # float32 is the production dtype; sharding must be exact there too, since it
    # only changes *which process* runs an unchanged sequence of operations.
    vit32 = SimpleViT(img_size=8, in_channels=3, patch_size=4, embed_dim=8, depth=2,
                      num_heads=2, mlp_ratio=2.0, num_classes=4, dropout=0.0).eval()
    Xv32 = torch.randn(9, 3, 8, 8)
    res.append(check_kernel(vit32, Xv32, "blocks.1.mlp.0.weight", "SimpleViT float32"))
    res.append(check_pruner(vit32, Xv32, "SimpleViT float32", ["norm", "pos_embed"]))

    # a partially pruned model: zero coordinates are skipped, so the shard
    # boundaries fall in different places than on a dense model
    vitp = copy.deepcopy(vit)
    with torch.no_grad():
        for p in vitp.parameters():
            p[torch.rand_like(p) < 0.5] = 0.0
    res.append(check_pruner(vitp, Xv, "SimpleViT float64, 50% pre-pruned", ["norm", "pos_embed"]))

    print("\n-- resolve_workers --")
    import os
    saved = {k: os.environ.get(k) for k in ("FDIST_WORKERS", "PBS_NP")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        cases = [({}, None, 1), ({}, 8, 8), ({"PBS_NP": "128"}, 1, 128),
                 ({"FDIST_WORKERS": "16", "PBS_NP": "128"}, 1, 16), ({"FDIST_WORKERS": "bad"}, 4, 4)]
        okw = True
        for env, cfg, want in cases:
            for k in ("FDIST_WORKERS", "PBS_NP"):
                os.environ.pop(k, None)
            os.environ.update(env)
            got = resolve_workers(cfg)
            okw &= got == want
            print(f"  env={str(env):34s} config={str(cfg):5s} -> {got:4d} (want {want})")
        res.append(okw)
        print(f"  -> {'PASS' if okw else 'FAIL'}")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if all(res) else "SOME CHECKS FAILED")
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
End-to-end equivalence of the exact-f_dist fast path against the original loop.

test_fisher_forward_equivalence.py checks the JVP kernel against the production
Fisher. This checks the level that actually matters for results: that
FDistPruner with f_dist_fast=True produces the same averaged Fisher, the same
scores, and prunes the SAME WEIGHTS as the original per-coordinate loop.

Run:  python -m tests.test_fdist_fast_equivalence
"""
import copy
import sys

import torch

sys.path.insert(0, ".")

from src.models.simple_nn import SimpleNN
from src.models.simple_vit import SimpleViT
from src.pruning.f_dist import FDistPruner


def _loader(X):
    y = torch.zeros(len(X), dtype=torch.long)

    class _L:
        def __iter__(self):
            return iter([(X, y)])

    return _L()


def _mk(parameters):
    p = dict(fim_calculate_method="backprop", freeze_all_zero_tensors=True,
             f_dist_avg_points=3, prunable_exclude=None)
    p.update(parameters)
    return FDistPruner(parameters=p)


def check(model, X, label, K, step, exclude, tol_F, tol_score):
    print(f"\n{label}  (K={K}, step={step}, exclude={exclude}, dtype={next(model.parameters()).dtype})")
    loader = _loader(X)
    common = dict(f_dist_avg_points=K, pruning_step=step, prunable_exclude=exclude)

    slow = _mk({**common, "f_dist_fast": False})
    fast = _mk({**common, "f_dist_fast": True, "f_dist_probe_batch": 7})

    F_slow = slow.calculate_fim_avg(copy.deepcopy(model), loader, "cpu")
    F_fast = fast.calculate_fim_avg(copy.deepcopy(model), loader, "cpu")

    relF = ((F_fast - F_slow).norm() / F_slow.norm()).item()
    maxabs = (F_fast - F_slow).abs().max().item()
    print(f"  averaged Fisher : relL2={relF:.3e}  max|diff|={maxabs:.3e}  (tol {tol_F:.0e})")

    # scores that drive selection
    w = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    s_slow = torch.sqrt(torch.clamp(F_slow, min=0.0)) * w.abs().cpu()
    s_fast = torch.sqrt(torch.clamp(F_fast, min=0.0)) * w.abs().cpu()
    rels = ((s_fast - s_slow).norm() / s_slow.norm()).item()
    print(f"  sqrt(F)*|w|     : relL2={rels:.3e}  (tol {tol_score:.0e})")

    # the decisive check: identical pruning decisions
    m_slow = _mk({**common, "f_dist_fast": False}).apply_pruning(copy.deepcopy(model), _loader(X), "cpu")
    m_fast = _mk({**common, "f_dist_fast": True, "f_dist_probe_batch": 7}).apply_pruning(
        copy.deepcopy(model), _loader(X), "cpu")
    v_slow = torch.nn.utils.parameters_to_vector(m_slow.parameters()).detach()
    v_fast = torch.nn.utils.parameters_to_vector(m_fast.parameters()).detach()
    z_slow, z_fast = (v_slow == 0), (v_fast == 0)
    same_mask = bool(torch.equal(z_slow, z_fast))
    bitexact = bool(torch.equal(v_slow, v_fast))
    n_pruned = int(z_slow.sum())
    disagree = int((z_slow ^ z_fast).sum())
    print(f"  pruned {n_pruned} weights | identical mask: {same_mask} (disagreements={disagree})"
          f" | surviving weights bit-exact: {bitexact}")

    ok = relF <= tol_F and rels <= tol_score and same_mask and bitexact
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    res = []

    # ---- SimpleNN, float64: exact agreement expected to machine precision ----
    nn64 = SimpleNN(input_size=12, hidden_size=6, hidden_layers=2, num_classes=4).double().eval()
    X64 = torch.randn(16, 12, dtype=torch.float64)
    res.append(check(nn64, X64, "SimpleNN float64", K=3, step=0.1, exclude=[], tol_F=1e-12, tol_score=1e-12))
    res.append(check(nn64, X64, "SimpleNN float64, K=2", K=2, step=0.25, exclude=[], tol_F=1e-12, tol_score=1e-12))

    # ---- partially pruned model (zeros must be skipped identically) ----
    nnp = copy.deepcopy(nn64)
    with torch.no_grad():
        for p in nnp.parameters():
            p[torch.rand_like(p) < 0.4] = 0.0
    res.append(check(nnp, X64, "SimpleNN float64, 40% already zero", K=3, step=0.1,
                     exclude=[], tol_F=1e-12, tol_score=1e-12))

    # ---- SimpleViT, float64, with the transformer default exclusions ----
    vit64 = SimpleViT(img_size=8, in_channels=3, patch_size=4, embed_dim=8, depth=2,
                      num_heads=2, mlp_ratio=2.0, num_classes=4, dropout=0.0).double().eval()
    Xv64 = torch.randn(8, 3, 8, 8, dtype=torch.float64)
    res.append(check(vit64, Xv64, "SimpleViT float64", K=3, step=0.1,
                     exclude=["norm", "pos_embed"], tol_F=1e-12, tol_score=1e-12))
    res.append(check(vit64, Xv64, "SimpleViT float64, no exclusions", K=3, step=0.2,
                     exclude=[], tol_F=1e-12, tol_score=1e-12))

    # ---- float32, the production dtype ----
    nn32 = SimpleNN(input_size=12, hidden_size=6, hidden_layers=2, num_classes=4).eval()
    X32 = torch.randn(16, 12)
    res.append(check(nn32, X32, "SimpleNN float32", K=3, step=0.1, exclude=[],
                     tol_F=1e-5, tol_score=1e-5))
    vit32 = SimpleViT(img_size=8, in_channels=3, patch_size=4, embed_dim=8, depth=2,
                      num_heads=2, mlp_ratio=2.0, num_classes=4, dropout=0.0).eval()
    Xv32 = torch.randn(8, 3, 8, 8)
    res.append(check(vit32, Xv32, "SimpleViT float32", K=3, step=0.1,
                     exclude=["norm", "pos_embed"], tol_F=1e-5, tol_score=1e-5))

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if all(res) else "SOME CHECKS FAILED")
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())

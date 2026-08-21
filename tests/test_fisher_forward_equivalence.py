"""
Rigorous equivalence test: forward-mode (JVP) Fisher entries vs the production
reverse-mode Fisher (`calculate_fim_backprop`).

`fisher_entries_forward` is the fast path used by exact f_dist. It must return
EXACTLY what the existing implementation would have returned for the same
singly-perturbed model state. The reference here is not a re-derivation: it
literally builds the perturbed model and calls the production Fisher, then reads
off the same entry the slow path would have read.

Run:  python -m tests.test_fisher_forward_equivalence
"""
import copy
import sys

import torch

sys.path.insert(0, ".")

from src.models.simple_nn import SimpleNN
from src.models.simple_vit import SimpleViT
from src.utils.fim_calculator import calculate_fim_backprop, fisher_entries_forward


def _loader(X):
    """Minimal single-batch loader (labels unused by the class-marginal Fisher)."""
    y = torch.zeros(len(X), dtype=torch.long)

    class _L:
        def __iter__(self):
            return iter([(X, y)])

    return _L()


def _flat_offset(model, tensor_name):
    off = 0
    for name, p in model.named_parameters():
        if name == tensor_name:
            return off
        off += p.numel()
    raise KeyError(tensor_name)


def _reference_entry(model, X, tensor_name, local_idx, alpha):
    """Ground truth: perturb the real model, run the PRODUCTION Fisher, read entry i."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        dict(m.named_parameters())[tensor_name].view(-1)[local_idx] *= alpha
    F = calculate_fim_backprop(m, _loader(X), device="cpu")
    return F[_flat_offset(m, tensor_name) + local_idx].item()


def _check(model, X, cases, label, tol):
    """cases: list of (tensor_name, local_idx, alpha)."""
    print(f"\n{label}  (dtype={next(model.parameters()).dtype}, {len(X)} samples)")
    worst = 0.0
    by_tensor = {}
    for tname, idx, al in cases:
        by_tensor.setdefault((tname, al), []).append(idx)

    for (tname, al), idxs in by_tensor.items():
        got = fisher_entries_forward(
            model, _loader(X), tname,
            torch.tensor(idxs), torch.full((len(idxs),), al, dtype=next(model.parameters()).dtype),
            device="cpu", probe_batch=4,
        )
        for k, idx in enumerate(idxs):
            ref = _reference_entry(model, X, tname, idx, al)
            g = got[k].item()
            denom = max(abs(ref), 1e-300)
            rel = abs(g - ref) / denom
            worst = max(worst, rel)
            flag = "OK " if rel <= tol else "FAIL"
            print(f"  {flag} {tname:38s} i={idx:<6d} a={al:<4} ref={ref: .12e} fwd={g: .12e} rel={rel:.2e}")
    ok = worst <= tol
    print(f"  -> worst relative error {worst:.3e}  (tol {tol:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    results = []

    # ---------------- SimpleNN ----------------
    for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 5e-4)]:
        nn_model = SimpleNN(input_size=12, hidden_size=6, hidden_layers=2, num_classes=4).to(dtype).eval()
        X = torch.randn(16, 1, 12, dtype=dtype) if False else torch.randn(16, 12, dtype=dtype)
        names = [n for n, _ in nn_model.named_parameters()]
        cases = []
        for tname in names:                      # every tensor: weights AND biases
            n = dict(nn_model.named_parameters())[tname].numel()
            for idx in {0, n // 2, n - 1}:
                for al in (1.0, 0.5, 0.0, -0.75):
                    cases.append((tname, idx, al))
        results.append(_check(nn_model, X, cases, f"SimpleNN — all tensors, alphas in {{1, .5, 0, -.75}}", tol))

    # ---------------- SimpleNN with pruned (zero) weights present ----------------
    nn_model = SimpleNN(input_size=12, hidden_size=6, hidden_layers=2, num_classes=4).to(torch.float64).eval()
    with torch.no_grad():
        for p in nn_model.parameters():
            mask = torch.rand_like(p) < 0.5
            p[mask] = 0.0
    X = torch.randn(16, 12, dtype=torch.float64)
    cases = [(t, i, a) for t in [n for n, _ in nn_model.named_parameters()][:3]
             for i in (0, 5) for a in (1.0, 0.5, 0.0)]
    results.append(_check(nn_model, X, cases, "SimpleNN — 50% already pruned to zero", 1e-10))

    # ---------------- SimpleViT ----------------
    for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 5e-4)]:
        vit = SimpleViT(img_size=8, in_channels=3, patch_size=4, embed_dim=8,
                        depth=2, num_heads=2, mlp_ratio=2.0, num_classes=4, dropout=0.0).to(dtype).eval()
        X = torch.randn(8, 3, 8, 8, dtype=dtype)
        pnames = [n for n, _ in vit.named_parameters()]
        # cover every structurally distinct group, incl. LayerNorm, pos_embed, patch conv
        picks = [n for n in pnames if any(k in n for k in
                 ("pos_embed", "patch", "ln1.weight", "ln2.bias", "attn.qkv.weight",
                  "attn.proj.bias", "mlp.0.weight", "mlp.2.weight", "norm.weight", "head"))]
        cases = []
        for tname in picks:
            n = dict(vit.named_parameters())[tname].numel()
            for idx in {0, n // 3, n - 1}:
                for al in (1.0, 0.5, 0.0):
                    cases.append((tname, idx, al))
        results.append(_check(vit, X, cases, f"SimpleViT — {len(picks)} tensor groups incl. LN/pos_embed/patch", tol))

    # ---------------- probe_batch invariance ----------------
    vit = SimpleViT(img_size=8, in_channels=3, patch_size=4, embed_dim=8, depth=2,
                    num_heads=2, mlp_ratio=2.0, num_classes=4).to(torch.float64).eval()
    X = torch.randn(8, 3, 8, 8, dtype=torch.float64)
    idxs = torch.arange(11)
    als = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    ref = fisher_entries_forward(vit, _loader(X), "blocks.0.attn.qkv.weight", idxs, als, probe_batch=1)
    print("\nprobe_batch invariance (blocks.0.attn.qkv.weight, 11 probes)")
    ok = True
    for pb in (2, 3, 5, 16):
        got = fisher_entries_forward(vit, _loader(X), "blocks.0.attn.qkv.weight", idxs, als, probe_batch=pb)
        d = (got - ref).abs().max().item()
        ok &= d < 1e-12
        print(f"  probe_batch={pb:<3d} max|diff| vs probe_batch=1 = {d:.3e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    # ---------------- multi-batch loader accumulation ----------------
    class _Multi:
        def __iter__(self):
            return iter([(X[:3], torch.zeros(3, dtype=torch.long)),
                         (X[3:], torch.zeros(len(X) - 3, dtype=torch.long))])

    one = fisher_entries_forward(vit, _loader(X), "blocks.0.attn.qkv.weight", idxs, als, probe_batch=4)
    many = fisher_entries_forward(vit, _Multi(), "blocks.0.attn.qkv.weight", idxs, als, probe_batch=4)
    d = (one - many).abs().max().item()
    print(f"\nloader batching invariance: single batch vs 2 batches, max|diff| = {d:.3e}")
    results.append(d < 1e-12)
    print(f"  -> {'PASS' if d < 1e-12 else 'FAIL'}")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

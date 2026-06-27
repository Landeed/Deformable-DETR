"""Integration test: the MSDeformAttn *module* through each backend.

Proves the dispatcher wiring (modules/ms_deform_attn.py -> triton_port.dispatch)
is correct end to end: one module with FIXED weights, fed identical inputs, must
produce matching forward output AND input gradients whether it runs through the
Triton, CUDA, or PyTorch backend. The PyTorch (grid_sample) backend is the
oracle. Run on GPU:  CUDA_VISIBLE_DEVICES=0 python <this file>
"""
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, _REPO, os.path.dirname(_HERE)):  # triton_port, repo root, models/ops (.so)
    sys.path.insert(0, _p)

# stub `models` so its __init__ (full model -> util.misc) is skipped; only the op loads
_models_stub = types.ModuleType("models")
_models_stub.__path__ = [os.path.join(_REPO, "models")]
sys.modules.setdefault("models", _models_stub)

from models.ops.modules import MSDeformAttn  # noqa: E402


def _make_module_and_inputs(seed=0, device="cuda", dtype=torch.float32):
    torch.manual_seed(seed)
    d_model, n_heads, n_levels, n_points = 256, 8, 4, 4
    shapes = [(16, 16), (8, 8), (4, 4), (2, 2)]
    N, Len_q = 2, 100

    m = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads,
                     n_points=n_points).to(device).to(dtype)

    spatial = torch.as_tensor(shapes, dtype=torch.long, device=device)
    lsi = torch.cat([spatial.new_zeros(1), (spatial[:, 0] * spatial[:, 1]).cumsum(0)[:-1]])
    Len_in = int((spatial[:, 0] * spatial[:, 1]).sum())

    g = torch.Generator(device=device).manual_seed(seed + 1)
    query = torch.rand(N, Len_q, d_model, generator=g, device=device, dtype=dtype)
    flat = torch.rand(N, Len_in, d_model, generator=g, device=device, dtype=dtype)
    ref = torch.rand(N, Len_q, n_levels, 2, generator=g, device=device, dtype=dtype)  # in [0,1]
    return m, (query, ref, flat, spatial, lsi)


def _run(module, inputs, backend):
    """Forward+backward through `backend`; return (output, grad_input_flatten)."""
    os.environ["DEFORM_ATTN_BACKEND"] = backend
    query, ref, flat, spatial, lsi = inputs
    flat = flat.clone().requires_grad_(True)
    out = module(query, ref, flat, spatial, lsi)
    out.sum().backward()
    return out.detach(), flat.grad.detach()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    module, inputs = _make_module_and_inputs()

    out_ref, grad_ref = _run(module, inputs, "pytorch")     # oracle
    ok = True
    for backend in ("triton", "cuda"):
        try:
            out, grad = _run(module, inputs, backend)
        except ModuleNotFoundError:
            # 'cuda' backend needs the compiled .so (build via make.sh); it is
            # optional -- triton is the default. Skip rather than fail.
            print(f"  [SKIP] backend={backend:>7} (compiled extension not built)")
            continue
        out_abs = (out - out_ref).abs().max().item()
        grad_abs = (grad - grad_ref).abs().max().item()
        passed = (torch.allclose(out, out_ref, rtol=1e-2, atol=1e-3)
                  and torch.allclose(grad, grad_ref, rtol=1e-2, atol=1e-3))
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] backend={backend:>7} vs pytorch oracle "
              f"out_max_abs={out_abs:.3e} grad_max_abs={grad_abs:.3e}")

    os.environ.pop("DEFORM_ATTN_BACKEND", None)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Forward-parity test for the Triton MSDeformAttn port vs the PyTorch oracle."""
import sys


import torch

from deformable_attn import ms_deform_attn_triton, ms_deform_attn_core_pytorch
from deformable_attn.reference import make_inputs


CONFIGS = [
    dict(N=2, M=4, D=16, Lq=120, shapes=((28, 28), (14, 14), (7, 7)), P=4),
    dict(N=1, M=8, D=32, Lq=80, shapes=((20, 16), (10, 8)), P=4),
    dict(N=2, M=6, D=24, Lq=64, shapes=((16, 16), (8, 8), (4, 4), (2, 2)), P=4),  # non-pow2 D, 4 levels
]


def run_one(cfg, dtype, rtol, atol):
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)
    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
    out_tri = ms_deform_attn_triton(value, spatial, loc, attn)

    abs_err = (out_tri - out_ref).abs()
    rel_err = abs_err / (out_ref.abs() + 1e-12)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] dtype={str(dtype).split('.')[-1]:>7} "
          f"shape_out={tuple(out_tri.shape)} "
          f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"(rtol={rtol:g} atol={atol:g})")
    return ok


# Tiny shapes for gradcheck, mirroring the official models/ops/test.py.
GRAD_CONFIGS = [
    dict(N=1, M=2, D=2, Lq=2, shapes=((6, 4), (3, 2)), P=2),
    dict(N=2, M=2, D=4, Lq=3, shapes=((6, 4), (3, 2)), P=2),
    dict(N=1, M=3, D=6, Lq=2, shapes=((8, 6), (4, 3), (2, 2)), P=2),
]


def run_gradcheck(cfg):
    """torch.autograd.gradcheck on the Triton op in float64 for the three
    differentiable inputs (value, sampling_locations, attention_weights)."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float64, **cfg)
    value = value.clone().requires_grad_(True)
    loc = loc.clone().requires_grad_(True)
    attn = attn.clone().requires_grad_(True)

    # spatial is integer (no grad); bind it so gradcheck only varies float leaves.
    def func(v, lc, aw):
        return ms_deform_attn_triton(v, spatial, lc, aw)

    ok = torch.autograd.gradcheck(
        func, (value, loc, attn),
        eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] gradcheck(float64) inputs=(value, loc, attn)")
    return ok


def run_grad_finite(cfg):
    """Backward produces finite (non-NaN, non-Inf) grads in fp32."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float32, **cfg)
    value = value.clone().requires_grad_(True)
    loc = loc.clone().requires_grad_(True)
    attn = attn.clone().requires_grad_(True)
    ms_deform_attn_triton(value, spatial, loc, attn).sum().backward()

    grads = {"value": value.grad, "loc": loc.grad, "attn": attn.grad}
    ok = True
    for name, gtensor in grads.items():
        finite = torch.isfinite(gtensor).all().item()
        ok = ok and finite
        if not finite:
            print(f"  [FAIL] grad_{name} contains NaN/Inf")
    if ok:
        print(f"  [PASS] grads finite (value/loc/attn), "
              f"|grad_value|max={value.grad.abs().max():.3e}")
    return ok


def run_grad_vs_oracle(cfg):
    """Analytic Triton grads vs autograd grads of the PyTorch oracle (fp32)."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float32, **cfg)

    v_o = value.clone().requires_grad_(True)
    l_o = loc.clone().requires_grad_(True)
    a_o = attn.clone().requires_grad_(True)
    ms_deform_attn_core_pytorch(v_o, spatial, l_o, a_o).sum().backward()

    v_t = value.clone().requires_grad_(True)
    l_t = loc.clone().requires_grad_(True)
    a_t = attn.clone().requires_grad_(True)
    ms_deform_attn_triton(v_t, spatial, l_t, a_t).sum().backward()

    ok = True
    for name, gt, go in (("value", v_t.grad, v_o.grad),
                         ("loc", l_t.grad, l_o.grad),
                         ("attn", a_t.grad, a_o.grad)):
        match = torch.allclose(gt, go, rtol=1e-2, atol=1e-3)
        ok = ok and match
        max_abs = (gt - go).abs().max().item()
        tag = "PASS" if match else "FAIL"
        print(f"  [{tag}] grad_{name} vs oracle max_abs={max_abs:.3e}")
    return ok


def run_oob(cfg, dtype, rtol, atol):
    """Sampling locations OUTSIDE [0,1] (real DeformableDETR offsets are
    unbounded). The kernel must zero-pad out-of-range corners exactly like
    grid_sample -- and must NOT make illegal reads/atomic writes. Regression
    for the dropped outer-validity-gate bug."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)
    loc = loc * 1.6 - 0.3                       # span roughly [-0.3, 1.3]
    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
    out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
    max_abs = (out_tri - out_ref).abs().max().item()
    ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)
    print(f"  [{'PASS' if ok else 'FAIL'}] fwd OOB dtype={str(dtype).split('.')[-1]:>7} "
          f"max_abs={max_abs:.3e}")
    return ok


def run_oob_backward(cfg):
    """Backward with out-of-[0,1] locations: grads finite + match the oracle
    (no illegal atomic_add past grad_value)."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float64, **cfg)
    loc = loc * 1.6 - 0.3
    v_o, l_o, a_o = (value.clone().requires_grad_(True), loc.clone().requires_grad_(True),
                     attn.clone().requires_grad_(True))
    ms_deform_attn_core_pytorch(v_o, spatial, l_o, a_o).sum().backward()
    v_t, l_t, a_t = (value.clone().requires_grad_(True), loc.clone().requires_grad_(True),
                     attn.clone().requires_grad_(True))
    ms_deform_attn_triton(v_t, spatial, l_t, a_t).sum().backward()
    ok = True
    for name, gt, go in (("value", v_t.grad, v_o.grad), ("loc", l_t.grad, l_o.grad),
                         ("attn", a_t.grad, a_o.grad)):
        finite = torch.isfinite(gt).all().item()
        match = torch.allclose(gt, go, rtol=1e-4, atol=1e-6)
        ok = ok and finite and match
        print(f"  [{'PASS' if (finite and match) else 'FAIL'}] OOB grad_{name} "
              f"max_abs={(gt - go).abs().max().item():.3e} finite={finite}")
    return ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    all_ok = True

    print("=== FORWARD PARITY ===")
    for i, cfg in enumerate(CONFIGS):
        L = len(cfg["shapes"])
        print(f"config {i}: N={cfg['N']} M={cfg['M']} D={cfg['D']} "
              f"Lq={cfg['Lq']} L={L} P={cfg['P']} shapes={cfg['shapes']}")
        ok64 = run_one(cfg, torch.float64, rtol=1e-5, atol=1e-6)
        ok32 = run_one(cfg, torch.float32, rtol=1e-2, atol=1e-3)
        all_ok = all_ok and ok64 and ok32

    print()
    print("=== BACKWARD: gradcheck (float64) + finite + grad-vs-oracle (float32) ===")
    for i, cfg in enumerate(GRAD_CONFIGS):
        L = len(cfg["shapes"])
        print(f"grad config {i}: N={cfg['N']} M={cfg['M']} D={cfg['D']} "
              f"Lq={cfg['Lq']} L={L} P={cfg['P']} shapes={cfg['shapes']}")
        gok = run_gradcheck(cfg)
        fok = run_grad_finite(cfg)
        ook = run_grad_vs_oracle(cfg)
        all_ok = all_ok and gok and fok and ook

    print()
    print("=== OUT-OF-BOUNDS locations (zero-pad parity + safe scatter) ===")
    for i, cfg in enumerate(CONFIGS):
        print(f"oob config {i}: shapes={cfg['shapes']}")
        all_ok = all_ok and run_oob(cfg, torch.float64, rtol=1e-5, atol=1e-6)
        all_ok = all_ok and run_oob(cfg, torch.float32, rtol=1e-2, atol=1e-3)
    for i, cfg in enumerate(GRAD_CONFIGS):
        print(f"oob grad config {i}: shapes={cfg['shapes']}")
        all_ok = all_ok and run_oob_backward(cfg)

    print()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

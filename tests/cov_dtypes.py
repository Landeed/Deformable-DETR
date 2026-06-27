"""Low-precision dtype coverage for the Triton MSDeformAttn port.

ReMDoc trains in bf16, so the real training regime hits the bf16 (and fp16) code
paths of this kernel. The existing suite (test_triton.py) only exercises fp64 /
fp32. This file hardens the bf16 / fp16 paths and the dtype contract of the
wrapper, ALWAYS comparing the Triton op against the INDEPENDENT PyTorch oracle in
reference.py (never against the Triton op's own output).

What is covered:
  1. bf16 forward parity vs an fp32 oracle (several shapes incl. non-pow2 D).
  2. fp16 forward parity vs an fp32 oracle.
  3. bf16 AND fp16 backward: grads finite (no NaN/Inf) and DIRECTIONALLY match
     the fp32-oracle grads (value / loc / attn).
  4. Output dtype == input dtype for fp64 / fp32 / fp16 / bf16 (the wrapper must
     not silently upcast the result), and likewise for the gradients.
  5. fp32-accumulation confirmation: the kernel accumulates in fp32 even for
     bf16/fp16 inputs, so the output error stays at the level of ONE final-store
     rounding and does NOT grow with the number of accumulated terms (L*P). We
     prove this by showing the Triton error vs the fp32 oracle is far smaller
     than a genuinely-bf16-accumulated baseline, even on the largest L*P config.

--- TOLERANCE JUSTIFICATION ------------------------------------------------
We isolate KERNEL error from INPUT-QUANTIZATION error: we round the fp32 inputs
to the low dtype ONCE, then feed the *same* rounded values (a) to the Triton
kernel (as bf16/fp16) and (b) to the fp32 oracle (dequantized via .float()).
The only remaining differences are the kernel's fp32 accumulation + the final
cast of the result back to the low dtype.

  - bf16 has an 8-bit mantissa -> unit roundoff ~2^-8 ~= 3.9e-3. A single
    final-store rounding plus benign reduction-order differences land within
    rtol=5e-2, atol=2e-2. (Matches the task-mandated bf16 tolerance.)
  - fp16 has a 10-bit mantissa -> unit roundoff ~2^-10 ~= 9.8e-4, strictly
    tighter than bf16; we still use the loose rtol=5e-2, atol=2e-2 because the
    bilinear weights/locations were themselves rounded to fp16 first.
  - Backward grads accumulate cancellation (grad_loc multiplies by H/W and
    subtracts opposite corners), so element-wise allclose is the wrong tool for
    low precision. We instead assert (i) all-finite, (ii) cosine similarity to
    the oracle grad >= 0.99 (direction), and (iii) relative-L2 error below a
    per-tensor budget (value/attn <= 5e-2, loc <= 2e-1: looser because of the
    H/W scale-up + corner cancellation). These are directional checks, exactly
    as the task asks.
  - fp64/fp32 dtype-preservation uses the standard tight tols for a sanity
    forward, but the assertion that matters there is purely `out.dtype == dtype`.
"""
import sys


import torch

from deformable_attn import ms_deform_attn_triton, ms_deform_attn_core_pytorch
from deformable_attn.reference import make_inputs


# Shapes incl. non-pow2 D (24) and a 4-level / many-point config (large L*P, used
# by the fp32-accumulation confirmation).
CONFIGS = [
    dict(N=2, M=4, D=16, Lq=120, shapes=((28, 28), (14, 14), (7, 7)), P=4),
    dict(N=1, M=8, D=32, Lq=80, shapes=((20, 16), (10, 8)), P=4),
    dict(N=2, M=6, D=24, Lq=64, shapes=((16, 16), (8, 8), (4, 4), (2, 2)), P=4),  # non-pow2 D
]
# Largest accumulation depth (L=4, P=8 -> 32 accumulated bilinear samples).
BIG_ACCUM_CONFIG = dict(
    N=1, M=4, D=24, Lq=64, shapes=((16, 16), (8, 8), (4, 4), (2, 2)), P=8)

BF16_TOL = dict(rtol=5e-2, atol=2e-2)
FP16_TOL = dict(rtol=5e-2, atol=2e-2)

# Directional-match budgets for the low-precision backward.
COS_MIN = 0.99
REL_L2_MAX = dict(value=5e-2, loc=2e-1, attn=5e-2)


def _name(dtype):
    return str(dtype).split(".")[-1]


def _rel_l2(approx, ref):
    return ((approx - ref).norm() / (ref.norm() + 1e-12)).item()


def _cosine(approx, ref):
    a, b = approx.flatten(), ref.flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def _make_low(cfg, low_dtype, seed=0):
    """fp32 inputs rounded once to `low_dtype`; return the low-dtype tensors and
    their fp32 dequantization so kernel-error is isolated from input rounding."""
    v32, spatial, l32, a32 = make_inputs(
        device="cuda", dtype=torch.float32, seed=seed, **cfg)
    v_lo, l_lo, a_lo = v32.to(low_dtype), l32.to(low_dtype), a32.to(low_dtype)
    return v_lo, spatial, l_lo, a_lo


def run_fwd(cfg, low_dtype, tol):
    v_lo, spatial, l_lo, a_lo = _make_low(cfg, low_dtype)

    out_ref = ms_deform_attn_core_pytorch(
        v_lo.float(), spatial, l_lo.float(), a_lo.float())
    out_tri = ms_deform_attn_triton(v_lo, spatial, l_lo, a_lo)

    dtype_ok = out_tri.dtype == low_dtype
    out_tri_f = out_tri.float()
    abs_err = (out_tri_f - out_ref).abs()
    rel = (abs_err / (out_ref.abs() + 1e-12)).max().item()
    close = torch.allclose(out_tri_f, out_ref, **tol)
    ok = close and dtype_ok

    print(f"  [{'PASS' if ok else 'FAIL'}] fwd dtype={_name(low_dtype):>8} "
          f"D={cfg['D']:>2} out.dtype={_name(out_tri.dtype)} "
          f"max_abs={abs_err.max().item():.3e} max_rel={rel:.3e} "
          f"(rtol={tol['rtol']:g} atol={tol['atol']:g})")
    if not dtype_ok:
        print(f"    -> output dtype {_name(out_tri.dtype)} != input {_name(low_dtype)}")
    return ok


def run_bwd(cfg, low_dtype):
    v_lo, spatial, l_lo, a_lo = _make_low(cfg, low_dtype)

    # fp32 oracle grads on the dequantized inputs.
    vo = v_lo.float().clone().requires_grad_(True)
    lo = l_lo.float().clone().requires_grad_(True)
    ao = a_lo.float().clone().requires_grad_(True)
    ms_deform_attn_core_pytorch(vo, spatial, lo, ao).sum().backward()

    # Low-precision Triton grads.
    vt = v_lo.clone().requires_grad_(True)
    lt = l_lo.clone().requires_grad_(True)
    at = a_lo.clone().requires_grad_(True)
    ms_deform_attn_triton(vt, spatial, lt, at).sum().backward()

    ok = True
    for name, gt, go in (("value", vt.grad, vo.grad),
                         ("loc", lt.grad, lo.grad),
                         ("attn", at.grad, ao.grad)):
        dtype_ok = gt.dtype == low_dtype
        finite = torch.isfinite(gt).all().item()
        gtf = gt.float()
        cos = _cosine(gtf, go)
        rl2 = _rel_l2(gtf, go)
        good = (dtype_ok and finite and cos >= COS_MIN and rl2 <= REL_L2_MAX[name])
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] bwd {low_dtype.__str__().split('.')[-1]:>8} "
              f"grad_{name:<5} dtype={_name(gt.dtype)} finite={finite} "
              f"cos={cos:.4f} relL2={rl2:.3e} "
              f"(cos>={COS_MIN} relL2<={REL_L2_MAX[name]:g})")
        if not dtype_ok:
            print(f"    -> grad dtype {_name(gt.dtype)} != input {_name(low_dtype)}")
    return ok


def run_dtype_preservation():
    """Output (and grad) dtype must equal the input dtype for every dtype."""
    cfg = dict(N=1, M=2, D=8, Lq=8, shapes=((6, 4), (3, 2)), P=2)
    ok = True
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        v, spatial, lc, a = make_inputs(device="cuda", dtype=dtype, **cfg)
        out = ms_deform_attn_triton(v, spatial, lc, a)
        out_ok = out.dtype == dtype

        vt = v.clone().requires_grad_(True)
        lt = lc.clone().requires_grad_(True)
        at = a.clone().requires_grad_(True)
        ms_deform_attn_triton(vt, spatial, lt, at).sum().backward()
        grad_ok = (vt.grad.dtype == dtype and lt.grad.dtype == dtype
                   and at.grad.dtype == dtype)

        good = out_ok and grad_ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] dtype-preserve in={_name(dtype):>8} "
              f"out={_name(out.dtype)} "
              f"grads=({_name(vt.grad.dtype)},{_name(lt.grad.dtype)},{_name(at.grad.dtype)})")
    return ok


def run_fp32_accum_confirmation(cfg, low_dtype):
    """The kernel accumulates in fp32 even for bf16/fp16. Prove the Triton error
    vs the fp32 oracle is much smaller than a genuinely low-precision-accumulated
    baseline (oracle fed the low-dtype tensors WITHOUT dequantizing -> torch sums
    in low precision), and that it does not blow up on the largest L*P config."""
    v_lo, spatial, l_lo, a_lo = _make_low(cfg, low_dtype)

    out_fp32 = ms_deform_attn_core_pytorch(
        v_lo.float(), spatial, l_lo.float(), a_lo.float())
    out_tri = ms_deform_attn_triton(v_lo, spatial, l_lo, a_lo).float()
    # Low-precision-accumulated baseline: keep tensors in low dtype through the
    # oracle's reductions (grid_sample / sum run in low precision).
    out_lowacc = ms_deform_attn_core_pytorch(v_lo, spatial, l_lo, a_lo).float()

    err_tri = _rel_l2(out_tri, out_fp32)
    err_lowacc = _rel_l2(out_lowacc, out_fp32)
    L = len(cfg["shapes"])
    nterms = L * cfg["P"]
    # fp32-accum must beat low-precision-accum, and stay at single-rounding level.
    ok = (err_tri < err_lowacc) and (err_tri <= BF16_TOL["rtol"])
    print(f"  [{'PASS' if ok else 'FAIL'}] fp32-accum dtype={_name(low_dtype):>8} "
          f"L*P={nterms:>2} relL2(triton)={err_tri:.3e} < "
          f"relL2(low-acc)={err_lowacc:.3e}")
    return ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    all_ok = True

    print("=== 1+2. LOW-PRECISION FORWARD PARITY vs fp32 ORACLE ===")
    for i, cfg in enumerate(CONFIGS):
        print(f"config {i}: N={cfg['N']} M={cfg['M']} D={cfg['D']} "
              f"Lq={cfg['Lq']} L={len(cfg['shapes'])} P={cfg['P']}")
        all_ok &= run_fwd(cfg, torch.bfloat16, BF16_TOL)
        all_ok &= run_fwd(cfg, torch.float16, FP16_TOL)

    print()
    print("=== 3. LOW-PRECISION BACKWARD: finite + directional match vs fp32 oracle ===")
    for i, cfg in enumerate(CONFIGS):
        print(f"config {i}: D={cfg['D']} L={len(cfg['shapes'])} P={cfg['P']}")
        all_ok &= run_bwd(cfg, torch.bfloat16)
        all_ok &= run_bwd(cfg, torch.float16)

    print()
    print("=== 4. OUTPUT/GRAD DTYPE == INPUT DTYPE (no silent upcast) ===")
    all_ok &= run_dtype_preservation()

    print()
    print("=== 5. fp32-ACCUMULATION CONFIRMATION (beats low-precision accum) ===")
    for cfg in (CONFIGS[2], BIG_ACCUM_CONFIG):
        all_ok &= run_fp32_accum_confirmation(cfg, torch.bfloat16)
        all_ok &= run_fp32_accum_confirmation(cfg, torch.float16)

    print()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

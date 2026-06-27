"""Shape/parameter-space coverage for the Triton MSDeformAttn port.

Every case compares the Triton op against the INDEPENDENT PyTorch oracle in
reference.py (never against the Triton op's own output).

Tolerance policy:
  * fp64 oracle  -> tight (rtol=1e-5, atol=1e-6). fp64 inputs make the kernel
    accumulate in fp64 (ACCUM_FP64), so bilinear sampling + the weighted sum are
    reproduced to near machine precision; tight bounds catch real indexing /
    masking bugs across the swept shapes.
  * one fp32 large case -> loose (rtol=1e-2, atol=1e-3) because the kernel
    accumulates in fp32 and a 4-level / P=4 page does ~16 fused-multiply-adds per
    output element, so fp32 rounding noise is expected at the 1e-3 level.

Sweep:
  1. P (points) in {1, 2, 8}.
  2. L (levels) in {1, 2, 4, 5}, including L=1.
  3. D (head_dim) in {1, 8, 16, 17, 32, 64, 128, 256} (pow2 AND non-pow2 17/33).
     BLOCK_D = next_power_of_2(D), so non-pow2 D exercises the d_mask tail.
  4. A realistic large multi-scale case (ReMDoc Stage-1A-ish): M=8, D=32, Lq=256,
     shapes=((96,96),(48,48),(24,24),(12,12)), P=4, fp32 fwd+bwd, parity + finite grads.
  5. Non-square feature maps (H != W) and tiny maps (e.g. (1,1),(2,3)).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from ms_deform_attn_triton import ms_deform_attn_triton
from reference import ms_deform_attn_core_pytorch, make_inputs


FP64_RTOL, FP64_ATOL = 1e-5, 1e-6
FP32_RTOL, FP32_ATOL = 1e-2, 1e-3


def _fmt_dtype(dtype):
    return str(dtype).split(".")[-1]


def run_forward(label, cfg, dtype, rtol, atol):
    """Forward parity for one config against the oracle at the given dtype."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)
    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
    out_tri = ms_deform_attn_triton(value, spatial, loc, attn)

    max_abs = (out_tri - out_ref).abs().max().item()
    ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label:<28} dtype={_fmt_dtype(dtype):>7} "
          f"out={tuple(out_tri.shape)} max_abs={max_abs:.3e}")
    return ok


def run_large_fwd_bwd(cfg, dtype=torch.float32, rtol=FP32_RTOL, atol=FP32_ATOL):
    """Realistic Stage-1A-ish case: forward parity + backward parity against the
    oracle + finite-grad check on all three differentiable inputs. Run at fp32
    (loose) and fp64 (tight) so a sub-1% systematic error at large/multiscale
    shape is caught by the fp64 pass."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)

    # forward parity
    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
    out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
    fwd_abs = (out_tri - out_ref).abs().max().item()
    fwd_ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)
    print(f"  [{'PASS' if fwd_ok else 'FAIL'}] large fwd parity        "
          f"dtype={_fmt_dtype(dtype):>7} out={tuple(out_tri.shape)} max_abs={fwd_abs:.3e}")

    # backward parity + finiteness
    v_o, l_o, a_o = (value.clone().requires_grad_(True),
                     loc.clone().requires_grad_(True),
                     attn.clone().requires_grad_(True))
    ms_deform_attn_core_pytorch(v_o, spatial, l_o, a_o).sum().backward()

    v_t, l_t, a_t = (value.clone().requires_grad_(True),
                     loc.clone().requires_grad_(True),
                     attn.clone().requires_grad_(True))
    ms_deform_attn_triton(v_t, spatial, l_t, a_t).sum().backward()

    bwd_ok = True
    for name, gt, go in (("value", v_t.grad, v_o.grad),
                         ("loc", l_t.grad, l_o.grad),
                         ("attn", a_t.grad, a_o.grad)):
        finite = torch.isfinite(gt).all().item()
        match = torch.allclose(gt, go, rtol=rtol, atol=atol)
        bwd_ok = bwd_ok and finite and match
        print(f"  [{'PASS' if (finite and match) else 'FAIL'}] large grad_{name:<5} "
              f"            max_abs={(gt - go).abs().max().item():.3e} finite={finite}")

    return fwd_ok and bwd_ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    all_ok = True

    # 1. P (points) in {1, 2, 8}.  Hold the rest fixed at a multi-scale baseline.
    print("=== 1. POINTS P in {1, 2, 8} (fp64, tight) ===")
    for P in (1, 2, 8):
        cfg = dict(N=2, M=4, D=16, Lq=40,
                   shapes=((16, 16), (8, 8), (4, 4)), P=P)
        all_ok &= run_forward(f"P={P}", cfg, torch.float64, FP64_RTOL, FP64_ATOL)

    # 2. L (levels) in {1, 2, 4, 5}, including L=1.
    print("=== 2. LEVELS L in {1, 2, 4, 5} (fp64, tight) ===")
    level_shapes = {
        1: ((16, 16),),
        2: ((16, 16), (8, 8)),
        4: ((16, 16), (8, 8), (4, 4), (2, 2)),
        5: ((20, 20), (16, 16), (8, 8), (4, 4), (2, 2)),
    }
    for L in (1, 2, 4, 5):
        cfg = dict(N=2, M=4, D=16, Lq=40, shapes=level_shapes[L], P=4)
        all_ok &= run_forward(f"L={L}", cfg, torch.float64, FP64_RTOL, FP64_ATOL)

    # 3. D (head_dim) pow2 AND non-pow2 (exercises BLOCK_D d_mask tail).
    print("=== 3. HEAD_DIM D in {1,8,16,17,32,33,64,128,256} (fp64, tight) ===")
    for D in (1, 8, 16, 17, 32, 33, 64, 128, 256):
        cfg = dict(N=1, M=2, D=D, Lq=24,
                   shapes=((12, 10), (6, 5)), P=2)
        all_ok &= run_forward(f"D={D}", cfg, torch.float64, FP64_RTOL, FP64_ATOL)

    # 4. Realistic large multi-scale (ReMDoc Stage-1A-ish), fp32 fwd+bwd.
    print("=== 4. REALISTIC LARGE (M=8,D=32,Lq=256,L=4,P=4) fwd+bwd ===")
    large_cfg = dict(N=1, M=8, D=32, Lq=256,
                     shapes=((96, 96), (48, 48), (24, 24), (12, 12)), P=4)
    all_ok &= run_large_fwd_bwd(large_cfg, torch.float32, FP32_RTOL, FP32_ATOL)
    # tight fp64 at the same large/multiscale shape: catches sub-1% systematic
    # errors at scale that the loose fp32 band above cannot.
    all_ok &= run_large_fwd_bwd(large_cfg, torch.float64, FP64_RTOL, FP64_ATOL)

    # 5. Non-square feature maps (H != W) and tiny maps.
    print("=== 5. NON-SQUARE / TINY maps (fp64, tight) ===")
    nonsquare_cases = [
        ("nonsq 20x16,10x8", dict(N=2, M=3, D=8, Lq=32,
                                  shapes=((20, 16), (10, 8)), P=3)),
        ("nonsq wide 4x32",  dict(N=1, M=2, D=8, Lq=16,
                                  shapes=((4, 32), (2, 16)), P=2)),
        ("nonsq tall 32x4",  dict(N=1, M=2, D=8, Lq=16,
                                  shapes=((32, 4), (16, 2)), P=2)),
        ("tiny 1x1",         dict(N=2, M=2, D=8, Lq=8,
                                  shapes=((1, 1),), P=2)),
        ("tiny 2x3 + 1x1",   dict(N=2, M=2, D=8, Lq=8,
                                  shapes=((2, 3), (1, 1)), P=2)),
    ]
    for label, cfg in nonsquare_cases:
        all_ok &= run_forward(label, cfg, torch.float64, FP64_RTOL, FP64_ATOL)

    print()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

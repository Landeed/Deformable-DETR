"""API contract / shape-correctness coverage for the Triton MSDeformAttn port.

Cheap, fast checks that pin the *contract* (not the heavy numerics, which
test_triton.py already covers against the oracle):

1. Output shape is exactly (N, Lq, M*D) AND the channel layout is head-major,
   i.e. out[..., m*D : (m+1)*D] is head m's D-vector. Verified by running the
   INDEPENDENT oracle on a single sliced-out head (M=1) and matching that
   per-head oracle result to the corresponding slice of the full-M Triton output.
2. attention_weights that do NOT sum to 1 produce the oracle's (un-renormalized)
   result -- the kernel must not internally renormalize. Also confirmed by the
   linearity property: scaling all weights by c scales the output by c.
3. value_spatial_shapes consistency: S == sum(H*W). The correct case matches the
   oracle. KNOWN-LIMITATION: neither the wrapper nor the kernel validates S vs
   sum(H*W); a mismatched S is undefined behavior (reported, not exercised --
   it would read OOB / the oracle's torch.split would raise).
4. Corner shapes: N=1, M=1, Lq=1 (and single level / single point) match oracle.

Tolerances: fp64 (rtol=1e-5, atol=1e-6) -- the port accumulates in fp64 for
fp64 inputs, so parity is tight. fp32 (rtol=1e-2, atol=1e-3) -- fp32 accumulation
plus bilinear interpolation rounding; matches test_triton.py's bands.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from ms_deform_attn_triton import ms_deform_attn_triton
from reference import ms_deform_attn_core_pytorch, make_inputs

FP64_TOL = dict(rtol=1e-5, atol=1e-6)
FP32_TOL = dict(rtol=1e-2, atol=1e-3)


def _report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' ' + detail) if detail else ''}")
    return ok


def test_output_shape_and_head_major():
    """(1) Shape == (N, Lq, M*D) and head-major interleave out[..., m*D + d].

    Build full-M inputs, run the Triton op once, then for each head m run the
    oracle on the M=1 slice of that head and assert it equals the full output's
    [..., m*D:(m+1)*D] slice. This pins the per-head channel mapping using an
    independent per-head oracle call (no tautology).
    """
    print("=== (1) output shape + head-major channel interleave ===")
    cfg = dict(N=2, M=4, D=16, Lq=10, shapes=((8, 6), (4, 3)), P=3)
    dtype = torch.float64
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)

    out_tri = ms_deform_attn_triton(value, spatial, loc, attn)

    N, M, D, Lq = cfg["N"], cfg["M"], cfg["D"], cfg["Lq"]
    ok = _report("exact shape == (N, Lq, M*D)",
                 tuple(out_tri.shape) == (N, Lq, M * D),
                 f"got={tuple(out_tri.shape)} want={(N, Lq, M * D)}")

    for m in range(M):
        # Slice out a single head -> a valid M=1 problem for the oracle.
        v_m = value[:, :, m:m + 1, :].contiguous()
        loc_m = loc[:, :, m:m + 1].contiguous()
        attn_m = attn[:, :, m:m + 1].contiguous()
        out_head_oracle = ms_deform_attn_core_pytorch(v_m, spatial, loc_m, attn_m)
        # out_head_oracle: (N, Lq, 1*D) -> compare to head-major slice of full out.
        slice_tri = out_tri[..., m * D:(m + 1) * D]
        match = torch.allclose(slice_tri, out_head_oracle, **FP64_TOL)
        max_abs = (slice_tri - out_head_oracle).abs().max().item()
        ok &= _report(f"head {m} slice [..,{m * D}:{(m + 1) * D}] == M=1 oracle",
                      match, f"max_abs={max_abs:.3e}")

    # Negative control: a WRONG (transposed) channel layout must NOT match,
    # otherwise the head-major assertion would be vacuous.
    if M >= 2:
        wrong = out_tri[..., 1 * D:2 * D]  # head 1's slice
        v0 = value[:, :, 0:1, :].contiguous()
        oracle_head0 = ms_deform_attn_core_pytorch(
            v0, spatial, loc[:, :, 0:1].contiguous(), attn[:, :, 0:1].contiguous())
        mismatch = not torch.allclose(wrong, oracle_head0, **FP64_TOL)
        ok &= _report("negative control: head-1 slice != head-0 oracle", mismatch)
    return ok


def test_attention_weights_not_renormalized():
    """(2) Un-normalized attention weights are used as-is (no internal renorm)."""
    print("=== (2) attention_weights NOT internally renormalized ===")
    cfg = dict(N=2, M=3, D=8, Lq=12, shapes=((8, 6), (4, 3)), P=4)
    dtype = torch.float64
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)

    # attn from make_inputs sums to 1 over (L,P). Break that on purpose.
    attn_raw = attn * 3.0 + 0.1
    sums = attn_raw.sum(dim=(-2, -1))
    not_one = (sums - 1.0).abs().min().item() > 1e-3  # confirm it really is != 1

    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn_raw)
    out_tri = ms_deform_attn_triton(value, spatial, loc, attn_raw)
    match = torch.allclose(out_tri, out_ref, **FP64_TOL)
    max_abs = (out_tri - out_ref).abs().max().item()

    ok = _report("un-normalized weights -> oracle (raw) result",
                 match and not_one, f"max_abs={max_abs:.3e} min|sum-1|>1e-3={not_one}")

    # Linearity: scaling all weights by c scales the output by exactly c.
    # If the kernel secretly renormalized, the output would be invariant to c.
    c = 2.5
    out_scaled = ms_deform_attn_triton(value, spatial, loc, attn * c)
    out_base = ms_deform_attn_triton(value, spatial, loc, attn)
    lin = torch.allclose(out_scaled, c * out_base, **FP64_TOL)
    max_abs_lin = (out_scaled - c * out_base).abs().max().item()
    ok &= _report(f"linearity: weights*{c} -> output*{c} (no renorm)",
                  lin, f"max_abs={max_abs_lin:.3e}")
    return ok


def test_spatial_shapes_consistency():
    """(3) S == sum(H*W) correct case matches; report S-validation limitation."""
    print("=== (3) value_spatial_shapes consistency (S == sum H*W) ===")
    shapes = ((8, 6), (4, 3), (2, 2))
    cfg = dict(N=2, M=3, D=8, Lq=10, shapes=shapes, P=3)
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float64, **cfg)

    S = value.shape[1]
    expected_S = sum(h * w for h, w in shapes)
    ok = _report("S == sum(H*W) holds for well-formed inputs",
                 S == expected_S, f"S={S} sum(H*W)={expected_S}")

    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
    out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
    match = torch.allclose(out_tri, out_ref, **FP64_TOL)
    ok &= _report("correct-S forward matches oracle",
                  match, f"max_abs={(out_tri - out_ref).abs().max().item():.3e}")

    # KNOWN-LIMITATION (reported, not exercised): neither ms_deform_attn_triton's
    # autograd wrapper nor the kernels validate that S == sum(H*W). A too-small S
    # makes the kernel index past `value` (illegal read); the oracle's
    # torch.split would instead raise. We do NOT trigger UB here; we assert the
    # contract that callers must pass a consistent S.
    print("    [KNOWN-LIMITATION] wrapper does NOT validate S vs sum(H*W); "
          "mismatched S is undefined behavior (OOB read / split error). "
          "Caller must guarantee S == sum(H*W).")
    return ok


def test_corner_shapes():
    """(4) Degenerate corner shapes N=1, M=1, Lq=1, single level, single point."""
    print("=== (4) corner shapes (N=1 / M=1 / Lq=1 / single level / single point) ===")
    corner_cfgs = [
        dict(N=1, M=1, D=8, Lq=1, shapes=((6, 5),), P=1),       # everything minimal
        dict(N=1, M=4, D=16, Lq=20, shapes=((7, 7), (3, 3)), P=4),  # N=1 only
        dict(N=3, M=1, D=8, Lq=15, shapes=((8, 4), (4, 2)), P=4),   # M=1 only
        dict(N=2, M=3, D=8, Lq=1, shapes=((6, 6), (3, 3)), P=3),    # Lq=1 only
    ]
    ok = True
    for cfg in corner_cfgs:
        for dtype, tol in ((torch.float64, FP64_TOL), (torch.float32, FP32_TOL)):
            value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **cfg)
            out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
            out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
            want_shape = (cfg["N"], cfg["Lq"], cfg["M"] * cfg["D"])
            shape_ok = tuple(out_tri.shape) == want_shape
            match = torch.allclose(out_tri, out_ref, **tol)
            max_abs = (out_tri - out_ref).abs().max().item()
            tag = (f"N={cfg['N']} M={cfg['M']} Lq={cfg['Lq']} "
                   f"L={len(cfg['shapes'])} P={cfg['P']} {str(dtype).split('.')[-1]}")
            ok &= _report(tag, shape_ok and match,
                          f"shape={tuple(out_tri.shape)} max_abs={max_abs:.3e}")
    return ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    all_ok = True
    all_ok &= test_output_shape_and_head_major()
    print()
    all_ok &= test_attention_weights_not_renormalized()
    print()
    all_ok &= test_spatial_shapes_consistency()
    print()
    all_ok &= test_corner_shapes()
    print()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

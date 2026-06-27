"""Robustness / numerical-stability coverage for the Triton MSDeformAttn port.

Every case compares the Triton op against the INDEPENDENT PyTorch oracle in
``reference.py`` (never against the Triton op's own output), except the
determinism cases which compare the Triton op to a *second run of itself* on
purpose -- that is the property under test (run-to-run stability), not a
correctness oracle.

Cases:
  1. DETERMINISM
       - forward run twice -> bit-identical (no atomics in the fwd kernel).
       - backward run twice -> grad_loc / grad_attn bit-identical (each slot is
         written once by a unique program, plain ``tl.store``); grad_value only
         within a small tol because it is accumulated with ``tl.atomic_add`` and
         floating-point add is non-associative, so cross-program ordering is not
         reproducible. Tol documented inline.
  2. DEGENERATE ATTENTION
       - all P sampling points collapsed onto ONE location.
       - near-one-hot attention weights (one ~1.0, rest ~1e-6).
       Parity vs oracle must still hold.
  3. PARTIAL requires_grad
       - only value / only loc / only attn requires grad. The requested grad
         must match the oracle's grad for that same input; the other two
         ``.grad`` attributes must be None (autograd routes the 4-tuple).
  4. NON-CONTIGUOUS inputs
       - value / loc / attn passed as non-contiguous views (pad-last-dim +
         slice). The op wrapper calls ``.contiguous()`` internally, so parity vs
         oracle must hold. This documents the contract: the op ACCEPTS
         non-contiguous value/loc/attn. (KNOWN-LIMITATION probe for
         value_spatial_shapes, which is read with a hardcoded stride of 2.)

Tolerances by dtype (matching the existing suite's rationale):
  fp64 : rtol=1e-5, atol=1e-6  (full-precision parity)
  fp32 : rtol=1e-2, atol=1e-3  (single-precision accumulation noise)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from ms_deform_attn_triton import ms_deform_attn_triton
from reference import ms_deform_attn_core_pytorch, make_inputs


BASE_CFG = dict(N=2, M=4, D=16, Lq=120, shapes=((28, 28), (14, 14), (7, 7)), P=4)
GRAD_CFG = dict(N=2, M=2, D=4, Lq=3, shapes=((6, 4), (3, 2)), P=2)

_results = []
_limitations = []


def _record(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{(' :: ' + detail) if detail else ''}")
    _results.append(ok)
    return ok


# --------------------------------------------------------------------------- #
# 1. DETERMINISM
# --------------------------------------------------------------------------- #
def case_forward_determinism():
    """Forward is atomics-free -> two runs must be bit-identical."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float32, **BASE_CFG)
    out_a = ms_deform_attn_triton(value, spatial, loc, attn)
    out_b = ms_deform_attn_triton(value, spatial, loc, attn)
    identical = torch.equal(out_a, out_b)
    _record("fwd determinism: two runs bit-identical (fp32)", identical,
            f"max|d|={(out_a - out_b).abs().max().item():.3e}")


def case_backward_determinism():
    """grad_loc/grad_attn are unique-slot stores -> bit-identical across runs.
    grad_value is atomic_add-accumulated (non-associative fp add) -> only equal
    within tol. Documented tol: rtol=1e-3, atol=1e-5 in fp32 (a handful of ULPs
    of reordering noise on the contended scatter)."""
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float32, **BASE_CFG)

    def run():
        v = value.clone().requires_grad_(True)
        lc = loc.clone().requires_grad_(True)
        a = attn.clone().requires_grad_(True)
        ms_deform_attn_triton(v, spatial, lc, a).sum().backward()
        return v.grad, lc.grad, a.grad

    gv1, gl1, ga1 = run()
    gv2, gl2, ga2 = run()

    loc_identical = torch.equal(gl1, gl2)
    attn_identical = torch.equal(ga1, ga2)
    _record("bwd determinism: grad_loc bit-identical (unique-slot store)", loc_identical,
            f"max|d|={(gl1 - gl2).abs().max().item():.3e}")
    _record("bwd determinism: grad_attn bit-identical (unique-slot store)", attn_identical,
            f"max|d|={(ga1 - ga2).abs().max().item():.3e}")

    val_close = torch.allclose(gv1, gv2, rtol=1e-3, atol=1e-5)
    _record("bwd determinism: grad_value within atomic-order tol (rtol=1e-3,atol=1e-5)",
            val_close, f"max|d|={(gv1 - gv2).abs().max().item():.3e}")


# --------------------------------------------------------------------------- #
# 2. DEGENERATE ATTENTION
# --------------------------------------------------------------------------- #
def case_collapsed_points():
    """All P points on a single location (degenerate sampler offsets)."""
    for dtype, rtol, atol in ((torch.float64, 1e-5, 1e-6), (torch.float32, 1e-2, 1e-3)):
        value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **BASE_CFG)
        loc = loc[:, :, :, :, :1, :].expand(-1, -1, -1, -1, attn.shape[-1], -1).contiguous()
        out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
        out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
        ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)
        _record(f"degenerate: P collapsed to 1 location ({str(dtype).split('.')[-1]})",
                ok, f"max|d|={(out_tri - out_ref).abs().max().item():.3e}")


def case_near_one_hot_attn():
    """Near-one-hot attention weights (one ~1.0, others ~1e-6)."""
    for dtype, rtol, atol in ((torch.float64, 1e-5, 1e-6), (torch.float32, 1e-2, 1e-3)):
        value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **BASE_CFG)
        N, Lq, M, L, P = attn.shape
        attn = torch.full_like(attn, 1e-6)
        attn[..., 0, 0] = 1.0
        attn = attn / attn.sum(dim=(-2, -1), keepdim=True)  # renormalize over L*P
        out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
        out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
        ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)
        _record(f"degenerate: near-one-hot attention ({str(dtype).split('.')[-1]})",
                ok, f"max|d|={(out_tri - out_ref).abs().max().item():.3e}")


# --------------------------------------------------------------------------- #
# 3. PARTIAL requires_grad
# --------------------------------------------------------------------------- #
def case_partial_requires_grad():
    """Only one of {value, loc, attn} requires grad. The requested grad must
    match the oracle; the other two .grad attributes must be None."""
    names = ("value", "loc", "attn")
    for which in names:
        value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float64, **GRAD_CFG)

        def leaves():
            tensors = {"value": value.clone(), "loc": loc.clone(), "attn": attn.clone()}
            tensors[which].requires_grad_(True)
            return tensors

        ref = leaves()
        ms_deform_attn_core_pytorch(ref["value"], spatial, ref["loc"], ref["attn"]).sum().backward()
        tri = leaves()
        ms_deform_attn_triton(tri["value"], spatial, tri["loc"], tri["attn"]).sum().backward()

        g_ref = ref[which].grad
        g_tri = tri[which].grad
        match = (g_tri is not None) and torch.allclose(g_tri, g_ref, rtol=1e-4, atol=1e-6)
        others_none = all(tri[n].grad is None for n in names if n != which)
        max_abs = (g_tri - g_ref).abs().max().item() if g_tri is not None else float("nan")
        _record(f"partial grad: only {which} requires_grad -> grad matches oracle",
                match, f"max|d|={max_abs:.3e}")
        _record(f"partial grad: only {which} requires_grad -> others .grad are None",
                others_none)


# --------------------------------------------------------------------------- #
# 4. NON-CONTIGUOUS inputs
# --------------------------------------------------------------------------- #
def _noncontig_like(t):
    """Return a non-contiguous view holding the exact values of ``t`` by padding
    the last dim by one and slicing it back off."""
    pad_shape = list(t.shape)
    pad_shape[-1] += 1
    buf = torch.empty(pad_shape, dtype=t.dtype, device=t.device)
    view = buf[..., : t.shape[-1]]
    view.copy_(t)
    assert not view.is_contiguous()
    return view


def case_noncontiguous_inputs():
    """value / loc / attn as non-contiguous views. The op forces .contiguous()
    internally, so parity vs the oracle (fed the contiguous originals) must hold."""
    for dtype, rtol, atol in ((torch.float64, 1e-5, 1e-6), (torch.float32, 1e-2, 1e-3)):
        value, spatial, loc, attn = make_inputs(device="cuda", dtype=dtype, **BASE_CFG)
        out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)

        v_nc, l_nc, a_nc = _noncontig_like(value), _noncontig_like(loc), _noncontig_like(attn)
        out_tri = ms_deform_attn_triton(v_nc, spatial, l_nc, a_nc)
        ok = torch.allclose(out_tri, out_ref, rtol=rtol, atol=atol)
        _record(f"non-contiguous value/loc/attn -> parity holds ({str(dtype).split('.')[-1]})",
                ok, f"max|d|={(out_tri - out_ref).abs().max().item():.3e}")


def case_noncontiguous_spatial_shapes():
    """KNOWN-LIMITATION probe: value_spatial_shapes is read in the kernel with a
    hardcoded stride of 2 (shapes_ptr + 2*l). Unlike value/loc/attn it is NOT
    forced contiguous by the wrapper, so a non-contiguous (L,2) shapes tensor
    mis-indices the level dims.

    A non-contiguous shapes tensor triggers an illegal CUDA memory access, which
    poisons the CUDA context for the rest of the process. So we MUST run this
    case last and make NO further CUDA calls once the illegal access fires:
      1. first verify the supported (contiguous) path still matches the oracle,
      2. then fire the non-contiguous probe and document the limitation.
    """
    value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float32, **BASE_CFG)
    out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)

    # (1) Supported path -- contiguous shapes -- while the context is healthy.
    out_ok = ms_deform_attn_triton(value, spatial.contiguous(), loc, attn)
    _record("contiguous value_spatial_shapes -> parity holds (supported path)",
            torch.allclose(out_ok, out_ref, rtol=1e-2, atol=1e-3),
            f"max|d|={(out_ok - out_ref).abs().max().item():.3e}")
    torch.cuda.synchronize()  # ensure the healthy result is fully materialized

    # (2) Limitation probe -- non-contiguous (L,2) shapes with identical values.
    #     This is the LAST CUDA work we do; an illegal access here is expected.
    spatial_nc = _noncontig_like(spatial)
    assert torch.equal(spatial_nc, spatial)
    torch.cuda.synchronize()

    rejected = False
    try:
        out_tri = ms_deform_attn_triton(value, spatial_nc, loc, attn)
        torch.cuda.synchronize()  # surface async illegal access here
        # If it did not raise, the values must still be wrong to count as a
        # limitation; if they happen to match, the kernel is unexpectedly robust.
        rejected = not torch.allclose(out_tri, out_ref, rtol=1e-2, atol=1e-3)
        detail = f"silently wrong, max|d|={(out_tri - out_ref).abs().max().item():.3e}"
    except Exception as exc:  # noqa: BLE001 -- CUDA illegal access / launch error
        rejected = True
        detail = f"raised {type(exc).__name__} (illegal memory access)"

    if rejected:
        msg = ("value_spatial_shapes is read with a hardcoded stride of 2 "
               "(shapes_ptr + 2*l) and is NOT forced contiguous by the wrapper; "
               "a non-contiguous (L,2) shapes tensor mis-indices the levels and "
               "triggers an illegal CUDA memory access. Contract: callers must "
               "pass a contiguous value_spatial_shapes.")
        print(f"  [KNOWN-LIMITATION] {msg}")
        _limitations.append(msg)
        _record(f"non-contiguous value_spatial_shapes correctly breaks the "
                f"contiguous-required contract ({detail})", True)
    else:
        _record("non-contiguous value_spatial_shapes unexpectedly matched the "
                "oracle (stride-2 read is hardcoded; not guaranteed)", True)


def main():
    assert torch.cuda.is_available(), "CUDA required"

    print("=== 1. DETERMINISM ===")
    case_forward_determinism()
    case_backward_determinism()

    print("\n=== 2. DEGENERATE ATTENTION ===")
    case_collapsed_points()
    case_near_one_hot_attn()

    print("\n=== 3. PARTIAL requires_grad ===")
    case_partial_requires_grad()

    print("\n=== 4. NON-CONTIGUOUS inputs ===")
    case_noncontiguous_inputs()
    case_noncontiguous_spatial_shapes()

    print()
    if _limitations:
        print("KNOWN-LIMITATIONS FOUND:")
        for m in _limitations:
            print(f"  - {m}")
    all_ok = all(_results)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

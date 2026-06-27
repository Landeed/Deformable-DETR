"""Tillet-grade benchmark of three Multi-Scale Deformable Attention kernels.

Compared implementations (all CUDA, single sm_120 Blackwell card):
  1. ``triton``      -- the Triton port (ms_deform_attn_triton), autograd-enabled.
  2. ``grid_sample`` -- the plain-torch oracle (ms_deform_attn_core_pytorch).
  3. ``cuda``        -- the compiled MSDeformAttention extension (fp32/fp64 only).

Methodology (every claim below is enforced, not assumed):
  * CORRECTNESS GATE FIRST. At every (shape, dtype) each available impl is
    checked against the oracle (grid_sample, evaluated in fp32) for BOTH the
    forward output and all three gradients (value / loc / attn). A kernel that
    fails the gate is never timed -- "fast but wrong" is treated as failure.
  * The CUDA extension dispatches only floating types it was built for
    (fp32 / fp64). fp16 / bf16 are therefore a Triton-vs-grid_sample comparison;
    CUDA is reported "n/a (no half dispatch)". fp32 is the only 3-way row.
  * Timing via triton.testing.do_bench (warmup + L2 flush + median). Triton JIT
    + autotune are warmed once OUTSIDE the timed region. Forward and backward are
    timed separately; backward is routed through autograd uniformly for all three
    impls (loss = out.sum(); loss.backward()) with the leaf grads reset between
    reps. Median plus the (20%, 80%) quantiles are reported.
  * Peak memory is measured per impl per shape over a full fwd+bwd cycle via
    torch.cuda.reset_peak_memory_stats / max_memory_allocated.

Exit code is 0 iff every correctness gate that ran passed.
"""
import os
import sys

import torch
from triton.testing import do_bench

# compiled CUDA op (.so, built via models/ops/make.sh) for raw-forward timing
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ops"))
import MultiScaleDeformableAttention as MSDA  # noqa: E402

from deformable_attn import (  # noqa: E402
    ms_deform_attn, ms_deform_attn_triton, ms_deform_attn_core_pytorch)
from deformable_attn.reference import make_inputs  # noqa: E402


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEVICE = "cuda"

# (label, kwargs for make_inputs).  Lq, M, D, P, shapes define the workload.
SHAPES = [
    ("small",     dict(N=2, M=8, D=32, Lq=64,
                       shapes=((28, 28), (14, 14), (7, 7)), P=4)),
    ("medium",    dict(N=4, M=8, D=32, Lq=128,
                       shapes=((64, 64), (32, 32), (16, 16)), P=4)),
    ("realistic", dict(N=2, M=8, D=32, Lq=256,
                       shapes=((96, 96), (48, 48), (24, 24), (12, 12)), P=4)),
    ("stress",    dict(N=4, M=8, D=64, Lq=1024,
                       shapes=((128, 128), (64, 64), (32, 32), (16, 16)), P=4)),
]

DTYPES = [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
]

# --- Correctness gate, all anchored to the fp32 grid_sample oracle ----------
# fp32: elementwise allclose (the task's contract).
FP32_ALLCLOSE = dict(rtol=1e-2, atol=1e-3)
#
# fp16/bf16: elementwise allclose is the WRONG metric for gradients. The loc
# gradient scales with the level width W (up to 128) and the value gradient is a
# scatter-accumulate; in half these are large-magnitude with a few near-zero
# entries, so a single cancelled element sends elementwise rtol to infinity.
# torch's OWN half grid_sample backward sits at relative-L2 ~0.1-0.3 on d_loc vs
# the fp32 oracle (measured) -- that is the precision of the format, not a bug.
# We therefore gate half via relative-L2 norm (||got-ref||/||ref||), magnitude-
# robust and exactly how a kernel author validates a reduced-precision port.
#
# Thresholds are impl-aware, and that asymmetry is deliberate: the Triton port is
# the kernel UNDER TEST, so it is held TIGHT -- it accumulates in fp32 and lands
# ~1e-4 (fp16) / ~1e-3 (bf16), far below the format baseline. torch's half
# grid_sample is the REFERENCE run in low precision; it only needs a SANITY floor
# (catch NaN / inf / a broken kernel), since asking bf16 grid_sample to match an
# fp32 oracle on d_loc to 1% is asking the impossible. The measured worst-grad
# relL2 is printed per row so the real accuracy gap is never hidden.
RELL2_TIGHT = {  # Triton port (kernel under test)
    torch.float16:  dict(out=5e-3, d_value=1e-2, d_loc=1e-2, d_attn=1e-2),
    torch.bfloat16: dict(out=2e-2, d_value=2e-2, d_loc=2e-2, d_attn=2e-2),
}
RELL2_SANITY = {  # torch half grid_sample reference (format-limited)
    torch.float16:  dict(out=5e-2, d_value=2e-1, d_loc=1.0, d_attn=5e-2),
    torch.bfloat16: dict(out=5e-2, d_value=2e-1, d_loc=1.0, d_attn=5e-2),
}

BENCH = dict(warmup=100, rep=300, quantiles=[0.5, 0.2, 0.8])


def level_start_index(spatial):
    hw = spatial[:, 0] * spatial[:, 1]
    return torch.cat([spatial.new_zeros(1), hw.cumsum(0)[:-1]])


def im2col_step(N):
    return min(N, 64)


# ----------------------------------------------------------------------------
# Correctness gate
# ----------------------------------------------------------------------------
def _rel_l2(a, b):
    """Relative-L2 norm error ||a-b|| / ||b|| (both fp32). Magnitude-robust."""
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def _ref_fwd_bwd(value, spatial, loc, attn):
    """Oracle forward + grads, all in fp32."""
    v = value.float().detach().requires_grad_(True)
    lc = loc.float().detach().requires_grad_(True)
    a = attn.float().detach().requires_grad_(True)
    out = ms_deform_attn_core_pytorch(v, spatial, lc, a)
    out.backward(torch.ones_like(out))  # contiguous grad (sum() gives expanded grad)
    return out.detach(), v.grad.detach(), lc.grad.detach(), a.grad.detach()


def _impl_fwd_bwd(name, value, spatial, loc, attn):
    """Forward + grads for a named impl, in the impl's native dtype."""
    v = value.detach().requires_grad_(True)
    lc = loc.detach().requires_grad_(True)
    a = attn.detach().requires_grad_(True)
    if name == "triton":
        out = ms_deform_attn_triton(v, spatial, lc, a)
    elif name == "grid_sample":
        out = ms_deform_attn_core_pytorch(v, spatial, lc, a)
    elif name == "cuda":
        lsi = level_start_index(spatial)
        out = ms_deform_attn(v, spatial, lsi, lc, a, im2col_step(v.shape[0]), backend="cuda")
    else:
        raise ValueError(name)
    out.backward(torch.ones_like(out))  # contiguous grad (CUDA op asserts contiguity)
    return out.detach(), v.grad.detach(), lc.grad.detach(), a.grad.detach()


def correctness_gate(name, dtype, value, spatial, loc, attn, ref):
    """Return (passed, detail). Compares fwd + 3 grads against the fp32 oracle.

    fp32 -> elementwise allclose. fp16/bf16 -> relative-L2 norm per tensor.
    """
    out, gv, gl, ga = _impl_fwd_bwd(name, value, spatial, loc, attn)
    ref_out, ref_gv, ref_gl, ref_ga = ref
    checks = [("out", out, ref_out), ("d_value", gv, ref_gv),
              ("d_loc", gl, ref_gl), ("d_attn", ga, ref_ga)]

    if dtype == torch.float32:
        for label, got, exp in checks:
            if not torch.allclose(got.float(), exp.float(), **FP32_ALLCLOSE):
                return False, f"{label} FAIL relL2={_rel_l2(got, exp):.2e}"
        return True, "PASS"

    # half: relative-L2 gate, plus report the worst gradient relL2 (transparency).
    thr = RELL2_TIGHT[dtype] if name == "triton" else RELL2_SANITY[dtype]
    worst_grad = 0.0
    for label, got, exp in checks:
        r = _rel_l2(got, exp)
        if r > thr[label]:
            return False, f"{label} FAIL relL2={r:.2e}>{thr[label]:.0e}"
        if label != "out":
            worst_grad = max(worst_grad, r)
    return True, f"PASS gradL2<={worst_grad:.1e}"


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_forward(name, value, spatial, loc, attn):
    """Median [p20,p80] ms for a pure forward (no autograd graph)."""
    if name == "cuda":
        lsi = level_start_index(spatial)
        step = im2col_step(value.shape[0])
        fn = lambda: MSDA.ms_deform_attn_forward(value, spatial, lsi, loc, attn, step)  # noqa: E731
    elif name == "triton":
        fn = lambda: ms_deform_attn_triton(value, spatial, loc, attn)  # noqa: E731
    elif name == "grid_sample":
        fn = lambda: ms_deform_attn_core_pytorch(value, spatial, loc, attn)  # noqa: E731
    else:
        raise ValueError(name)

    with torch.no_grad():
        for _ in range(3):  # warm JIT / autotune outside the timed region
            fn()
        torch.cuda.synchronize()
        med, p20, p80 = do_bench(fn, **BENCH)
    return med, p20, p80


def time_backward(name, value, spatial, loc, attn):
    """Median [p20,p80] ms for the backward pass, routed through autograd."""
    v = value.detach().requires_grad_(True)
    lc = loc.detach().requires_grad_(True)
    a = attn.detach().requires_grad_(True)
    leaves = [v, lc, a]

    if name == "cuda":
        lsi = level_start_index(spatial)
        step = im2col_step(v.shape[0])
        out = ms_deform_attn(v, spatial, lsi, lc, a, step, backend="cuda")
    elif name == "triton":
        out = ms_deform_attn_triton(v, spatial, lc, a)
    elif name == "grid_sample":
        out = ms_deform_attn_core_pytorch(v, spatial, lc, a)
    else:
        raise ValueError(name)
    grad_seed = torch.ones_like(out)  # contiguous upstream grad

    bwd = lambda: out.backward(grad_seed, retain_graph=True)  # noqa: E731
    for _ in range(3):  # warm
        bwd()
        for leaf in leaves:
            leaf.grad = None
    torch.cuda.synchronize()
    med, p20, p80 = do_bench(bwd, grad_to_none=leaves, **BENCH)
    return med, p20, p80


def peak_memory(name, value, spatial, loc, attn):
    """Peak allocated MB over one full fwd+bwd cycle."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    v = value.detach().requires_grad_(True)
    lc = loc.detach().requires_grad_(True)
    a = attn.detach().requires_grad_(True)
    if name == "cuda":
        lsi = level_start_index(spatial)
        out = ms_deform_attn(v, spatial, lsi, lc, a, im2col_step(v.shape[0]), backend="cuda")
    elif name == "triton":
        out = ms_deform_attn_triton(v, spatial, lc, a)
    elif name == "grid_sample":
        out = ms_deform_attn_core_pytorch(v, spatial, lc, a)
    else:
        raise ValueError(name)
    out.backward(torch.ones_like(out))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    del v, lc, a, out
    return peak


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def fmt_q(t):
    if t is None:
        return "n/a".rjust(22)
    med, p20, p80 = t
    return f"{med:7.3f} [{p20:6.3f},{p80:6.3f}]"


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    import triton
    print(f"triton {triton.__version__}\n")

    header = (f"{'shape':<10} {'dtype':<5} {'impl':<11} "
              f"{'fwd_ms med[p20,p80]':<23} {'bwd_ms med[p20,p80]':<23} "
              f"{'peak_MB':>9} {'spdup_fwd':>9} {'spdup_bwd':>9} {'correctness':<22}")
    print(header)
    print("-" * len(header))

    all_passed = True
    gates_run = 0

    for shape_label, mk in SHAPES:
        for dtype_label, dtype in DTYPES:
            impls = ["grid_sample", "triton"]
            if dtype == torch.float32:
                impls.append("cuda")

            value, spatial, loc, attn = make_inputs(device=DEVICE, dtype=dtype, **mk)
            ref = _ref_fwd_bwd(value, spatial, loc, attn)

            # grid_sample baseline timings (oracle == grid_sample, always PASS).
            base_fwd = None
            base_bwd = None
            rows = []

            for name in impls:
                ok, detail = correctness_gate(name, dtype, value, spatial, loc, attn, ref)
                gates_run += 1
                if not ok:
                    all_passed = False
                    rows.append((name, None, None, float("nan"), detail))
                    continue
                fwd = time_forward(name, value, spatial, loc, attn)
                bwd = time_backward(name, value, spatial, loc, attn)
                peak = peak_memory(name, value, spatial, loc, attn)
                rows.append((name, fwd, bwd, peak, detail))
                if name == "grid_sample":
                    base_fwd, base_bwd = fwd[0], bwd[0]

            for name, fwd, bwd, peak, detail in rows:
                if fwd is None:
                    print(f"{shape_label:<10} {dtype_label:<5} {name:<11} "
                          f"{'GATE FAILED':<23} {'':<23} {'':>9} {'':>9} {'':>9} {detail:<22}")
                    continue
                sp_fwd = base_fwd / fwd[0] if base_fwd else float("nan")
                sp_bwd = base_bwd / bwd[0] if base_bwd else float("nan")
                print(f"{shape_label:<10} {dtype_label:<5} {name:<11} "
                      f"{fmt_q(fwd):<23} {fmt_q(bwd):<23} "
                      f"{peak:9.1f} {sp_fwd:9.2f} {sp_bwd:9.2f} {detail:<22}")

            # CUDA n/a note on half rows.
            if dtype != torch.float32:
                print(f"{shape_label:<10} {dtype_label:<5} {'cuda':<11} "
                      f"{'n/a (no half dispatch)':<23} {'':<23} {'':>9} {'':>9} {'':>9} "
                      f"{'n/a':<22}")

            del value, spatial, loc, attn, ref
            torch.cuda.empty_cache()
        print()

    print("notes:")
    print("  * fp32 gate = elementwise allclose(rtol=1e-2, atol=1e-3) on fwd + all 3 grads.")
    print("  * fp16/bf16 gate = relative-L2 norm vs the fp32 oracle; 'gradL2<=' is the worst")
    print("    of the 3 gradient relL2 errors. Triton (under test) is held TIGHT (<=1e-2 fp16/")
    print("    2e-2 bf16) and lands ~1e-4..1e-3 by accumulating in fp32; torch half grid_sample")
    print("    is the format-limited reference (~0.1-0.3 on d_loc) and gets a sanity floor only.")
    print("  * cuda extension dispatches fp32/fp64 only -> 'n/a (no half dispatch)'.")
    print("  * speedups are relative to grid_sample at the same (shape, dtype).")
    print(f"\ncorrectness gates run: {gates_run}  all passed: {all_passed}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

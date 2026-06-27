"""Triton port of Multi-Scale Deformable Attention (forward + backward).

This implements both the FORWARD and BACKWARD passes of MSDeformAttn as Triton
kernels plus a ``torch.autograd.Function`` wrapper. The numerics match
``reference.ms_deform_attn_core_pytorch`` (which uses
``F.grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False)`` with
``grid = 2*loc - 1``).

Shapes (N batch, S=sum_l H_l*W_l, M heads, D head_dim, Lq queries, L levels, P points):
    value               (N, S, M, D)
    value_spatial_shapes(L, 2)            int64 [(H_0,W_0), ...]
    sampling_locations  (N, Lq, M, L, P, 2)  in [0,1], (x, y) order
    attention_weights   (N, Lq, M, L, P)
    returns             (N, Lq, M*D)

Accumulation policy (PORT_SPEC.md invariant #4): accumulate in fp32 for fp16 /
bf16 / fp32 inputs, casting to the input dtype only on the final store. For fp64
inputs the kernels accumulate in fp64 so that ``torch.autograd.gradcheck`` (which
requires double precision and amplifies forward rounding noise via finite
differences) passes against the Triton kernel itself. The ``ACCUM_FP64``
constexpr selects the compute dtype; it never drops below fp32.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    value_ptr,            # (N, S, M, D)
    shapes_ptr,           # (L, 2) int64, rows (H_l, W_l)
    lsi_ptr,              # (L,) int, level_start_index
    loc_ptr,              # (N, Lq, M, L, P, 2)
    attn_ptr,             # (N, Lq, M, L, P)
    out_ptr,              # (N, Lq, M*D)
    N, Lq, M, D, S,       # runtime scalars
    sN, sS, sM, sD,       # value strides (elements)
    L: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACCUM_FP64: tl.constexpr,
):
    cdt: tl.constexpr = tl.float64 if ACCUM_FP64 else tl.float32

    pid = tl.program_id(0)
    m = pid % M
    q = (pid // M) % Lq
    n = pid // (M * Lq)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    acc = tl.zeros([BLOCK_D], dtype=cdt)
    base = value_ptr + n * sN + m * sM + d * sD   # [BLOCK_D] base for this (n, m)

    nqm = (n * Lq + q) * M + m

    for lvl in range(L):
        H = tl.load(shapes_ptr + 2 * lvl)
        W = tl.load(shapes_ptr + 2 * lvl + 1)
        lsi = tl.load(lsi_ptr + lvl)
        Wf = W.to(cdt)
        Hf = H.to(cdt)
        for p in range(P):
            loc_off = ((nqm * L + lvl) * P + p) * 2
            attn_off = (nqm * L + lvl) * P + p
            loc_x = tl.load(loc_ptr + loc_off + 0).to(cdt)
            loc_y = tl.load(loc_ptr + loc_off + 1).to(cdt)
            aw = tl.load(attn_ptr + attn_off).to(cdt)

            w_im = loc_x * Wf - 0.5     # x <-> W
            h_im = loc_y * Hf - 0.5     # y <-> H

            w_low = tl.floor(w_im).to(tl.int32)   # floor BEFORE int-cast (negatives!)
            h_low = tl.floor(h_im).to(tl.int32)
            w_high = w_low + 1
            h_high = h_low + 1

            lw = w_im - w_low.to(cdt)
            lh = h_im - h_low.to(cdt)
            hw = 1.0 - lw
            hh = 1.0 - lh

            w1 = hh * hw   # NW (h_low , w_low )
            w2 = hh * lw   # NE (h_low , w_high)
            w3 = lh * hw   # SW (h_high, w_low )
            w4 = lh * lw   # SE (h_high, w_high)

            # Full two-sided bounds per corner. CUDA gets away with one-sided
            # checks only because of its outer validity gate
            # (h_im>-1 && h_im<H && w_im>-1 && w_im<W); this port dropped that
            # gate, so each corner must bound BOTH sides or sampling_locations
            # outside [0,1] (offsets are unbounded) index past the level -> an
            # illegal read (fwd) / atomic_add corruption (bwd). Equivalent to
            # CUDA's gate + grid_sample(padding_mode='zeros').
            m1 = (h_low >= 0) & (h_low <= H - 1) & (w_low >= 0) & (w_low <= W - 1)
            m2 = (h_low >= 0) & (h_low <= H - 1) & (w_high >= 0) & (w_high <= W - 1)
            m3 = (h_high >= 0) & (h_high <= H - 1) & (w_low >= 0) & (w_low <= W - 1)
            m4 = (h_high >= 0) & (h_high <= H - 1) & (w_high >= 0) & (w_high <= W - 1)

            off1 = (lsi + h_low * W + w_low) * sS
            off2 = (lsi + h_low * W + w_high) * sS
            off3 = (lsi + h_high * W + w_low) * sS
            off4 = (lsi + h_high * W + w_high) * sS

            v1 = tl.load(base + off1, mask=m1 & d_mask, other=0.0).to(cdt)
            v2 = tl.load(base + off2, mask=m2 & d_mask, other=0.0).to(cdt)
            v3 = tl.load(base + off3, mask=m3 & d_mask, other=0.0).to(cdt)
            v4 = tl.load(base + off4, mask=m4 & d_mask, other=0.0).to(cdt)

            acc += aw * (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)

    out_off = (n * Lq + q) * (M * D) + m * D + d
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def _bwd_kernel(
    value_ptr,            # (N, S, M, D)
    shapes_ptr,           # (L, 2) int64
    lsi_ptr,              # (L,) int
    loc_ptr,              # (N, Lq, M, L, P, 2)
    attn_ptr,             # (N, Lq, M, L, P)
    grad_out_ptr,         # (N, Lq, M*D)
    grad_value_ptr,       # (N, S, M, D)  -- accumulated via atomic_add
    grad_loc_ptr,         # (N, Lq, M, L, P, 2)
    grad_attn_ptr,        # (N, Lq, M, L, P)
    N, Lq, M, D, S,       # runtime scalars
    sN, sS, sM, sD,       # value strides (elements)
    gvN, gvS, gvM, gvD,   # grad_value strides (elements)
    L: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACCUM_FP64: tl.constexpr,
):
    cdt: tl.constexpr = tl.float64 if ACCUM_FP64 else tl.float32

    pid = tl.program_id(0)
    m = pid % M
    q = (pid // M) % Lq
    n = pid // (M * Lq)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    base = value_ptr + n * sN + m * sM + d * sD          # value read base
    base_gv = grad_value_ptr + n * gvN + m * gvM + d * gvD  # grad_value scatter base

    nqm = (n * Lq + q) * M + m

    # grad_output[n, q, m, :] is owned uniquely by this program -- load once.
    out_off = (n * Lq + q) * (M * D) + m * D + d
    g = tl.load(grad_out_ptr + out_off, mask=d_mask, other=0.0).to(cdt)

    for lvl in range(L):
        H = tl.load(shapes_ptr + 2 * lvl)
        W = tl.load(shapes_ptr + 2 * lvl + 1)
        lsi = tl.load(lsi_ptr + lvl)
        Wf = W.to(cdt)
        Hf = H.to(cdt)
        for p in range(P):
            loc_off = ((nqm * L + lvl) * P + p) * 2
            attn_off = (nqm * L + lvl) * P + p
            loc_x = tl.load(loc_ptr + loc_off + 0).to(cdt)
            loc_y = tl.load(loc_ptr + loc_off + 1).to(cdt)
            aw = tl.load(attn_ptr + attn_off).to(cdt)

            w_im = loc_x * Wf - 0.5
            h_im = loc_y * Hf - 0.5

            w_low = tl.floor(w_im).to(tl.int32)
            h_low = tl.floor(h_im).to(tl.int32)
            w_high = w_low + 1
            h_high = h_low + 1

            lw = w_im - w_low.to(cdt)
            lh = h_im - h_low.to(cdt)
            hw = 1.0 - lw
            hh = 1.0 - lh

            w1 = hh * hw   # NW
            w2 = hh * lw   # NE
            w3 = lh * hw   # SW
            w4 = lh * lw   # SE

            # Full two-sided bounds per corner. CUDA gets away with one-sided
            # checks only because of its outer validity gate
            # (h_im>-1 && h_im<H && w_im>-1 && w_im<W); this port dropped that
            # gate, so each corner must bound BOTH sides or sampling_locations
            # outside [0,1] (offsets are unbounded) index past the level -> an
            # illegal read (fwd) / atomic_add corruption (bwd). Equivalent to
            # CUDA's gate + grid_sample(padding_mode='zeros').
            m1 = (h_low >= 0) & (h_low <= H - 1) & (w_low >= 0) & (w_low <= W - 1)
            m2 = (h_low >= 0) & (h_low <= H - 1) & (w_high >= 0) & (w_high <= W - 1)
            m3 = (h_high >= 0) & (h_high <= H - 1) & (w_low >= 0) & (w_low <= W - 1)
            m4 = (h_high >= 0) & (h_high <= H - 1) & (w_high >= 0) & (w_high <= W - 1)

            off1 = (lsi + h_low * W + w_low) * sS
            off2 = (lsi + h_low * W + w_high) * sS
            off3 = (lsi + h_high * W + w_low) * sS
            off4 = (lsi + h_high * W + w_high) * sS

            v1 = tl.load(base + off1, mask=m1 & d_mask, other=0.0).to(cdt)
            v2 = tl.load(base + off2, mask=m2 & d_mask, other=0.0).to(cdt)
            v3 = tl.load(base + off3, mask=m3 & d_mask, other=0.0).to(cdt)
            v4 = tl.load(base + off4, mask=m4 & d_mask, other=0.0).to(cdt)

            # (a) grad_attn_weight[n,q,m,l,p] = sum_d g(d) * sample(d)
            sample = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
            gaw = tl.sum(tl.where(d_mask, g * sample, 0.0))
            tl.store(grad_attn_ptr + attn_off, gaw)

            # (b) grad_value scatter -- the only contended output (atomic_add)
            gv = aw * g                                  # [BLOCK_D] top_grad_value
            off1_gv = (lsi + h_low * W + w_low) * gvS
            off2_gv = (lsi + h_low * W + w_high) * gvS
            off3_gv = (lsi + h_high * W + w_low) * gvS
            off4_gv = (lsi + h_high * W + w_high) * gvS
            tl.atomic_add(base_gv + off1_gv, w1 * gv, mask=m1 & d_mask)
            tl.atomic_add(base_gv + off2_gv, w2 * gv, mask=m2 & d_mask)
            tl.atomic_add(base_gv + off3_gv, w3 * gv, mask=m3 & d_mask)
            tl.atomic_add(base_gv + off4_gv, w4 * gv, mask=m4 & d_mask)

            # (c) grad_sampling_loc[...,{0,1}] -- reduce over D, plain store
            grad_w = -hh * v1 + hh * v2 - lh * v3 + lh * v4   # d sample / d w_im (x)
            grad_h = -hw * v1 - lw * v2 + hw * v3 + lw * v4   # d sample / d h_im (y)
            d_loc_x = Wf * tl.sum(tl.where(d_mask, gv * grad_w, 0.0))  # W <-> x
            d_loc_y = Hf * tl.sum(tl.where(d_mask, gv * grad_h, 0.0))  # H <-> y
            tl.store(grad_loc_ptr + loc_off + 0, d_loc_x)
            tl.store(grad_loc_ptr + loc_off + 1, d_loc_y)


def _make_level_start_index(value_spatial_shapes):
    hw = value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]
    return torch.cat([value_spatial_shapes.new_zeros(1), hw.cumsum(0)[:-1]])


def _compute_dtype(value):
    """Compute/accumulate dtype: fp64 for fp64 inputs, else fp32 (never lower)."""
    if value.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _forward_launch(value, value_spatial_shapes, level_start_index,
                    sampling_locations, attention_weights):
    N, S, M, D = value.shape
    _, Lq, _, L, P, _ = sampling_locations.shape

    output = torch.empty(N, Lq, M * D, dtype=value.dtype, device=value.device)

    BLOCK_D = triton.next_power_of_2(D)
    sN, sS, sM, sD = value.stride()
    accum_fp64 = value.dtype == torch.float64

    grid = (N * Lq * M,)
    _fwd_kernel[grid](
        value, value_spatial_shapes, level_start_index,
        sampling_locations, attention_weights, output,
        N, Lq, M, D, S,
        sN, sS, sM, sD,
        L=L, P=P, BLOCK_D=BLOCK_D, ACCUM_FP64=accum_fp64,
        num_warps=4,
    )
    return output


def _backward_launch(value, value_spatial_shapes, level_start_index,
                     sampling_locations, attention_weights, grad_output):
    N, S, M, D = value.shape
    _, Lq, _, L, P, _ = sampling_locations.shape

    compute_dtype = _compute_dtype(value)
    accum_fp64 = compute_dtype == torch.float64

    # grad_value: zero-init in compute dtype (atomics accumulate onto contents).
    grad_value = torch.zeros(N, S, M, D, dtype=compute_dtype, device=value.device)
    # grad_loc / grad_attn: written once per slot; allocate in compute dtype.
    grad_loc = torch.empty(N, Lq, M, L, P, 2, dtype=compute_dtype, device=value.device)
    grad_attn = torch.empty(N, Lq, M, L, P, dtype=compute_dtype, device=value.device)

    BLOCK_D = triton.next_power_of_2(D)
    sN, sS, sM, sD = value.stride()
    gvN, gvS, gvM, gvD = grad_value.stride()

    grid = (N * Lq * M,)
    _bwd_kernel[grid](
        value, value_spatial_shapes, level_start_index,
        sampling_locations, attention_weights, grad_output,
        grad_value, grad_loc, grad_attn,
        N, Lq, M, D, S,
        sN, sS, sM, sD,
        gvN, gvS, gvM, gvD,
        L=L, P=P, BLOCK_D=BLOCK_D, ACCUM_FP64=accum_fp64,
        num_warps=4,
    )

    return (grad_value.to(value.dtype),
            grad_loc.to(sampling_locations.dtype),
            grad_attn.to(attention_weights.dtype))


class MSDeformAttnTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, sampling_locations, attention_weights):
        assert value.is_cuda, "Triton MSDeformAttn is CUDA-only"
        # Fail fast at the boundary: the kernel indexes value_spatial_shapes with
        # raw pointer arithmetic (shapes_ptr + 2*lvl), so a non-contiguous shapes
        # tensor reads garbage; and value length S must match the levels or the
        # per-level offsets run off the end of `value` (out-of-bounds).
        assert value_spatial_shapes.is_contiguous(), \
            "value_spatial_shapes must be contiguous"
        expected_S = int((value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum())
        assert value.shape[1] == expected_S, (
            f"value length S={value.shape[1]} must equal sum(H*W)={expected_S} over "
            f"levels {value_spatial_shapes.tolist()}")
        value = value.contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        level_start_index = _make_level_start_index(value_spatial_shapes)
        output = _forward_launch(
            value, value_spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
        )
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_value, grad_sampling_loc, grad_attn_weight = _backward_launch(
            value, value_spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output,
        )
        # 4-tuple aligned 1:1 with the 4 public forward params; None for int shapes.
        return grad_value, None, grad_sampling_loc, grad_attn_weight


def ms_deform_attn_triton(value, value_spatial_shapes,
                          sampling_locations, attention_weights):
    return MSDeformAttnTritonFunction.apply(
        value, value_spatial_shapes, sampling_locations, attention_weights)

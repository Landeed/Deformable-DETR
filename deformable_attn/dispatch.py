"""Backend dispatcher for Multi-Scale Deformable Attention.

Routes one call to a numerically-equivalent backend: ``"triton"`` (GPU,
fp16/bf16/fp32/fp64), ``"cuda"`` (compiled extension, fp32/fp64), or
``"pytorch"`` (grid_sample, CPU fallback). Precedence: ``backend=`` arg >
env ``DEFORM_ATTN_BACKEND`` > auto (CUDA -> triton, CPU -> pytorch). Backends
import lazily, so the compiled ``.so`` is only needed when ``"cuda"`` is used.
"""
import os

VALID_BACKENDS = ("triton", "cuda", "pytorch")

_CUDA_FN = None


def _cuda_function():
    """Lazily build an autograd wrapper around the compiled extension.

    Self-contained: imports the installed ``MultiScaleDeformableAttention`` op
    (built via ``models/ops/make.sh``) by name, so the package never depends on
    the research repo. Raises a clear error if the extension is not built.
    """
    global _CUDA_FN
    if _CUDA_FN is not None:
        return _CUDA_FN

    import torch
    from torch.autograd.function import once_differentiable
    try:
        import MultiScaleDeformableAttention as MSDA
    except ImportError as e:
        raise ImportError(
            "DEFORM_ATTN_BACKEND='cuda' needs the compiled MultiScaleDeformableAttention "
            "extension; build it with models/ops/make.sh, or use 'triton'/'pytorch'.") from e

    class _CudaMSDA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, spatial, lsi, loc, attn, im2col_step):
            ctx.im2col_step = im2col_step
            out = MSDA.ms_deform_attn_forward(value, spatial, lsi, loc, attn, im2col_step)
            ctx.save_for_backward(value, spatial, lsi, loc, attn)
            return out

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_out):
            value, spatial, lsi, loc, attn = ctx.saved_tensors
            gv, gloc, gattn = MSDA.ms_deform_attn_backward(
                value, spatial, lsi, loc, attn, grad_out.contiguous(), ctx.im2col_step)
            return gv, None, None, gloc, gattn, None

    _CUDA_FN = _CudaMSDA
    return _CUDA_FN


def _resolve_backend(value, backend):
    if backend is None:
        backend = os.environ.get("DEFORM_ATTN_BACKEND")
    if backend is None:
        backend = "triton" if value.is_cuda else "pytorch"
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"DEFORM_ATTN_BACKEND={backend!r} is invalid; expected one of {VALID_BACKENDS}")
    return backend


def ms_deform_attn(value, value_spatial_shapes, level_start_index,
                   sampling_locations, attention_weights, im2col_step=64,
                   backend=None):
    """Dispatch MSDeformAttn to the selected backend.

    Signature mirrors the compiled-op call site (``level_start_index`` and
    ``im2col_step`` are only used by the ``"cuda"`` backend; the Triton and
    PyTorch backends derive level offsets internally and ignore them).
    """
    backend = _resolve_backend(value, backend)

    if backend == "triton":
        from .ms_deform_attn_triton import ms_deform_attn_triton
        return ms_deform_attn_triton(
            value, value_spatial_shapes, sampling_locations, attention_weights)

    if backend == "pytorch":
        from .reference import ms_deform_attn_core_pytorch
        return ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights)

    # backend == "cuda" -- compiled extension (build via models/ops/make.sh)
    return _cuda_function().apply(
        value, value_spatial_shapes, level_start_index,
        sampling_locations, attention_weights, im2col_step)

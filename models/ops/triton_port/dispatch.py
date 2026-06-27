"""Backend dispatcher for Multi-Scale Deformable Attention.

Routes one call to a numerically-equivalent backend: ``"triton"`` (GPU,
fp16/bf16/fp32/fp64), ``"cuda"`` (compiled extension, fp32/fp64), or
``"pytorch"`` (grid_sample, CPU fallback). Precedence: ``backend=`` arg >
env ``DEFORM_ATTN_BACKEND`` > auto (CUDA -> triton, CPU -> pytorch). Backends
import lazily, so the compiled ``.so`` is only needed when ``"cuda"`` is used.
"""
import os

VALID_BACKENDS = ("triton", "cuda", "pytorch")


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

    # backend == "cuda" -- import the compiled extension lazily (needs the .so)
    from ..functions.ms_deform_attn_func import MSDeformAttnFunction
    return MSDeformAttnFunction.apply(
        value, value_spatial_shapes, level_start_index,
        sampling_locations, attention_weights, im2col_step)

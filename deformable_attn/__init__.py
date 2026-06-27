"""Triton port of Multi-Scale Deformable Attention.

- ``ms_deform_attn_triton`` -- the Triton op (forward + backward, autograd).
- ``ms_deform_attn`` -- backend dispatcher (triton / cuda / pytorch).
- ``ms_deform_attn_core_pytorch`` -- the grid_sample reference / oracle.
"""
from .dispatch import ms_deform_attn
from .ms_deform_attn_triton import ms_deform_attn_triton
from .reference import ms_deform_attn_core_pytorch

__all__ = ["ms_deform_attn", "ms_deform_attn_triton", "ms_deform_attn_core_pytorch"]

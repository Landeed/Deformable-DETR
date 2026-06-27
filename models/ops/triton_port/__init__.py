"""Triton port of Multi-Scale Deformable Attention. ``ms_deform_attn`` is the
backend-dispatching entry point used by ``modules.ms_deform_attn.MSDeformAttn``."""
from .dispatch import ms_deform_attn

__all__ = ["ms_deform_attn"]

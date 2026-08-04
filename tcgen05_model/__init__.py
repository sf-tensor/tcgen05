"""Bit-accurate software model of NVIDIA Blackwell tcgen05 MMA arithmetic."""

from .formats import bits_to_f32, f32_to_bits
from .model import (
    DotProduct,
    ScalarModel,
    Tcgen05BlockScaledMmaModel,
    Tcgen05I8S32MmaModel,
    Tcgen05MixedRawWindowMmaModel,
    Tcgen05RawWindowNvfp4MmaModel,
    make_model,
    normalize_format,
)
from .simulator import mma, mma_dot, sparse_active_indices

__all__ = [
    "DotProduct",
    "ScalarModel",
    "Tcgen05BlockScaledMmaModel",
    "Tcgen05I8S32MmaModel",
    "Tcgen05MixedRawWindowMmaModel",
    "Tcgen05RawWindowNvfp4MmaModel",
    "bits_to_f32",
    "f32_to_bits",
    "make_model",
    "mma",
    "mma_dot",
    "normalize_format",
    "sparse_active_indices",
]

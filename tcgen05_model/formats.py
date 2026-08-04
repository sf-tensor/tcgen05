"""Raw-bit constants and container conversions used by the model."""

from __future__ import annotations

import struct

F32_SIGN_MASK = 0x8000_0000
F32_EXP_MASK = 0x7F80_0000
F32_FRAC_MASK = 0x007F_FFFF
F32_EXP_BIAS = 127
F32_POS_ZERO = 0x0000_0000
F32_POS_INF = 0x7F80_0000
F32_NEG_INF = 0xFF80_0000


def f32_to_bits(value: float) -> int:
    """Return the IEEE F32 word produced by converting a Python float."""

    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    """Interpret the low 32 bits of an integer as an IEEE F32 value."""

    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def tf32_truncate_bits(bits: int) -> int:
    """Keep the F32 sign, exponent, and top ten fraction bits."""

    return bits & 0xFFFF_E000


def bf16_truncate_bits(bits: int) -> int:
    """Keep the F32 sign, exponent, and top seven fraction bits."""

    return bits & 0xFFFF_0000

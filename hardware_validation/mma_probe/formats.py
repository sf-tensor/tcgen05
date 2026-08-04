from __future__ import annotations

import math
import struct
from fractions import Fraction


F32_SIGN_MASK = 0x8000_0000
F32_EXP_MASK = 0x7F80_0000
F32_FRAC_MASK = 0x007F_FFFF
F32_EXP_BIAS = 127
F32_POS_ZERO = 0x0000_0000
F32_NEG_ZERO = 0x8000_0000
F32_POS_ONE = 0x3F80_0000
F32_NEG_ONE = 0xBF80_0000
F32_POS_INF = 0x7F80_0000
F32_NEG_INF = 0xFF80_0000
F32_CANONICAL_NAN = 0x7FC0_0000


def f32_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def f32_pow2_bits(exp: int) -> int:
    return f32_to_bits(math.ldexp(1.0, exp))


def f32_neg(bits: int) -> int:
    return bits ^ F32_SIGN_MASK


def f32_is_nan(bits: int) -> bool:
    return (bits & F32_EXP_MASK) == F32_EXP_MASK and (bits & F32_FRAC_MASK) != 0


def f32_is_inf(bits: int) -> bool:
    return (bits & 0x7FFF_FFFF) == F32_POS_INF


def f32_is_zero(bits: int) -> bool:
    return (bits & 0x7FFF_FFFF) == 0


def f32_class(bits: int) -> str:
    bits &= 0xFFFF_FFFF
    if f32_is_nan(bits):
        return "nan"
    if bits == F32_POS_INF:
        return "+inf"
    if bits == F32_NEG_INF:
        return "-inf"
    if bits == F32_POS_ZERO:
        return "+0"
    if bits == F32_NEG_ZERO:
        return "-0"
    return f"{bits_to_f32(bits):.9g}"


def tf32_truncate_bits(bits: int) -> int:
    """Apply the usual TF32 container truncation: keep sign, exponent, top 10 fraction bits."""
    return bits & 0xFFFF_E000


def bf16_truncate_bits(bits: int) -> int:
    """Apply BF16 container truncation: keep sign, exponent, and top 7 fraction bits."""
    return bits & 0xFFFF_0000


def f32_to_fraction(bits: int, *, tf32_truncate: bool = False, bf16_truncate: bool = False) -> Fraction | str:
    """Decode finite f32/TF32/BF16-container bits exactly.

    Returns "+inf", "-inf", or "nan" for non-finite values.
    """
    if tf32_truncate and bf16_truncate:
        raise ValueError("cannot request both TF32 and BF16 truncation")
    if tf32_truncate:
        bits = tf32_truncate_bits(bits)
    if bf16_truncate:
        bits = bf16_truncate_bits(bits)
    bits &= 0xFFFF_FFFF
    sign = -1 if (bits & F32_SIGN_MASK) else 1
    exp = (bits & F32_EXP_MASK) >> 23
    frac = bits & F32_FRAC_MASK
    if exp == 0xFF:
        if frac:
            return "nan"
        return "-inf" if sign < 0 else "+inf"
    if exp == 0:
        if frac == 0:
            return Fraction(0)
        mant = frac
        exponent = 1 - F32_EXP_BIAS - 23
    else:
        mant = (1 << 23) | frac
        exponent = exp - F32_EXP_BIAS - 23
    value = Fraction(mant)
    if exponent >= 0:
        value *= 1 << exponent
    else:
        value /= 1 << (-exponent)
    return value if sign > 0 else -value


def _round_fraction_to_int(value: Fraction, mode: str) -> int:
    floor = value.numerator // value.denominator
    rem = value.numerator - floor * value.denominator
    if rem == 0:
        return floor
    if mode == "rz":
        return floor
    if mode == "ru":
        return floor + 1
    if mode == "rne":
        twice = rem * 2
        if twice < value.denominator:
            return floor
        if twice > value.denominator:
            return floor + 1
        return floor if (floor & 1) == 0 else floor + 1
    raise ValueError(f"unsupported rounding mode: {mode}")


def fraction_to_f32_bits(value: Fraction, *, mode: str = "rne") -> int:
    """Round an exact finite value to IEEE float32 bits.

    This is intentionally small but exact enough for model validation over finite values.
    """
    if value == 0:
        return F32_POS_ZERO
    sign_bit = 0
    if value < 0:
        sign_bit = F32_SIGN_MASK
        value = -value

    # Find unbiased exponent e such that 1 <= value / 2**e < 2.
    nbits = value.numerator.bit_length()
    dbits = value.denominator.bit_length()
    e = nbits - dbits
    if value < Fraction(1 << max(e, 0), 1 << max(-e, 0)):
        e -= 1

    if e > 127:
        return sign_bit | F32_POS_INF

    if e >= -126:
        scaled = value * Fraction(1 << 23, 1)
        if e >= 0:
            scaled /= 1 << e
        else:
            scaled *= 1 << (-e)
        mant = _round_fraction_to_int(scaled, mode)
        if mant == (1 << 24):
            e += 1
            mant >>= 1
            if e > 127:
                return sign_bit | F32_POS_INF
        exp_bits = (e + F32_EXP_BIAS) << 23
        frac_bits = mant & F32_FRAC_MASK
        return sign_bit | exp_bits | frac_bits

    # Subnormal: value = mantissa * 2**-149.
    scaled = value * (1 << 149)
    mant = _round_fraction_to_int(scaled, mode)
    if mant == 0:
        return sign_bit
    if mant >= (1 << 23):
        return sign_bit | (1 << 23)
    return sign_bit | mant

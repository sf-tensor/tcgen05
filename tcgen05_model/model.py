"""Bit-accurate scalar arithmetic for NVIDIA Blackwell tcgen05 MMA."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from .formats import (
    F32_EXP_BIAS,
    F32_EXP_MASK,
    F32_FRAC_MASK,
    F32_NEG_INF,
    F32_POS_INF,
    F32_POS_ZERO,
    F32_SIGN_MASK,
    bf16_truncate_bits,
    tf32_truncate_bits,
)


@dataclass(frozen=True)
class DotProduct:
    """Raw operand words for one output element of a tcgen05 MMA."""

    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int = 0
    scale_a: tuple[int, ...] | None = None
    scale_b: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.a) != len(self.b):
            raise ValueError("A and B dot-product vectors must have equal length")
        if not self.a:
            raise ValueError("a dot product must contain at least one element")
        for value in (*self.a, *self.b, self.c):
            if not isinstance(value, int) or not 0 <= value <= 0xFFFF_FFFF:
                raise ValueError(f"operand is not a uint32 word: {value!r}")
        if (self.scale_a is None) != (self.scale_b is None):
            raise ValueError("scale_a and scale_b must be supplied together")
        for value in (*(self.scale_a or ()), *(self.scale_b or ())):
            if not isinstance(value, int) or not 0 <= value <= 0xFF:
                raise ValueError(f"scale is not an 8-bit word: {value!r}")


class ScalarModel(Protocol):
    """Interface implemented by every scalar tcgen05 arithmetic model."""

    def eval(self, case: DotProduct) -> int: ...

    def matches(self, case: DotProduct, observed: int) -> bool: ...


class _ModelBase:
    canonical_nan = 0x7FFF_FFFF

    def matches(self, case: DotProduct, observed: int) -> bool:
        return self.eval(case) == (observed & 0xFFFF_FFFF)


def _special_sum(values: list[str]) -> str | None:
    if "nan" in values or ("+inf" in values and "-inf" in values):
        return "nan"
    if "+inf" in values:
        return "+inf"
    if "-inf" in values:
        return "-inf"
    return None


def _decode_tf32_int(bits: int) -> tuple[int, int, int] | str:
    bits = tf32_truncate_bits(bits)
    sign = -1 if bits & F32_SIGN_MASK else 1
    exp = (bits & F32_EXP_MASK) >> 23
    frac = bits & F32_FRAC_MASK
    if exp == 0xFF:
        if frac:
            return "nan"
        return "-inf" if sign < 0 else "+inf"
    if exp == 0:
        mant = frac >> 13
        raw_exp = 1 - F32_EXP_BIAS
    else:
        mant = (1 << 10) | (frac >> 13)
        raw_exp = exp - F32_EXP_BIAS
    return sign * mant, raw_exp - 10, raw_exp


def _decode_bf16_int(bits: int) -> tuple[int, int, int] | str:
    bits = bf16_truncate_bits(bits)
    sign = -1 if bits & F32_SIGN_MASK else 1
    exp = (bits & F32_EXP_MASK) >> 23
    frac = bits & F32_FRAC_MASK
    if exp == 0xFF:
        if frac:
            return "nan"
        return "-inf" if sign < 0 else "+inf"
    if exp == 0:
        mant = frac >> 16
        raw_exp = 1 - F32_EXP_BIAS
    else:
        mant = (1 << 7) | (frac >> 16)
        raw_exp = exp - F32_EXP_BIAS
    return sign * mant, raw_exp - 7, raw_exp


def _decode_f16_int(bits: int) -> tuple[int, int, int] | str:
    bits &= 0xFFFF
    sign = -1 if bits & 0x8000 else 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0x1F:
        if frac:
            return "nan"
        return "-inf" if sign < 0 else "+inf"
    if exp == 0:
        mant = frac
        raw_exp = 1 - 15
    else:
        mant = (1 << 10) | frac
        raw_exp = exp - 15
    return sign * mant, raw_exp - 10, raw_exp


def _decode_fp8e4m3_int(bits: int) -> tuple[int, int, int] | str:
    bits &= 0xFF
    sign = -1 if bits & 0x80 else 1
    exp = (bits >> 3) & 0xF
    frac = bits & 0x7
    if exp == 0xF and frac == 0x7:
        return "nan"
    if exp == 0:
        mant = frac
        raw_exp = 1 - 7
    else:
        mant = (1 << 3) | frac
        raw_exp = exp - 7
    return sign * mant, raw_exp - 3, raw_exp


def _decode_fp8e5m2_int(bits: int) -> tuple[int, int, int] | str:
    bits &= 0xFF
    sign = -1 if bits & 0x80 else 1
    exp = (bits >> 2) & 0x1F
    frac = bits & 0x3
    if exp == 0x1F:
        if frac:
            return "nan"
        return "-inf" if sign < 0 else "+inf"
    if exp == 0:
        mant = frac
        raw_exp = 1 - 15
    else:
        mant = (1 << 2) | frac
        raw_exp = exp - 15
    return sign * mant, raw_exp - 2, raw_exp


def _decode_fp6e2m3_int(bits: int) -> tuple[int, int, int]:
    bits &= 0x3F
    sign = -1 if bits & 0x20 else 1
    exp = (bits >> 3) & 0x3
    frac = bits & 0x7
    if exp == 0:
        mant = frac
        raw_exp = 1 - 1
    else:
        mant = (1 << 3) | frac
        raw_exp = exp - 1
    return sign * mant, raw_exp - 3, raw_exp


def _decode_fp6e3m2_int(bits: int) -> tuple[int, int, int]:
    bits &= 0x3F
    sign = -1 if bits & 0x20 else 1
    exp = (bits >> 2) & 0x7
    frac = bits & 0x3
    if exp == 0:
        mant = frac
        raw_exp = 1 - 3
    else:
        mant = (1 << 2) | frac
        raw_exp = exp - 3
    return sign * mant, raw_exp - 2, raw_exp


def _decode_ue4m3_int(bits: int) -> tuple[int, int, int] | str:
    bits &= 0x7F
    exp = (bits >> 3) & 0xF
    frac = bits & 0x7
    if exp == 0xF and frac == 0x7:
        return "nan"
    if exp == 0:
        mant = frac
        raw_exp = 1 - 7
    else:
        mant = (1 << 3) | frac
        raw_exp = exp - 7
    return mant, raw_exp - 3, raw_exp


def _decode_ue8m0_int(bits: int) -> tuple[int, int, int] | str:
    bits &= 0xFF
    if bits == 0xFF:
        return "nan"
    raw_exp = bits - 127
    return 1, raw_exp, raw_exp


def _decode_fp4e2m1_int(bits: int) -> tuple[int, int, int]:
    bits &= 0xF
    sign = -1 if bits & 0x8 else 1
    exp = (bits >> 1) & 0x3
    frac = bits & 0x1
    if exp == 0:
        mant = frac
        raw_exp = 1 - 1
    else:
        mant = (1 << 1) | frac
        raw_exp = exp - 1
    return sign * mant, raw_exp - 1, raw_exp


def _decode_f32_int(bits: int) -> tuple[int, int, int] | str:
    sign = -1 if bits & F32_SIGN_MASK else 1
    exp = (bits & F32_EXP_MASK) >> 23
    frac = bits & F32_FRAC_MASK
    if exp == 0xFF:
        if frac:
            return "nan"
        return "-inf" if sign < 0 else "+inf"
    if exp == 0:
        mant = frac
        raw_exp = 1 - F32_EXP_BIAS
    else:
        mant = (1 << 23) | frac
        raw_exp = exp - F32_EXP_BIAS
    return sign * mant, raw_exp - 23, raw_exp


def _trunc_int_to_quantum(mant: int, value_exp: int, quantum_exp: int) -> int:
    if mant == 0:
        return 0
    shift = value_exp - quantum_exp
    if shift >= 0:
        return mant << shift
    sign = -1 if mant < 0 else 1
    return sign * (abs(mant) >> (-shift))


def _trunc_quarter_units_to_quantum(units: int, quantum_exp: int) -> int:
    """Return signed units of 2**quantum_exp from an exact value of units * 2**-2."""
    if units == 0:
        return 0
    shift = quantum_exp + 2
    if shift <= 0:
        return units << (-shift)
    sign = -1 if units < 0 else 1
    return sign * (abs(units) >> shift)


def _sum_int_terms_to_common_exp(terms: list[tuple[int, int]]) -> tuple[int, int]:
    nonzero = [(mant, exp) for mant, exp in terms if mant != 0]
    if not nonzero:
        return 0, 0
    common_exp = min(exp for _mant, exp in nonzero)
    units = sum(mant << (exp - common_exp) for mant, exp in nonzero)
    return units, common_exp


def _round_int_to_f32_rz(mant: int, value_exp: int) -> int:
    if mant == 0:
        return F32_POS_ZERO

    sign_bit = 0
    if mant < 0:
        sign_bit = F32_SIGN_MASK
        mant = -mant

    exp = mant.bit_length() - 1 + value_exp
    if exp > 127:
        return sign_bit | F32_POS_INF

    if exp >= -126:
        shift = value_exp - exp + 23
        if shift >= 0:
            rounded_mant = mant << shift
        else:
            rounded_mant = mant >> (-shift)
        if rounded_mant >= (1 << 24):
            exp += 1
            rounded_mant >>= 1
            if exp > 127:
                return sign_bit | F32_POS_INF
        return sign_bit | ((exp + F32_EXP_BIAS) << 23) | (rounded_mant & F32_FRAC_MASK)

    shift = value_exp + 149
    if shift >= 0:
        rounded_mant = mant << shift
    else:
        rounded_mant = mant >> (-shift)
    if rounded_mant == 0:
        return F32_POS_ZERO
    if rounded_mant >= (1 << 23):
        return sign_bit | (1 << 23)
    return sign_bit | rounded_mant


def _round_shift_positive(value: int, shift: int, mode: str) -> int:
    if shift <= 0:
        return value << (-shift)
    q = value >> shift
    if mode == "rz":
        return q
    rem = value & ((1 << shift) - 1)
    twice = rem << 1
    threshold = 1 << shift
    if twice > threshold or (twice == threshold and (q & 1)):
        return q + 1
    return q


def _round_int_to_f16(mant: int, value_exp: int, *, mode: str = "rz") -> int:
    if mant == 0:
        return 0

    sign_bit = 0
    if mant < 0:
        sign_bit = 0x8000
        mant = -mant

    exp = mant.bit_length() - 1 + value_exp
    if exp > 15:
        return sign_bit | 0x7C00

    if exp >= -14:
        shift = value_exp - exp + 10
        rounded_mant = _round_shift_positive(mant, -shift, mode)
        if rounded_mant >= (1 << 11):
            exp += 1
            rounded_mant >>= 1
            if exp > 15:
                return sign_bit | 0x7C00
        return sign_bit | ((exp + 15) << 10) | (rounded_mant & 0x03FF)

    shift = value_exp + 24
    rounded_mant = _round_shift_positive(mant, -shift, mode)
    if rounded_mant == 0:
        return 0
    if rounded_mant >= (1 << 10):
        return sign_bit | (1 << 10)
    return sign_bit | rounded_mant


class _Tcgen05RawWindowMmaModel(_ModelBase):
    input_format = "unknown"
    product_window_fraction_bits = 25

    def __init__(self):
        # Generated NaNs use the exact tcgen05 word rather than an arbitrary
        # IEEE NaN payload.
        super().__init__()

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        raise NotImplementedError

    def _decode_a_input(self, bits: int) -> tuple[int, int, int] | str:
        return self._decode_input(bits)

    def _decode_b_input(self, bits: int) -> tuple[int, int, int] | str:
        return self._decode_input(bits)

    def _decode_c_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_f32_int(bits)

    def _round_output(self, units: int, quantum_exp: int) -> int:
        return _round_int_to_f32_rz(units, quantum_exp)

    def eval(self, case: DotProduct) -> int:
        products: list[tuple[int, int]] = []
        raw_exponents: list[int] = []
        specials: list[str] = []

        for a_bits, b_bits in zip(case.a, case.b):
            a = self._decode_a_input(a_bits)
            b = self._decode_b_input(b_bits)
            if a == "nan" or b == "nan":
                return self.canonical_nan
            if a in ("+inf", "-inf") or b in ("+inf", "-inf"):
                a_is_zero = isinstance(a, tuple) and a[0] == 0
                b_is_zero = isinstance(b, tuple) and b[0] == 0
                if a_is_zero or b_is_zero:
                    return self.canonical_nan
                sign = 1
                if a == "-inf":
                    sign *= -1
                if b == "-inf":
                    sign *= -1
                if isinstance(a, tuple) and a[0] < 0:
                    sign *= -1
                if isinstance(b, tuple) and b[0] < 0:
                    sign *= -1
                specials.append("-inf" if sign < 0 else "+inf")
                continue

            assert isinstance(a, tuple) and isinstance(b, tuple)
            product_mant = a[0] * b[0]
            product_exp = a[1] + b[1]
            products.append((product_mant, product_exp))
            if product_mant != 0:
                raw_exponents.append(a[2] + b[2])

        c = self._decode_c_input(case.c)
        if c == "nan":
            return self.canonical_nan
        if c in ("+inf", "-inf"):
            specials.append(c)

        special = _special_sum(specials)
        if special == "nan":
            return self.canonical_nan
        if special == "+inf":
            return F32_POS_INF
        if special == "-inf":
            return F32_NEG_INF

        finite_terms = products[:]
        if isinstance(c, tuple):
            finite_terms.append((c[0], c[1]))
            if c[0] != 0:
                raw_exponents.append(c[2])

        if not raw_exponents:
            return F32_POS_ZERO

        quantum_exp = max(raw_exponents) - self.product_window_fraction_bits
        units = sum(_trunc_int_to_quantum(mant, exp, quantum_exp) for mant, exp in finite_terms)
        return self._round_output(units, quantum_exp)


class Tcgen05RawWindowTf32MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for the tcgen05 TF32 finite datapath.

    Measured rule for finite inputs:
    - A/B are truncated to TF32 before classification.
    - Each product contributes to one shared raw-exponent window.
    - The window exponent is max(product operand exponent sums, C exponent) minus
      25 fractional bits.
    - Every product and C is shifted directly into that window with truncation
      toward zero, then the signed integer terms are summed.
    - The final finite result is converted to float32 by truncating magnitude.
    """

    input_format = "tf32"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_tf32_int(bits)


class Tcgen05RawWindowBf16MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for the tcgen05 BF16 finite datapath.

    This mirrors the TF32 rule while changing only the A/B decode to BF16
    container bits.
    """

    input_format = "bf16"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_bf16_int(bits)


class Tcgen05RawWindowF16MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for tcgen05 FP16 x FP16 -> F32.

    This uses the same raw-window accumulator rule as the TF32/BF16/FP8
    paths and changes only the A/B decode to IEEE FP16 container bits.
    """

    input_format = "f16"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_f16_int(bits)


class Tcgen05RawWindowFp8E4M3MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for tcgen05 FP8 E4M3 x E4M3 -> F32.

    The finite datapath uses the same raw-window rule as TF32/BF16; the
    format-specific part is the E4M3 input decode.
    """

    input_format = "fp8e4m3"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_fp8e4m3_int(bits)


class Tcgen05RawWindowFp8E5M2MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for tcgen05 FP8 E5M2 x E5M2 -> F32."""

    input_format = "fp8e5m2"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_fp8e5m2_int(bits)


class Tcgen05RawWindowFp6E2M3MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for tcgen05 FP6 E2M3 x E2M3 -> F32."""

    input_format = "fp6e2m3"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_fp6e2m3_int(bits)


class Tcgen05RawWindowFp6E3M2MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for tcgen05 FP6 E3M2 x E3M2 -> F32."""

    input_format = "fp6e3m2"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_fp6e3m2_int(bits)


class Tcgen05RawWindowFp4E2M1MmaModel(_Tcgen05RawWindowMmaModel):
    """Bit model for unscaled tcgen05 FP4 E2M1 x E2M1 -> F32."""

    input_format = "fp4e2m1"

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_fp4e2m1_int(bits)


_F8F6F4_FORMATS = frozenset(("fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1"))


def _tcgen05_decoder_for_format(format_name: str):
    if format_name == "tf32":
        return _decode_tf32_int
    if format_name == "bf16":
        return _decode_bf16_int
    if format_name == "f16":
        return _decode_f16_int
    if format_name in ("fp8", "e4m3"):
        return _decode_fp8e4m3_int
    if format_name == "e5m2":
        return _decode_fp8e5m2_int
    if format_name == "e2m3":
        return _decode_fp6e2m3_int
    if format_name == "e3m2":
        return _decode_fp6e3m2_int
    if format_name == "e2m1":
        return _decode_fp4e2m1_int
    raise ValueError(f"unsupported tcgen05 input format: {format_name}")


class Tcgen05MixedRawWindowMmaModel(_Tcgen05RawWindowMmaModel):
    """Raw-window tcgen05 model with independently selected A/B decoders."""

    input_format = "mixed"

    def __init__(self, a_format: str, b_format: str, *, d_type: str = "f32"):
        super().__init__()
        self.a_format = a_format
        self.b_format = b_format
        self.d_type = d_type
        self._a_decoder = lru_cache(maxsize=None)(_tcgen05_decoder_for_format(a_format))
        self._b_decoder = lru_cache(maxsize=None)(_tcgen05_decoder_for_format(b_format))
        self._product_decoder = (
            lru_cache(maxsize=None)(self._decode_product_uncached)
            if a_format in _F8F6F4_FORMATS and b_format in _F8F6F4_FORMATS
            else None
        )

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        raise NotImplementedError("mixed model uses separate A/B decoders")

    def _decode_a_input(self, bits: int) -> tuple[int, int, int] | str:
        return self._a_decoder(bits)

    def _decode_b_input(self, bits: int) -> tuple[int, int, int] | str:
        return self._b_decoder(bits)

    def _decode_product_uncached(self, a_bits: int, b_bits: int):
        a = self._decode_a_input(a_bits)
        b = self._decode_b_input(b_bits)
        if a == "nan" or b == "nan":
            return ("nan",)
        if a in ("+inf", "-inf") or b in ("+inf", "-inf"):
            a_is_zero = isinstance(a, tuple) and a[0] == 0
            b_is_zero = isinstance(b, tuple) and b[0] == 0
            if a_is_zero or b_is_zero:
                return ("nan",)
            sign = 1
            if a == "-inf":
                sign *= -1
            if b == "-inf":
                sign *= -1
            if isinstance(a, tuple) and a[0] < 0:
                sign *= -1
            if isinstance(b, tuple) and b[0] < 0:
                sign *= -1
            return ("inf", sign)

        assert isinstance(a, tuple) and isinstance(b, tuple)
        product_mant = a[0] * b[0]
        product_exp = a[1] + b[1]
        product_raw = a[2] + b[2] if product_mant != 0 else None
        return ("finite", product_mant, product_exp, product_raw)

    def _decode_c_input(self, bits: int) -> tuple[int, int, int] | str:
        if self.d_type == "f16":
            return _decode_f16_int(bits)
        return _decode_f32_int(bits)

    def _round_output(self, units: int, quantum_exp: int) -> int:
        if self.d_type == "f16":
            return _round_int_to_f16(units, quantum_exp, mode="rne")
        return _round_int_to_f32_rz(units, quantum_exp)

    def _special_output(self, value: str) -> int:
        if self.d_type == "f16":
            return {
                "nan": 0x7FFF,
                "+inf": 0x7C00,
                "-inf": 0xFC00,
            }[value]
        return {
            "nan": self.canonical_nan,
            "+inf": F32_POS_INF,
            "-inf": F32_NEG_INF,
        }[value]

    def eval(self, case: DotProduct) -> int:
        if self._product_decoder is None:
            return super().eval(case)

        products: list[tuple[int, int]] = []
        raw_exponents: list[int] = []
        specials: list[str] = []

        for a_bits, b_bits in zip(case.a, case.b):
            product = self._product_decoder(a_bits, b_bits)
            tag = product[0]
            if tag == "nan":
                return self._special_output("nan")
            if tag == "inf":
                specials.append("-inf" if product[1] < 0 else "+inf")
                continue

            assert tag == "finite"
            product_mant = product[1]
            product_exp = product[2]
            product_raw = product[3]
            products.append((product_mant, product_exp))
            if product_raw is not None:
                raw_exponents.append(product_raw)

        c = self._decode_c_input(case.c)
        if c == "nan":
            return self._special_output("nan")
        if c in ("+inf", "-inf"):
            specials.append(c)

        special = _special_sum(specials)
        if special == "nan":
            return self._special_output("nan")
        if special == "+inf":
            return self._special_output("+inf")
        if special == "-inf":
            return self._special_output("-inf")

        finite_terms = products[:]
        if isinstance(c, tuple):
            finite_terms.append((c[0], c[1]))
            if c[0] != 0:
                raw_exponents.append(c[2])

        if not raw_exponents:
            return F32_POS_ZERO

        quantum_exp = max(raw_exponents) - self.product_window_fraction_bits
        units = sum(_trunc_int_to_quantum(mant, exp, quantum_exp) for mant, exp in finite_terms)
        return self._round_output(units, quantum_exp)


class Tcgen05RawWindowNvfp4MmaModel(_Tcgen05RawWindowMmaModel):
    """Known-good bit model for tcgen05 NVFP4 E2M1 inputs with UE4M3 scales.

    The product-dominant path reduces aligned K=4 groups into the 39-bit
    product window plus the scale_vec::4X rescale floor. Unit scales made that
    floor look like a fixed 2**-35 quantum, but arbitrary UE4M3 scales show it
    is actually anchored at
    max(scale_a_raw + scale_b_raw) - 35 for the nonzero product blocks. When
    input C's raw exponent reaches the product max exponent, the datapath
    exposes four K=16 product block sums; each block sum is truncated into a
    C-relative 35-bit window before the final signed integer sum and F32 RZ
    conversion.
    """

    input_format = "nvfp4"

    def __init__(self, *, c_merge_group_size: int = 16, scale_block_size: int = 16):
        super().__init__()
        self.c_merge_group_size = c_merge_group_size
        self.scale_block_size = scale_block_size

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return _decode_fp4e2m1_int(bits)

    def eval(self, case: DotProduct) -> int:
        products: list[tuple[int, int]] = []
        max_product_raw: int | None = None
        max_scale_raw: int | None = None
        block_count = (len(case.a) + self.scale_block_size - 1) // self.scale_block_size
        scale_a = case.scale_a or (0x38,) * block_count
        scale_b = case.scale_b or (0x38,) * block_count
        if len(scale_a) != len(scale_b):
            raise ValueError("NVFP4 A/B scale block counts differ")
        if len(scale_a) * self.scale_block_size < len(case.a):
            raise ValueError("NVFP4 case has more K values than scale blocks cover")
        merge_groups: list[list[tuple[int, int]]] = [
            [] for _ in range((len(case.a) + self.c_merge_group_size - 1) // self.c_merge_group_size)
        ]
        decoded_scale_a = [_decode_ue4m3_int(bits) for bits in scale_a]
        decoded_scale_b = [_decode_ue4m3_int(bits) for bits in scale_b]

        for k, (a_bits, b_bits) in enumerate(zip(case.a, case.b)):
            a = self._decode_input(a_bits)
            b = self._decode_input(b_bits)
            assert isinstance(a, tuple) and isinstance(b, tuple)
            block = k // self.scale_block_size
            sa = decoded_scale_a[block]
            sb = decoded_scale_b[block]
            if sa == "nan" or sb == "nan":
                return self.canonical_nan
            assert isinstance(sa, tuple) and isinstance(sb, tuple)
            product_mant = a[0] * b[0] * sa[0] * sb[0]
            product_exp = a[1] + b[1] + sa[1] + sb[1]
            product_raw = a[2] + b[2] + sa[2] + sb[2]
            scale_raw = sa[2] + sb[2]
            products.append((product_mant, product_exp))
            merge_groups[k // self.c_merge_group_size].append((product_mant, product_exp))
            if product_mant != 0:
                max_product_raw = (
                    product_raw if max_product_raw is None else max(max_product_raw, product_raw)
                )
                max_scale_raw = scale_raw if max_scale_raw is None else max(max_scale_raw, scale_raw)

        c = _decode_f32_int(case.c)
        if c == "nan":
            return self.canonical_nan
        if c == "+inf":
            return F32_POS_INF
        if c == "-inf":
            return F32_NEG_INF
        assert isinstance(c, tuple)
        if max_product_raw is None:
            return F32_POS_ZERO if (case.c & 0x7FFF_FFFF) == 0 else case.c

        raw_exponents: list[int] = []
        raw_exponents.append(max_product_raw)
        if c[0] != 0:
            raw_exponents.append(c[2])
        if not raw_exponents:
            return case.c if (case.c & 0x7FFF_FFFF) == 0 else F32_POS_ZERO

        assert max_scale_raw is not None

        if c[0] != 0 and c[2] >= max_product_raw:
            quantum_exp = c[2] - 35
            units = _trunc_int_to_quantum(c[0], c[1], quantum_exp)
            for group in merge_groups:
                block_mant, block_exp = _sum_int_terms_to_common_exp(group)
                units += _trunc_int_to_quantum(block_mant, block_exp, quantum_exp)
            return _round_int_to_f32_rz(units, quantum_exp)

        quantum_exp = max(max(raw_exponents) - 39, max_scale_raw - 35)
        units = _trunc_int_to_quantum(c[0], c[1], quantum_exp)
        # The product-dominant reduction preserves each aligned K=4 dot group
        # before quantizing it into the shared window.  Quantizing individual
        # products loses a carry on rare scale-floor boundary cases.
        for start in range(0, len(products), 4):
            group_mant, group_exp = _sum_int_terms_to_common_exp(products[start : start + 4])
            units += _trunc_int_to_quantum(group_mant, group_exp, quantum_exp)
        return _round_int_to_f32_rz(units, quantum_exp)


class Tcgen05BlockScaledMmaModel(_Tcgen05RawWindowMmaModel):
    """Parameterized model for tcgen05 block-scaled MX paths.

    This reuses the NVFP4 full-rescale rule with pluggable element and scale
    decoders. The arithmetic has the same integer fixed-point structure:
    scale-aware products, a product/C raw window, and a scale-exponent floor.
    """

    input_format = "block-scaled"

    def __init__(
        self,
        *,
        input_decoder,
        input_decoder_b=None,
        scale_decoder=_decode_ue8m0_int,
        scale_block_size: int,
        c_merge_group_size: int,
        product_window_fraction_bits: int = 39,
        scale_floor_offset: int = 35,
        product_merge_group_size: int | None = None,
        product_subnormal_group_floor_exp: int | None = None,
        c_merge_fraction_bits: int = 35,
        use_c_dominant_merge: bool = True,
        use_per_group_c_dominant_merge: bool = False,
    ):
        super().__init__()
        self.input_decoder = input_decoder
        self.input_decoder_b = input_decoder if input_decoder_b is None else input_decoder_b
        self.scale_decoder = scale_decoder
        self.scale_block_size = scale_block_size
        self.c_merge_group_size = c_merge_group_size
        self.product_window_fraction_bits = product_window_fraction_bits
        self.scale_floor_offset = scale_floor_offset
        self.product_merge_group_size = product_merge_group_size
        self.product_subnormal_group_floor_exp = product_subnormal_group_floor_exp
        self.c_merge_fraction_bits = c_merge_fraction_bits
        self.use_c_dominant_merge = use_c_dominant_merge
        self.use_per_group_c_dominant_merge = use_per_group_c_dominant_merge

    def _decode_input(self, bits: int) -> tuple[int, int, int] | str:
        return self.input_decoder(bits)

    def _decode_b_input(self, bits: int) -> tuple[int, int, int] | str:
        return self.input_decoder_b(bits)

    def eval(self, case: DotProduct) -> int:
        products: list[tuple[int, int]] = []
        product_scale_raws: list[int | None] = []
        product_raws: list[int | None] = []
        specials: list[str] = []
        max_product_raw: int | None = None
        max_scale_raw: int | None = None
        block_count = (len(case.a) + self.scale_block_size - 1) // self.scale_block_size
        scale_a = case.scale_a or (0x7F,) * block_count
        scale_b = case.scale_b or (0x7F,) * block_count
        if len(scale_a) != len(scale_b):
            raise ValueError("block-scaled A/B scale block counts differ")
        if len(scale_a) * self.scale_block_size < len(case.a):
            raise ValueError("case has more K values than scale blocks cover")

        merge_groups: list[list[tuple[int, int]]] = [
            [] for _ in range((len(case.a) + self.c_merge_group_size - 1) // self.c_merge_group_size)
        ]
        decoded_scale_a = [self.scale_decoder(bits) for bits in scale_a]
        decoded_scale_b = [self.scale_decoder(bits) for bits in scale_b]

        for k, (a_bits, b_bits) in enumerate(zip(case.a, case.b)):
            a = self._decode_input(a_bits)
            b = self._decode_b_input(b_bits)
            if a == "nan" or b == "nan":
                return self.canonical_nan

            block = k // self.scale_block_size
            sa = decoded_scale_a[block]
            sb = decoded_scale_b[block]
            if sa == "nan" or sb == "nan":
                return self.canonical_nan
            assert isinstance(sa, tuple) and isinstance(sb, tuple)

            if a in ("+inf", "-inf") or b in ("+inf", "-inf"):
                a_is_zero = isinstance(a, tuple) and a[0] == 0
                b_is_zero = isinstance(b, tuple) and b[0] == 0
                if a_is_zero or b_is_zero:
                    return self.canonical_nan
                sign = 1
                if a == "-inf":
                    sign *= -1
                if b == "-inf":
                    sign *= -1
                if isinstance(a, tuple) and a[0] < 0:
                    sign *= -1
                if isinstance(b, tuple) and b[0] < 0:
                    sign *= -1
                specials.append("-inf" if sign < 0 else "+inf")
                continue

            assert isinstance(a, tuple) and isinstance(b, tuple)
            product_mant = a[0] * b[0] * sa[0] * sb[0]
            product_exp = a[1] + b[1] + sa[1] + sb[1]
            product_raw = a[2] + b[2] + sa[2] + sb[2]
            scale_raw = sa[2] + sb[2]
            products.append((product_mant, product_exp))
            product_scale_raws.append(scale_raw if product_mant != 0 else None)
            product_raws.append(product_raw if product_mant != 0 else None)
            merge_groups[k // self.c_merge_group_size].append((product_mant, product_exp))
            if product_mant != 0:
                max_product_raw = (
                    product_raw if max_product_raw is None else max(max_product_raw, product_raw)
                )
                max_scale_raw = scale_raw if max_scale_raw is None else max(max_scale_raw, scale_raw)

        c = _decode_f32_int(case.c)
        if c == "nan":
            return self.canonical_nan
        if c in ("+inf", "-inf"):
            specials.append(c)

        special = _special_sum(specials)
        if special == "nan":
            return self.canonical_nan
        if special == "+inf":
            return F32_POS_INF
        if special == "-inf":
            return F32_NEG_INF

        assert isinstance(c, tuple)
        if max_product_raw is None:
            return F32_POS_ZERO if (case.c & 0x7FFF_FFFF) == 0 else case.c

        raw_exponents: list[int] = [max_product_raw]
        if c[0] != 0:
            raw_exponents.append(c[2])
        assert max_scale_raw is not None

        if self.use_c_dominant_merge and c[0] != 0 and c[2] >= max_product_raw:
            quantum_exp = c[2] - self.c_merge_fraction_bits
            units = _trunc_int_to_quantum(c[0], c[1], quantum_exp)
            for group in merge_groups:
                block_mant, block_exp = _sum_int_terms_to_common_exp(group)
                units += _trunc_int_to_quantum(block_mant, block_exp, quantum_exp)
            return _round_int_to_f32_rz(units, quantum_exp)

        quantum_exp = max(
            max(raw_exponents) - self.product_window_fraction_bits,
            max_scale_raw - self.scale_floor_offset,
        )
        units = _trunc_int_to_quantum(c[0], c[1], quantum_exp)
        if self.product_merge_group_size is None:
            units += sum(_trunc_int_to_quantum(mant, exp, quantum_exp) for mant, exp in products)
        else:
            product_group_units: list[tuple[int, int | None, int | None, int]] = []
            for start in range(0, len(case.a), self.product_merge_group_size):
                group = products[start : start + self.product_merge_group_size]
                block_mant, block_exp = _sum_int_terms_to_common_exp(group)
                group_scale_raws = [
                    scale_raw
                    for scale_raw in product_scale_raws[start : start + self.product_merge_group_size]
                    if scale_raw is not None
                ]
                group_scale_raw = max(group_scale_raws) if group_scale_raws else None
                group_product_raws = [
                    product_raw
                    for product_raw in product_raws[start : start + self.product_merge_group_size]
                    if product_raw is not None
                ]
                group_product_raw = max(group_product_raws) if group_product_raws else None
                block_quantum_exp = quantum_exp
                # MXF4 accumulates each K=16 group through the C-dominant
                # 35-bit window when C outranks that group, even if another
                # product group sets the wider global product window.
                if (
                    self.use_per_group_c_dominant_merge
                    and c[0] != 0
                    and group_product_raw is not None
                    and c[2] >= group_product_raw
                ):
                    block_quantum_exp = max(
                        block_quantum_exp,
                        c[2] - self.c_merge_fraction_bits,
                    )
                block_units = _trunc_int_to_quantum(block_mant, block_exp, block_quantum_exp)
                block_units <<= block_quantum_exp - quantum_exp
                product_group_units.append((block_units, group_scale_raw, group_product_raw, block_exp))
                units += block_units

            result = _round_int_to_f32_rz(units, quantum_exp)
            if (
                self.product_subnormal_group_floor_exp is not None
                and 0 < (result & 0x7FFF_FFFF) < (1 << 23)
            ):
                if quantum_exp < self.product_subnormal_group_floor_exp:
                    threshold_units = 1 << (self.product_subnormal_group_floor_exp - quantum_exp)
                else:
                    threshold_units = 1
                filtered_units = _trunc_int_to_quantum(c[0], c[1], quantum_exp)
                for block_units, group_scale_raw, _group_product_raw, _block_exp in product_group_units:
                    if (
                        group_scale_raw is not None
                        and group_scale_raw < max_scale_raw
                        and abs(block_units) < threshold_units
                    ):
                        continue
                    filtered_units += block_units
                units = filtered_units
            return _round_int_to_f32_rz(units, quantum_exp)
        return _round_int_to_f32_rz(units, quantum_exp)


class Tcgen05BlockScaledMxf8f6f4E4M3MmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf8f6f4-e4m3-ue8m0"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp8e4m3_int,
            scale_block_size=32,
            c_merge_group_size=32,
            product_window_fraction_bits=25,
            scale_floor_offset=1000,
            use_c_dominant_merge=False,
        )


class Tcgen05BlockScaledMxf8f6f4E5M2MmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf8f6f4-e5m2-ue8m0"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp8e5m2_int,
            scale_block_size=32,
            c_merge_group_size=32,
            product_window_fraction_bits=25,
            scale_floor_offset=1000,
            use_c_dominant_merge=False,
        )


class Tcgen05BlockScaledMxf8f6f4E2M3MmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf8f6f4-e2m3-ue8m0"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp6e2m3_int,
            scale_block_size=32,
            c_merge_group_size=32,
            product_window_fraction_bits=25,
            scale_floor_offset=1000,
            use_c_dominant_merge=False,
        )


class Tcgen05BlockScaledMxf8f6f4E3M2MmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf8f6f4-e3m2-ue8m0"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp6e3m2_int,
            scale_block_size=32,
            c_merge_group_size=32,
            product_window_fraction_bits=25,
            scale_floor_offset=1000,
            use_c_dominant_merge=False,
        )


class Tcgen05BlockScaledMxf8f6f4E2M1MmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf8f6f4-e2m1-ue8m0"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp4e2m1_int,
            scale_block_size=32,
            c_merge_group_size=32,
            product_window_fraction_bits=25,
            scale_floor_offset=1000,
            use_c_dominant_merge=False,
        )


class Tcgen05BlockScaledMxf4E2M1MmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf4-e2m1-ue8m0"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp4e2m1_int,
            scale_block_size=32,
            c_merge_group_size=16,
            product_merge_group_size=16,
            product_subnormal_group_floor_exp=-174,
            use_per_group_c_dominant_merge=True,
        )


class Tcgen05BlockScaledMxf4Nvfp4E2M1Scale2XMmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf4nvf4-e2m1-ue8m0-2x"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp4e2m1_int,
            scale_block_size=32,
            c_merge_group_size=16,
            product_merge_group_size=16,
            product_subnormal_group_floor_exp=-174,
            use_per_group_c_dominant_merge=True,
        )


class Tcgen05BlockScaledMxf4Nvfp4E2M1Scale4XMmaModel(Tcgen05BlockScaledMmaModel):
    input_format = "mxf4nvf4-e2m1-ue8m0-4x"

    def __init__(self):
        super().__init__(
            input_decoder=_decode_fp4e2m1_int,
            scale_block_size=16,
            c_merge_group_size=16,
            product_merge_group_size=16,
            product_subnormal_group_floor_exp=-174,
            use_per_group_c_dominant_merge=True,
        )


def _u32_to_s32(bits: int) -> int:
    bits &= 0xFFFF_FFFF
    return bits - 0x1_0000_0000 if bits & 0x8000_0000 else bits


def _i8_decode(bits: int, kind: str) -> int:
    value = bits & 0xFF
    if kind == "u8":
        return value
    if kind == "s8":
        return value - 0x100 if value & 0x80 else value
    raise ValueError(f"unsupported i8 operand type: {kind}")


class Tcgen05I8S32MmaModel(_ModelBase):
    """Exact scalar model for tcgen05.kind::i8 dense K32 S32 accumulation."""

    def __init__(self, *, a_type: str = "u8", b_type: str = "u8", saturate: bool = False):
        if a_type not in ("u8", "s8"):
            raise ValueError(f"unsupported A type: {a_type}")
        if b_type not in ("u8", "s8"):
            raise ValueError(f"unsupported B type: {b_type}")
        self.a_type = a_type
        self.b_type = b_type
        self.saturate = saturate

    def eval(self, case: DotProduct) -> int:
        total = _u32_to_s32(case.c)
        for a_bits, b_bits in zip(case.a, case.b):
            total += _i8_decode(a_bits, self.a_type) * _i8_decode(b_bits, self.b_type)
        if self.saturate:
            total = min(max(total, -0x8000_0000), 0x7FFF_FFFF)
        return total & 0xFFFF_FFFF

FORMAT_MODELS = {
    "tf32": Tcgen05RawWindowTf32MmaModel,
    "bf16": Tcgen05RawWindowBf16MmaModel,
    "f16": Tcgen05RawWindowF16MmaModel,
    "e4m3": Tcgen05RawWindowFp8E4M3MmaModel,
    "e5m2": Tcgen05RawWindowFp8E5M2MmaModel,
    "e2m3": Tcgen05RawWindowFp6E2M3MmaModel,
    "e3m2": Tcgen05RawWindowFp6E3M2MmaModel,
    "e2m1": Tcgen05RawWindowFp4E2M1MmaModel,
}

F16_INPUT_FORMATS = frozenset(("bf16", "f16"))
F8F6F4_INPUT_FORMATS = frozenset(("e4m3", "e5m2", "e2m3", "e3m2", "e2m1"))


def normalize_format(name: str) -> str:
    """Return the canonical short name used by this package."""

    normalized = name.lower().replace("_", "").replace("-", "")
    aliases = {
        "fp8": "e4m3",
        "fp8e4m3": "e4m3",
        "fp8e5m2": "e5m2",
        "fp6e2m3": "e2m3",
        "fp6e3m2": "e3m2",
        "fp4": "e2m1",
        "fp4e2m1": "e2m1",
    }
    return aliases.get(normalized, normalized)


def make_model(
    a_format: str,
    b_format: str | None = None,
    *,
    d_type: str = "f32",
    saturate: bool = False,
    scaling: str | None = None,
    scale_vec: int | None = None,
    sparse: bool = False,
    kind: str | None = None,
) -> ScalarModel:
    """Construct the arithmetic model selected by a tcgen05 descriptor.

    ``scaling`` accepts ``None``, ``"ue8m0"``, or ``"ue4m3"``. ``kind``
    disambiguates the E2M1 UE8M0 families: ``"mxf8f6f4"``, ``"mxf4"``, or
    ``"mxf4nvf4"``.
    """

    a_format = normalize_format(a_format)
    b_format = normalize_format(b_format or a_format)
    d_type = d_type.lower()
    kind = None if kind is None else kind.lower()

    known_formats = frozenset(("tf32", *F16_INPUT_FORMATS, *F8F6F4_INPUT_FORMATS, "u8", "s8"))
    if a_format not in known_formats or b_format not in known_formats:
        raise ValueError(f"unsupported input format combination: {a_format}/{b_format}")
    if d_type not in ("f16", "f32", "s32"):
        raise ValueError(f"unsupported C/D type: {d_type!r}")

    if a_format in ("u8", "s8") or b_format in ("u8", "s8"):
        if scaling is not None or kind is not None or scale_vec is not None:
            raise ValueError("kind::i8 does not use block scaling")
        if a_format not in ("u8", "s8") or b_format not in ("u8", "s8") or d_type != "s32":
            raise ValueError("kind::i8 requires U8/S8 A and B with S32 C/D")
        return Tcgen05I8S32MmaModel(a_type=a_format, b_type=b_format, saturate=saturate)

    if saturate:
        raise ValueError("saturate is only valid for kind::i8")

    if scaling is None:
        if kind is not None:
            raise ValueError("kind is only used by block-scaled MMA")
        if scale_vec is not None:
            raise ValueError("scale_vec is only used by block-scaled MMA")
        if a_format == "tf32" or b_format == "tf32":
            if a_format != "tf32" or b_format != "tf32" or d_type != "f32":
                raise ValueError("kind::tf32 requires TF32 A/B and F32 C/D")
        elif a_format in F16_INPUT_FORMATS or b_format in F16_INPUT_FORMATS:
            if a_format != b_format or a_format not in F16_INPUT_FORMATS:
                raise ValueError("B200 kind::f16 requires matching BF16/BF16 or F16/F16 A/B")
            if d_type == "f16" and a_format != "f16":
                raise ValueError("B200 kind::f16 permits F16 C/D only with F16 A/B")
            if d_type not in ("f16", "f32"):
                raise ValueError("kind::f16 requires F16 or F32 C/D")
        elif a_format in F8F6F4_INPUT_FORMATS and b_format in F8F6F4_INPUT_FORMATS:
            if d_type not in ("f16", "f32"):
                raise ValueError("kind::f8f6f4 requires F16 or F32 C/D")
        else:
            raise ValueError(f"unsupported floating-point descriptor: {a_format}/{b_format}/{d_type}")
        if d_type == "f32" and a_format == b_format and a_format in FORMAT_MODELS:
            return FORMAT_MODELS[a_format]()
        return Tcgen05MixedRawWindowMmaModel(a_format, b_format, d_type=d_type)

    scaling = scaling.lower()
    if scaling == "ue4m3":
        if kind not in (None, "nvfp4", "mxf4nvf4"):
            raise ValueError("UE4M3 scaling is only valid for the NVFP4 family")
        scale_vec = 4 if scale_vec is None else scale_vec
        if a_format != "e2m1" or b_format != "e2m1" or d_type != "f32" or scale_vec != 4:
            raise ValueError("UE4M3 scaling models NVFP4 E2M1/E2M1, F32 D, scale_vec::4X")
        return Tcgen05RawWindowNvfp4MmaModel()

    if scaling != "ue8m0":
        raise ValueError(f"unsupported scale format: {scaling!r}")
    if kind not in (None, "mxf8f6f4", "mxf4", "mxf4nvf4"):
        raise ValueError(f"unsupported block-scaled kind: {kind!r}")
    if d_type != "f32":
        raise ValueError("block-scaled model currently supports F32 C/D")
    if kind is None:
        if (
            a_format in F8F6F4_INPUT_FORMATS
            and b_format in F8F6F4_INPUT_FORMATS
            and (a_format != "e2m1" or b_format != "e2m1")
        ):
            kind = "mxf8f6f4"
        else:
            raise ValueError("kind is required for ambiguous UE8M0 descriptors")
    if kind == "mxf8f6f4":
        scale_vec = 1 if scale_vec is None else scale_vec
        if scale_vec != 1:
            raise ValueError("MXF8F6F4 UE8M0 requires scale_vec::1X")
        if a_format not in F8F6F4_INPUT_FORMATS or b_format not in F8F6F4_INPUT_FORMATS:
            raise ValueError("MXF8F6F4 requires f8/f6/f4 A and B")
        classes = {
            "e4m3": Tcgen05BlockScaledMxf8f6f4E4M3MmaModel,
            "e5m2": Tcgen05BlockScaledMxf8f6f4E5M2MmaModel,
            "e2m3": Tcgen05BlockScaledMxf8f6f4E2M3MmaModel,
            "e3m2": Tcgen05BlockScaledMxf8f6f4E3M2MmaModel,
            "e2m1": Tcgen05BlockScaledMxf8f6f4E2M1MmaModel,
        }
        if a_format != b_format or a_format not in classes:
            return Tcgen05BlockScaledMmaModel(
                input_decoder=_tcgen05_decoder_for_format(a_format),
                input_decoder_b=_tcgen05_decoder_for_format(b_format),
                scale_block_size=32,
                c_merge_group_size=32,
                product_window_fraction_bits=25,
                scale_floor_offset=1000,
                use_c_dominant_merge=False,
            )
        return classes[a_format]()
    if kind in ("mxf4", "mxf4nvf4"):
        if a_format != "e2m1" or b_format != "e2m1":
            raise ValueError(f"{kind} requires E2M1 A and B")
        scale_vec = 2 if scale_vec is None else scale_vec
        if kind == "mxf4":
            if scale_vec != 2:
                raise ValueError("MXF4 UE8M0 requires scale_vec::2X")
            return Tcgen05BlockScaledMxf4E2M1MmaModel()
        if scale_vec == 2:
            return Tcgen05BlockScaledMxf4Nvfp4E2M1Scale2XMmaModel()
        if scale_vec == 4:
            return Tcgen05BlockScaledMxf4Nvfp4E2M1Scale4XMmaModel()
        raise ValueError("MXF4NVF4 UE8M0 requires scale_vec::2X or scale_vec::4X")
    raise ValueError(f"unsupported block-scaled descriptor: {kind}/{a_format}/{b_format}")

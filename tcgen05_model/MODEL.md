# How the tcgen05 arithmetic model works

This document describes the software model from raw operand bits to the raw D
word. The implementation uses integer arithmetic throughout; host floating
point never participates in an MMA result.

## 1. Simulation boundary

A tcgen05 instruction updates a matrix, but every logical D element has the
same arithmetic form:

```text
D[row, col] = dot(A[row, :], B[:, col]) + C[row, col]
```

`mma_dot` implements one such output and `mma` applies it to every row/column
pair. M, N, CTA group, `.ws`, descriptor layout, and the physical TMEM word
holding D affect placement and execution, not the arithmetic described here.
Consequently the matrix API accepts any positive M/N and a K compatible with
the selected instruction variant.

The API consumes raw container words. This avoids conversion by the host and
preserves subnormals, signed zero, NaN payloads, and format-specific values.

## 2. Exact integer representation

Every finite floating input is decoded as a triple:

```text
(signed_mantissa, value_exponent, raw_exponent)
value = signed_mantissa * 2**value_exponent
```

`raw_exponent` is the unbiased encoded exponent, using the minimum normal
exponent for subnormals. Keeping it separately matters: tcgen05 chooses its
accumulation window from encoded operand exponents rather than from the
normalized mathematical product.

The decoders are:

| API name | Stored bits | Sign / exponent / fraction | Bias | Special values |
|---|---:|---:|---:|---|
| `tf32` | F32 container | 1 / 8 / 10 retained | 127 | IEEE Inf/NaN after truncation |
| `bf16` | high 16 of F32 | 1 / 8 / 7 | 127 | IEEE Inf/NaN after truncation |
| `f16` | low 16 | 1 / 5 / 10 | 15 | IEEE Inf/NaN |
| `e4m3` | low 8 | 1 / 4 / 3 | 7 | only exponent 15, fraction 7 is NaN |
| `e5m2` | low 8 | 1 / 5 / 2 | 15 | IEEE-style Inf/NaN |
| `e2m3` | low 6 | 1 / 2 / 3 | 1 | all encodings finite |
| `e3m2` | low 6 | 1 / 3 / 2 | 3 | all encodings finite |
| `e2m1` | low 4 | 1 / 2 / 1 | 1 | all encodings finite |

TF32 clears F32 fraction bits 0 through 12 before classification. BF16 clears
bits 0 through 15. This ordering is observable: a source F32 NaN whose payload
exists only in cleared bits becomes infinity at the tcgen05 input.

## 3. The shared fixed-precision window

Unscaled floating paths use one 25-fraction-bit raw-exponent window. For every
nonzero product, let `r_k = raw_exp(A_k) + raw_exp(B_k)`. Let `r_c` be C's raw
exponent when C is nonzero. The window quantum is:

```text
q = max(r_0, r_1, ..., r_c) - 25
```

Each exact product and C is converted independently to signed units of `2**q`.
Right shifts discard low magnitude bits, which is truncation toward zero for
both signs:

```text
units(term, q) = trunc_toward_zero(term / 2**q)
```

The signed units are then summed exactly as Python integers. This is important:
the products are not first rounded to their input format or to F32, and the
dot product is not a sequence of host FMAs. A small term lost while aligning
to the shared window cannot reappear through later cancellation.

Finally the integer sum is normalized to F32 and its discarded magnitude bits
are truncated (round toward zero). An exact zero result is canonical `+0`.

Mixed f8/f6/f4 with F16 D uses the same 25-bit window, then converts the final
integer to F16 with round-to-nearest-even. Its returned container word uses the
low 16 bits.

## 4. Special values

Special-value handling happens before finite accumulation. For F32 D:

- an input NaN produces `0x7fffffff`;
- infinity times zero produces `0x7fffffff`;
- opposite-signed infinite contributions produce `0x7fffffff`;
- otherwise an infinite contribution produces the corresponding F32 infinity;
- generated NaNs use exactly `0x7fffffff`, rather than accepting any IEEE NaN
  payload as equivalent.

For mixed f8/f6/f4 instructions with F16 D, the same rules use destination-width
encodings: generated NaN is exactly `0x7fff`, and infinities are `0x7c00` and
`0xfc00`. This distinction is bit-visible even though the classifications are
the same.

Sparse-disabled operands are absent from the dot product. They are masked on
both A and B before classification, so a disabled NaN or infinity has no
effect.

## 5. Block scaling

Scale values are decoded to the same integer/exponent representation and are
multiplied into every affected product exactly. `scale_a` is selected by the A
row and K block; `scale_b` by the B column and K block.

### UE8M0 MXF8F6F4

UE8M0 is an unsigned exponent-only value:

```text
scale = 2**(bits - 127), bits != 0xff
```

`0xff` is NaN. MXF8F6F4 uses `scale_vec::1X`: E4M3/E5M2/E2M3/E3M2/E2M1
paths use K=32 physical scale blocks and the same 25-bit shared raw window as
unscaled f8/f6/f4. Mixed A/B formats use their respective decoders before
applying the scales.

For E2M1, pass `kind="mxf8f6f4"` to select this K=32 arithmetic instead of the
MXF4 family described below.

### UE8M0 MXF4 and MXF4NVF4

The E2M1 MXF4 family has a wider product window. Define:

```text
R = maximum nonzero scaled-product raw exponent
S = maximum nonzero (A-scale raw exponent + B-scale raw exponent)
q = max(R - 39, S - 35, raw_exp(C) - 39 when C is nonzero)
```

Products are formed in K=16 merge groups. MXF4 requires `scale_vec::2X`;
MXF4NVF4 accepts `scale_vec::2X` or `scale_vec::4X`. `2X` uses one scale for
32 physical K values and `4X` uses one for 16. When C outranks an individual
product group, that group is first truncated through the C-relative 35-bit
window. The model also applies the observed `2**-174` subnormal group floor to
lower-scale groups.

### UE4M3 NVFP4

UE4M3 uses the positive E4M3 encoding (7 bits); `0x7f` is NaN. Dense NVFP4
`scale_vec::4X` has K=16 scale and merge blocks. Its product-dominant window is:

```text
q = max(max(product raw exponent, C raw exponent) - 39,
        max(scale-pair raw exponent) - 35)
```

Products are first reduced exactly in aligned K=4 groups, and each group is
then truncated into this shared window. This ordering preserves carries within
a four-product dot group that would be lost by per-product truncation.

If nonzero C's raw exponent is at least the maximum product raw exponent, the
datapath exposes each K=16 physical block separately: each exact block sum is
truncated to `q = raw_exp(C) - 35`, then the block units and C units are added.
For sparse input, each stored physical block represents twice as much logical
K; the public API compresses metadata-selected logical lanes before assigning
these physical blocks.

## 6. Integer MMA

Integer A/B words use their low byte. U8 is decoded as 0 through 255; S8 uses
two's-complement -128 through 127. C is a signed two's-complement S32 word.
The full dot product and C are summed exactly.

- nonsaturating mode returns the sum modulo `2**32`;
- saturating mode clamps once, after the exact sum, to
  `[-2**31, 2**31 - 1]`.

## 7. Sparse metadata

Metadata is read as eight 4-bit selectors per uint32 word, least-significant
nibble first.

- TF32 sparse MMA selects one element per pair. Selector `0x4` chooses the
  lower element and `0xe` the upper.
- BF16, F16, and f8/f6/f4 select two elements per four. The low and high
  2-bit fields identify the two positions; legal selectors are
  `4, 8, c, 9, d, 6, e`.
- MXF4 and MXF4NVF4, under either UE8M0 or UE4M3 scaling, interpret those two
  fields as adjacent pairs within each eight-wide chunk, selecting four
  logical values. Other f8/f6/f4 families use 2:4 selection.

`sparse_active_indices` exposes this decoder; pass `kind="mxf4"` or
`kind="mxf4nvf4"` for pairwise 4:8. `mma_dot` and `mma` accept full logical-K
matrices and remove all disabled A/B positions before arithmetic. Block-scaled
sparse paths compress selected A/B lanes to physical K before scale assignment.

## 8. Public API

- `mma_dot(...) -> int`: simulate one logical D word.
- `mma(...) -> tuple[tuple[int, ...], ...]`: simulate row-major matrices.
- `make_model(...)`: construct and reuse a scalar model directly.
- `DotProduct`: immutable raw scalar input for `model.eval(...)`.
- `sparse_active_indices(...)`: inspect metadata selection.
- `f32_to_bits` / `bits_to_f32`: convenience conversions at API boundaries.

The concrete classes in `tcgen05_model.model` remain available for callers
that need to select a particular accumulator family directly. `make_model` is
preferred because it checks descriptor combinations.

`make_model` accepts only B200-supported descriptor families: TF32/TF32 with
F32 D; matching BF16 or F16 inputs (F16 D only with F16 inputs); arbitrary
f8/f6/f4 A/B combinations with F32 or F16 D; U8/S8 with S32 D; and the exact
block-scale kind/scale-vector combinations described above. Unsupported names
and cross-family combinations raise `ValueError`.

## 9. Validation and confidence

`python -m tcgen05_model.validation` replays frozen raw result words captured
on NVIDIA B200 hardware. It covers all unscaled floating formats, integer
signedness and saturation, the UE8M0 MX families, non-unit UE4M3 NVFP4
scaling, sparse non-finite masking, and matrix/scalar consistency. These are
regression vectors, not a hardware runner; validation is deterministic and
GPU-free.

The original development corpus also compared the raw-window models against
large randomized hardware samples. That discovery machinery is intentionally
not part of this release. The package models arithmetic, not undocumented
future hardware revisions, so consumers should treat behavior on architectures
other than the validated Blackwell tcgen05 implementation as unverified.

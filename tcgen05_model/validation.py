"""GPU-free validation against frozen B200 result words."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from .model import DotProduct, make_model
from .simulator import mma, mma_dot


@dataclass(frozen=True)
class ValidationFailure:
    name: str
    expected: int
    actual: int


@dataclass(frozen=True)
class ValidationReport:
    checked: int
    failures: tuple[ValidationFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def _finite_f32(rng: random.Random) -> int:
    bits = rng.getrandbits(32)
    if bits & 0x7F80_0000 == 0x7F80_0000:
        bits &= 0x807F_FFFF
    return bits


def _unscaled_vectors():
    specs = (
        ("tf32", 8, 104, _finite_f32, (0xFF800000, 0xFC347E7C)),
        ("bf16", 16, 104, _finite_f32, (0x7F800000, 0x7F800000)),
        ("f16", 16, 103, lambda rng: rng.getrandbits(16), (0x7FFFFFFF, 0x49776BBA)),
        ("e4m3", 32, 104, lambda rng: rng.randrange(0x7F), (0xEC63D12A, 0x452AF963)),
        ("e5m2", 32, 300, lambda rng: rng.randrange(0x7C), (0x4D92B09A, 0x4C9C1102)),
        ("e2m3", 32, 301, lambda rng: rng.randrange(0x40), (0xC29DD800, 0x449D14C1)),
        ("e3m2", 32, 302, lambda rng: rng.randrange(0x40), (0x52C9EC63, 0xC33B7700)),
        ("e2m1", 32, 303, lambda rng: rng.randrange(0x10), (0xC1FE0000, 0xDC3B62A3)),
    )
    for format_name, k, seed, element, expected_words in specs:
        rng = random.Random(seed)
        model = make_model(format_name)
        for index, expected in enumerate(expected_words):
            case = DotProduct(
                tuple(element(rng) for _ in range(k)),
                tuple(element(rng) for _ in range(k)),
                _finite_f32(rng),
            )
            yield f"{format_name}-{index}", model, case, expected


def _integer_vectors():
    specs = (
        ("u8", "u8", False, 500, 0x078A5C64),
        ("s8", "u8", False, 501, 0xF66E2FA6),
        ("u8", "s8", False, 502, 0xD73F1A46),
        ("s8", "s8", True, 503, 0x8063D12F),
        ("s8", "s8", False, 504, 0xA4679E93),
        ("u8", "u8", True, 505, 0x55DB8BB8),
        ("s8", "u8", True, 506, 0xB47F6330),
        ("u8", "s8", True, 507, 0x8533C14D),
    )
    for a_type, b_type, saturate, seed, expected in specs:
        rng = random.Random(seed)
        case = DotProduct(
            tuple(rng.randrange(256) for _ in range(32)),
            tuple(rng.randrange(256) for _ in range(32)),
            _finite_f32(rng),
        )
        yield (
            f"i8-{a_type}-{b_type}-{'sat' if saturate else 'wrap'}",
            make_model(a_type, b_type, d_type="s32", saturate=saturate),
            case,
            expected,
        )


def _nvfp4_vectors():
    rng = random.Random(700)
    expected_words = (0x4429730B, 0xC30545D3)
    model = make_model("e2m1", scaling="ue4m3", scale_vec=4)
    for index, expected in enumerate(expected_words):
        a = tuple(rng.randrange(16) for _ in range(64))
        b = tuple(rng.randrange(16) for _ in range(64))
        c = _finite_f32(rng)
        scale_a = tuple(rng.randrange(0x77) for _ in range(4))
        scale_b = tuple(rng.randrange(0x77) for _ in range(4))
        yield f"nvfp4-{index}", model, DotProduct(a, b, c, scale_a, scale_b), expected


def _ue8m0_vectors():
    rng = random.Random(910)
    a = tuple(rng.randrange(16) for _ in range(32))
    b = tuple(rng.randrange(16) for _ in range(32))
    scale_a = (rng.randrange(115, 140), 0x7F, 0x7F, 0x7F)
    scale_b = (rng.randrange(115, 140), 0x7F, 0x7F, 0x7F)
    yield (
        "mxf8f6f4-e2m1",
        make_model("e2m1", scaling="ue8m0", kind="mxf8f6f4"),
        DotProduct(a, b, 0, scale_a, scale_b),
        0x3F094000,
    )

    specs = (
        ("mxf4", 2, 920, 2, 0x46BA8000),
        ("mxf4nvf4", 2, 921, 2, 0x40580000),
        ("mxf4nvf4", 4, 922, 4, 0xC6A88896),
    )
    for kind, scale_vec, seed, scale_count, expected in specs:
        rng = random.Random(seed)
        a = tuple(rng.randrange(16) for _ in range(64))
        b = tuple(rng.randrange(16) for _ in range(64))
        scale_a = tuple(rng.randrange(115, 140) for _ in range(scale_count)) + (0x7F,) * (4 - scale_count)
        scale_b = tuple(rng.randrange(115, 140) for _ in range(scale_count)) + (0x7F,) * (4 - scale_count)
        yield (
            f"{kind}-{scale_vec}x",
            make_model("e2m1", scaling="ue8m0", kind=kind, scale_vec=scale_vec),
            DotProduct(a, b, 0, scale_a, scale_b),
            expected,
        )


def _edge_vectors():
    """Labeled B200 vectors for values raw randomness rarely hits exactly."""

    specs = (
        # format, K, one, minimum subnormal, NaN, infinity
        ("tf32", 8, 0x3F800000, 0x00002000, 0x7FC00000, 0x7F800000),
        ("bf16", 16, 0x3F800000, 0x00010000, 0x7FC00000, 0x7F800000),
        ("f16", 16, 0x00003C00, 0x00000001, 0x00007E00, 0x00007C00),
        ("e4m3", 32, 0x38, 0x01, 0x7F, None),
        ("e5m2", 32, 0x3C, 0x01, 0x7F, 0x7C),
    )
    subnormal_results = {
        "tf32": 0x00002000,
        "bf16": 0x00010000,
        "f16": 0x33800000,
        "e4m3": 0x3B000000,
        "e5m2": 0x37800000,
    }
    for format_name, k, one, subnormal, nan, infinity in specs:
        model = make_model(format_name)

        def case(a0: int, b0: int, c: int = 0) -> DotProduct:
            return DotProduct((a0,) + (0,) * (k - 1), (b0,) + (0,) * (k - 1), c)

        yield f"{format_name}-negative-zero", model, case(0, 0, 0x80000000), 0
        yield f"{format_name}-minimum-subnormal", model, case(subnormal, one), subnormal_results[format_name]
        yield f"{format_name}-nan", model, case(nan, one), 0x7FFFFFFF
        if infinity is not None:
            yield f"{format_name}-infinity", model, case(infinity, one), 0x7F800000
            yield f"{format_name}-infinity-times-zero", model, case(infinity, 0), 0x7FFFFFFF

    # TF32/BF16 clear low payload bits before classifying the input. A NaN
    # whose payload exists only in those bits therefore arrives as infinity.
    for format_name, k in (("tf32", 8), ("bf16", 16)):
        case = DotProduct(
            (0x7F800001,) + (0,) * (k - 1),
            (0x3F800000,) + (0,) * (k - 1),
            0,
        )
        yield f"{format_name}-low-payload-pretruncate", make_model(format_name), case, 0x7F800000

    nvfp4 = DotProduct((1,) * 64, (1,) * 64, 0, (0x7F, 0x38, 0x38, 0x38), (0x38,) * 4)
    yield "nvfp4-scale-nan", make_model("e2m1", scaling="ue4m3"), nvfp4, 0x7FFFFFFF

    # This hardware case lands one internal quantum below an F32 boundary if
    # products are truncated independently. B200 preserves the carry by first
    # reducing aligned K=4 product groups.
    nvfp4_k4_carry = DotProduct(
        tuple(bytes.fromhex(
            "0e 0e 0b 06 00 08 0f 09 0a 0b 03 09 0f 07 0e 00 "
            "02 06 01 05 0a 08 06 04 07 0c 06 0b 01 0c 06 0d "
            "0b 0b 08 06 0c 08 0c 05 00 04 01 0f 01 06 0f 05 "
            "02 04 06 05 0a 07 0b 07 00 01 0a 0d 0b 0f 0e 07"
        )),
        tuple(bytes.fromhex(
            "06 0c 0b 02 08 09 0b 04 03 0d 02 09 09 0c 01 08 "
            "07 02 06 03 0d 02 09 0a 05 09 0f 05 0a 08 06 00 "
            "08 0d 0d 02 0c 0c 05 02 08 03 04 0f 0f 01 01 0a "
            "0b 0a 05 0b 0f 03 0b 0e 0a 05 07 01 01 07 08 0a"
        )),
        0x094E5986,
        (0x09, 0x7C, 0x6D, 0x44),
        (0x07, 0x78, 0x26, 0x79),
    )
    yield (
        "nvfp4-k4-product-carry",
        make_model("e2m1", scaling="ue4m3"),
        nvfp4_k4_carry,
        0x49FCAFB7,
    )

    mx = DotProduct((1,) * 32, (1,) * 32, 0, (0xFF, 0x7F, 0x7F, 0x7F), (0x7F,) * 4)
    yield (
        "mxf8f6f4-scale-nan",
        make_model("e2m1", scaling="ue8m0", kind="mxf8f6f4"),
        mx,
        0x7FFFFFFF,
    )

    # F16-D instructions return a 16-bit result word.  Keep the canonical
    # special-value encoding in that destination type rather than F32.
    mixed_f16 = make_model("e2m1", "e4m3", d_type="f16")
    yield (
        "mixed-e2m1-e4m3-f16-input-nan",
        mixed_f16,
        DotProduct((1,) * 32, (0x7F,) + (0,) * 31, 0),
        0x7FFF,
    )
    yield (
        "mixed-e2m1-e4m3-f16-c-nan",
        mixed_f16,
        DotProduct((0,) * 32, (0,) * 32, 0x7E00),
        0x7FFF,
    )


def validate() -> ValidationReport:
    """Run all frozen hardware vectors and public-interface consistency checks."""

    failures: list[ValidationFailure] = []
    checked = 0
    for name, model, case, expected in (
        *_unscaled_vectors(),
        *_integer_vectors(),
        *_nvfp4_vectors(),
        *_ue8m0_vectors(),
        *_edge_vectors(),
    ):
        actual = model.eval(case)
        checked += 1
        if actual != expected:
            failures.append(ValidationFailure(name, expected, actual))

    # The matrix interface must be exactly the scalar model applied per output.
    a = ((0x3F800000, 0x40000000), (0x40400000, 0x40800000))
    b = ((0x3F800000, 0xBF800000), (0x3F000000, 0x40000000))
    c = ((0, 0), (0x3F800000, 0))
    result = mma(a, b, c, a_format="tf32")
    for row in range(2):
        for col in range(2):
            expected = mma_dot(a[row], (b[k][col] for k in range(2)), c[row][col], a_format="tf32")
            checked += 1
            if result[row][col] != expected:
                failures.append(ValidationFailure(f"matrix-{row}-{col}", expected, result[row][col]))

    # Disabled sparse positions, including non-finite B values, are absent
    # from arithmetic and therefore must equal an explicitly masked dense dot.
    sparse = mma_dot(
        (0x3F800000,) * 16,
        tuple(0x3F800000 if k % 2 == 0 else 0x7FC00000 for k in range(16)),
        a_format="tf32",
        sparse_metadata=(0x44444444,),
    )
    dense_masked = mma_dot(
        tuple(0x3F800000 if k % 2 == 0 else 0 for k in range(16)),
        tuple(0x3F800000 if k % 2 == 0 else 0 for k in range(16)),
        a_format="tf32",
    )
    checked += 1
    if sparse != dense_masked:
        failures.append(ValidationFailure("sparse-mask", dense_masked, sparse))

    def check_public(name: str, expected: int, **kwargs) -> None:
        nonlocal checked
        actual = mma_dot(**kwargs)
        checked += 1
        if actual != expected:
            failures.append(ValidationFailure(name, expected, actual))

    # Frozen public-API sparse cases. These deliberately use full logical K;
    # the expected words were captured from B200 sparse instructions.
    metadata = (0x44444444, 0x44444444)
    check_public(
        "sparse-mxf8f6f4-logical-k-scale-association",
        0x42000000,
        a=(0x38,) * 64,
        b=(0x38,) * 64,
        a_format="e4m3",
        scaling="ue8m0",
        kind="mxf8f6f4",
        scale_vec=1,
        scale_a=(0x7F, 0x80, 0x7F, 0x7F),
        scale_b=(0x7F,) * 4,
        sparse_metadata=metadata,
    )
    pairwise_a = tuple(0x2 if k % 8 in (2, 3) else 0 for k in range(128))
    for name, scaling, kind, scale_vec, scale in (
        ("sparse-mxf4-pairwise", "ue8m0", "mxf4", 2, 0x7F),
        ("sparse-mxf4nvf4-2x-pairwise", "ue8m0", "mxf4nvf4", 2, 0x7F),
        ("sparse-mxf4nvf4-4x-pairwise", "ue8m0", "mxf4nvf4", 4, 0x7F),
        ("sparse-nvfp4-pairwise", "ue4m3", "mxf4nvf4", 4, 0x38),
    ):
        check_public(
            name,
            0x42000000,
            a=pairwise_a,
            b=(0x2,) * 128,
            a_format="e2m1",
            scaling=scaling,
            kind=kind,
            scale_vec=scale_vec,
            scale_a=(scale,) * 4,
            scale_b=(scale,) * 4,
            sparse_metadata=metadata,
        )

    # B200 canonicalizes an all-zero block-scaled result to +0 even when C is
    # -0. Cover every dense block-scaled arithmetic family explicitly.
    for format_name, scaling, kind, scale_vec, k, scale in (
        ("e4m3", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e5m2", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e2m3", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e3m2", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e2m1", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e2m1", "ue8m0", "mxf4", 2, 64, 0x7F),
        ("e2m1", "ue8m0", "mxf4nvf4", 2, 64, 0x7F),
        ("e2m1", "ue8m0", "mxf4nvf4", 4, 64, 0x7F),
        ("e2m1", "ue4m3", "mxf4nvf4", 4, 64, 0x38),
    ):
        check_public(
            f"{kind}-{format_name}-{scale_vec}x-negative-zero",
            0,
            a=(0,) * k,
            b=(0,) * k,
            c=0x80000000,
            a_format=format_name,
            scaling=scaling,
            kind=kind,
            scale_vec=scale_vec,
            scale_a=(scale,) * 4,
            scale_b=(scale,) * 4,
        )

    return ValidationReport(checked, tuple(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = validate()
    if report.passed:
        print(f"PASS: {report.checked} tcgen05 model checks")
        return 0
    for failure in report.failures:
        print(f"FAIL {failure.name}: expected 0x{failure.expected:08x}, got 0x{failure.actual:08x}")
    print(f"FAIL: {len(report.failures)} of {report.checked} checks failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

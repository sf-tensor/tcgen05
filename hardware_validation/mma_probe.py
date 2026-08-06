#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcgen05_model import mma_dot as public_mma_dot

from mma_probe.formats import bits_to_f32, f32_class
from mma_probe.harness import MmaHarness, ProbeCase
from mma_probe.model import (
    FixedAlignTf32MmaModel,
    SimpleTf32MmaModel,
    Tcgen05ExactTf32MmaModel,
    Tcgen05BlockScaledMmaModel,
    Tcgen05BlockScaledMxf4E2M1MmaModel,
    Tcgen05BlockScaledMxf4Nvfp4E2M1Scale2XMmaModel,
    Tcgen05BlockScaledMxf4Nvfp4E2M1Scale4XMmaModel,
    Tcgen05BlockScaledMxf8f6f4E2M1MmaModel,
    Tcgen05BlockScaledMxf8f6f4E2M3MmaModel,
    Tcgen05BlockScaledMxf8f6f4E3M2MmaModel,
    Tcgen05BlockScaledMxf8f6f4E4M3MmaModel,
    Tcgen05BlockScaledMxf8f6f4E5M2MmaModel,
    Tcgen05I8S32MmaModel,
    Tcgen05MixedRawWindowMmaModel,
    Tcgen05ProductSumThenCModel,
    Tcgen05RawWindowBf16MmaModel,
    Tcgen05RawWindowF16MmaModel,
    Tcgen05RawWindowFp4E2M1MmaModel,
    Tcgen05RawWindowFp6E2M3MmaModel,
    Tcgen05RawWindowFp6E3M2MmaModel,
    Tcgen05RawWindowFp8E4M3MmaModel,
    Tcgen05RawWindowFp8E5M2MmaModel,
    Tcgen05RawWindowNvfp4MmaModel,
    Tcgen05RawWindowTf32MmaModel,
    _tcgen05_decoder_for_format,
)
from mma_probe.probes import bit_hex, first_word, run_quick_probes, unique_words
from mma_probe.tcgen05 import (
    BLOCK_SCALED_OPS,
    BLOCK_SCALED_SPARSE_OPS,
    MXF8F6F4_UE8M0_FORMAT_OPS,
    MX_UE8M0_UNIT_SCALES,
    NVFP4_SPARSE_UNIT_SCALES,
    NVFP4_UNIT_SCALES,
    SPARSE_2OF4_SELECTORS,
    SPARSE_FIXED_METADATA,
    SPARSE_TF32_SELECTORS,
    CTA_GROUP2_THREADS,
    THREADS,
    Tcgen05Bf16SparseHarness,
    Tcgen05Bf16SparseProbeCase,
    Tcgen05Bf16Harness,
    Tcgen05Bf16ProbeCase,
    Tcgen05BlockScaledHarness,
    Tcgen05BlockScaledProbeCase,
    Tcgen05BlockScaledSparseHarness,
    Tcgen05BlockScaledSparseProbeCase,
    Tcgen05F16SparseHarness,
    Tcgen05F16SparseProbeCase,
    Tcgen05F16Harness,
    Tcgen05F16ProbeCase,
    Tcgen05F8f6f4SparseHarness,
    Tcgen05F8f6f4SparseProbeCase,
    Tcgen05F8f6f4Harness,
    Tcgen05F8f6f4ProbeCase,
    Tcgen05I8Harness,
    Tcgen05I8ProbeCase,
    Tcgen05Fp8SparseHarness,
    Tcgen05Fp8SparseProbeCase,
    Tcgen05Fp8Harness,
    Tcgen05Fp8ProbeCase,
    Tcgen05Harness,
    Tcgen05Nvfp4SparseHarness,
    Tcgen05Nvfp4SparseProbeCase,
    Tcgen05Nvfp4Harness,
    Tcgen05Nvfp4ProbeCase,
    Tcgen05ProbeCase,
    Tcgen05ShapeHarness,
    Tcgen05ShapeProbeCase,
    Tcgen05Tf32SparseHarness,
    Tcgen05Tf32SparseProbeCase,
    bit_hex as tcgen05_bit_hex,
    flatten_bf16_output as tcgen05_bf16_flatten_output,
    flatten_f16_output as tcgen05_f16_flatten_output,
    flatten_f8f6f4_output as tcgen05_f8f6f4_flatten_output,
    flatten_i8_output as tcgen05_i8_flatten_output,
    flatten_fp8_output as tcgen05_fp8_flatten_output,
    flatten_nvfp4_output as tcgen05_nvfp4_flatten_output,
    flatten_output as tcgen05_flatten_output,
    first_word as tcgen05_first_word,
    run_deep_probes as run_tcgen05_deep_probes,
    run_quick_probes as run_tcgen05_quick_probes,
    unique_bf16_words as tcgen05_bf16_unique_words,
    unique_f16_words as tcgen05_f16_unique_words,
    unique_f8f6f4_words as tcgen05_f8f6f4_unique_words,
    unique_i8_words as tcgen05_i8_unique_words,
    unique_fp8_words as tcgen05_fp8_unique_words,
    unique_nvfp4_words as tcgen05_nvfp4_unique_words,
    unique_words as tcgen05_unique_words,
)


def parse_hex_word(text: str) -> int:
    text = text.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    value = int(text, 16)
    if value < 0 or value > 0xFFFF_FFFF:
        raise argparse.ArgumentTypeError("expected uint32 hex word")
    return value


def parse_hex_words(text: str, count: int) -> tuple[int, ...]:
    words = tuple(parse_hex_word(part) for part in text.split(",") if part.strip())
    if len(words) != count:
        raise argparse.ArgumentTypeError(f"expected {count} comma-separated hex words")
    return words


def parse_ue4m3_words(text: str, count: int) -> tuple[int, ...]:
    words = parse_hex_words(text, count)
    for word in words:
        if word > 0x7F:
            raise argparse.ArgumentTypeError("expected 7-bit UE4M3 hex scale values")
    return words


def parse_ue8m0_words(text: str, count: int) -> tuple[int, ...]:
    words = parse_hex_words(text, count)
    for word in words:
        if word > 0xFF:
            raise argparse.ArgumentTypeError("expected 8-bit UE8M0 hex scale values")
    return words


def f16_word_class(bits: int) -> str:
    half = bits & 0xFFFF
    negative = (half & 0x8000) != 0
    exp = (half >> 10) & 0x1F
    frac = half & 0x03FF
    if exp == 0x1F:
        if frac:
            return "nan"
        return "-inf" if negative else "+inf"
    if exp == 0 and frac == 0:
        return "-0" if negative else "+0"
    if exp == 0:
        value = ((-1.0 if negative else 1.0) * frac) * (2.0 ** -24)
    else:
        value = ((-1.0 if negative else 1.0) * (1024 + frac)) * (2.0 ** (exp - 25))
    return f"{value:.9g}"


def _shape_result_class(bits: int, d_type: str) -> str:
    if d_type == "f16":
        return f16_word_class(bits)
    if d_type == "s32":
        return str(_i8_word_to_s32(bits))
    return f32_class(bits)


def cmd_quick(args: argparse.Namespace) -> int:
    harness = MmaHarness(args.runner, device=args.device)
    report = run_quick_probes(harness)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Runner:", report["runner_info"])
        print("Deterministic:", report["determinism"]["deterministic"])
        print("Position independent:", report["position_independence"]["position_independent"])
        pre = report["pretruncation"]
        print("A ignored mantissa bits:", pre["a_ignored_mantissa_bits"])
        print("B ignored mantissa bits:", pre["b_ignored_mantissa_bits"])
        print("NaN low-payload A:", pre["nan_low_payload_a"])
        print("NaN low-payload B:", pre["nan_low_payload_b"])
        print("Precision:", report["precision"])
        print("Rounding:", json.dumps(report["rounding"]["observed"], indent=2, sort_keys=True))
        print("Special values:", json.dumps(report["special_values"], indent=2, sort_keys=True))
        print("Swamping order:", json.dumps(report["swamping_order"], indent=2, sort_keys=True))
    return 0


def cmd_run_case(args: argparse.Namespace) -> int:
    harness = MmaHarness(args.runner, device=args.device)
    case = ProbeCase(parse_hex_words(args.a, 4), parse_hex_words(args.b, 4), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"lane {lane:02d}: " + " ".join(bit_hex(x) for x in regs))
    return 0


def cmd_random(args: argparse.Namespace) -> int:
    harness = MmaHarness(args.runner, device=args.device)
    rng = random.Random(args.seed)
    cases = [
        ProbeCase(
            tuple(rng.getrandbits(32) for _ in range(4)),
            tuple(rng.getrandbits(32) for _ in range(4)),
            rng.getrandbits(32),
        )
        for _ in range(args.cases)
    ]
    outputs = harness.run(cases)
    if args.model == "fixed":
        model = FixedAlignTf32MmaModel(nan_payload_policy="canonical")
    else:
        model = SimpleTf32MmaModel(canonical_nan=0x7FFF_FFFF, nan_payload_policy="canonical")
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        values = unique_words(out)
        if len(values) != 1:
            position_failures += 1
        observed = first_word(out)
        predicted = model.eval(case)
        if not model.matches(case, observed):
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "a": [bit_hex(x) for x in case.a],
                        "b": [bit_hex(x) for x in case.b],
                        "c": bit_hex(case.c),
                    }
                )
    report = {
        "cases": args.cases,
        "model": args.model,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    return _emit_validation_report(report)


def _random_f32_word(rng: random.Random, *, finite_only: bool) -> int:
    bits = rng.getrandbits(32)
    if finite_only and (bits & 0x7F80_0000) == 0x7F80_0000:
        bits = (bits & 0x807F_FFFF) | (rng.randrange(0, 255) << 23)
    return bits


def _make_tcgen05_model(name: str):
    if name == "hardware-oracle":
        return None
    if name == "exact-rne":
        return Tcgen05ExactTf32MmaModel(final_round="rne")
    if name == "exact-rz":
        return Tcgen05ExactTf32MmaModel(final_round="rz")
    if name == "fixed-rne":
        return FixedAlignTf32MmaModel(canonical_nan=0x7FFF_FFFF, nan_payload_policy="canonical", final_round="rne")
    if name == "fixed-rz":
        return FixedAlignTf32MmaModel(canonical_nan=0x7FFF_FFFF, nan_payload_policy="canonical", final_round="rz")
    if name == "raw-window-rz":
        return Tcgen05RawWindowTf32MmaModel()
    if name == "twostage-c23-rne":
        return Tcgen05ProductSumThenCModel(c_merge_fraction_bits=23, final_round="rne")
    if name == "twostage-c24-rne":
        return Tcgen05ProductSumThenCModel(c_merge_fraction_bits=24, final_round="rne")
    if name == "twostage-c24-rz":
        return Tcgen05ProductSumThenCModel(c_merge_fraction_bits=24, final_round="rz")
    if name == "twostage-p25-c25-rz":
        return Tcgen05ProductSumThenCModel(product_fraction_bits=25, c_merge_fraction_bits=25, final_round="rz")
    if name == "twostage-p26-c25-rz":
        return Tcgen05ProductSumThenCModel(product_fraction_bits=26, c_merge_fraction_bits=25, final_round="rz")
    if name == "twostage-p26-c26-rz":
        return Tcgen05ProductSumThenCModel(product_fraction_bits=26, c_merge_fraction_bits=26, final_round="rz")
    raise ValueError(f"unsupported tcgen05 model: {name}")


def _make_tcgen05_bf16_model(name: str):
    if name == "hardware-oracle":
        return None
    if name == "bf16-raw-window-rz":
        return Tcgen05RawWindowBf16MmaModel()
    raise ValueError(f"unsupported tcgen05 BF16 model: {name}")


def _make_tcgen05_f16_model(name: str):
    if name == "hardware-oracle":
        return None
    if name == "f16-raw-window-rz":
        return Tcgen05RawWindowF16MmaModel()
    raise ValueError(f"unsupported tcgen05 F16 model: {name}")


def _make_tcgen05_fp8_model(name: str):
    if name == "hardware-oracle":
        return None
    if name == "fp8e4m3-raw-window-rz":
        return Tcgen05RawWindowFp8E4M3MmaModel()
    raise ValueError(f"unsupported tcgen05 FP8 model: {name}")


def _make_tcgen05_f8f6f4_model(format_name: str, name: str):
    if name == "hardware-oracle":
        return None
    if name != "raw-window-rz":
        raise ValueError(f"unsupported tcgen05 f8f6f4 model: {name}")
    if format_name == "e5m2":
        return Tcgen05RawWindowFp8E5M2MmaModel()
    if format_name == "e2m3":
        return Tcgen05RawWindowFp6E2M3MmaModel()
    if format_name == "e3m2":
        return Tcgen05RawWindowFp6E3M2MmaModel()
    if format_name == "e2m1":
        return Tcgen05RawWindowFp4E2M1MmaModel()
    raise ValueError(f"unsupported tcgen05 f8f6f4 format: {format_name}")


def _make_tcgen05_nvfp4_model(name: str):
    if name == "hardware-oracle":
        return None
    if name in ("nvfp4-block16-window-rz", "nvfp4-raw-window-rz"):
        return Tcgen05RawWindowNvfp4MmaModel()
    raise ValueError(f"unsupported tcgen05 NVFP4 model: {name}")


def _make_tcgen05_shape_model(
    format_name: str,
    a_format_name: str | None = None,
    b_format_name: str | None = None,
    d_type: str = "f32",
    saturate: bool = False,
):
    a_format_name = a_format_name or ("u8" if format_name == "i8" else format_name)
    b_format_name = b_format_name or ("u8" if format_name == "i8" else format_name)
    if format_name == "i8":
        if d_type != "s32":
            raise ValueError("kind::i8 shape probes require S32 D")
        return Tcgen05I8S32MmaModel(a_type=a_format_name, b_type=b_format_name, saturate=saturate)
    if a_format_name != b_format_name or a_format_name != format_name or d_type != "f32":
        return Tcgen05MixedRawWindowMmaModel(a_format_name, b_format_name, d_type=d_type)
    if format_name == "tf32":
        return Tcgen05RawWindowTf32MmaModel()
    if format_name == "bf16":
        return Tcgen05RawWindowBf16MmaModel()
    if format_name == "f16":
        return Tcgen05RawWindowF16MmaModel()
    if format_name in ("fp8", "e4m3"):
        return Tcgen05RawWindowFp8E4M3MmaModel()
    if format_name == "e5m2":
        return Tcgen05RawWindowFp8E5M2MmaModel()
    if format_name == "e2m3":
        return Tcgen05RawWindowFp6E2M3MmaModel()
    if format_name == "e3m2":
        return Tcgen05RawWindowFp6E3M2MmaModel()
    if format_name == "e2m1":
        return Tcgen05RawWindowFp4E2M1MmaModel()
    raise ValueError(f"unsupported tcgen05 shape format: {format_name}")


def _shape_coordinate_arithmetic_model(format_name: str, a_format: str, b_format: str, d_type: str, saturate: bool):
    return _make_tcgen05_shape_model(format_name, a_format, b_format, d_type, saturate=saturate)


def _shape_coordinate_expected_word(
    format_name: str,
    a_format: str,
    b_format: str,
    d_type: str,
    saturate: bool,
    *,
    a_selected: bool,
    b_selected: bool,
) -> int:
    product_count = _shape_k(format_name)
    a_word = _shape_one_word(a_format) if a_selected else 0
    b_word = _shape_one_word(b_format) if b_selected else 0
    model = _shape_coordinate_arithmetic_model(format_name, a_format, b_format, d_type, saturate)
    return model.eval(
        SimpleNamespace(
            a=(a_word,) * product_count,
            b=(b_word,) * product_count,
            c=0,
        )
    )


def _coord_mix32(x: int) -> int:
    x &= 0xFFFF_FFFF
    x ^= x >> 16
    x = (x * 0x7FEB_352D) & 0xFFFF_FFFF
    x ^= x >> 15
    x = (x * 0x846C_A68B) & 0xFFFF_FFFF
    x ^= x >> 16
    return x & 0xFFFF_FFFF


def _coord_hash(seed: int, a: int, b: int, c: int, tag: int) -> int:
    x = (seed ^ 0x9E37_79B9) & 0xFFFF_FFFF
    x ^= (a * 0x85EB_CA6B) & 0xFFFF_FFFF
    x ^= (b * 0xC2B2_AE35) & 0xFFFF_FFFF
    x ^= (c * 0x27D4_EB2D) & 0xFFFF_FFFF
    x ^= (tag * 0x1656_67B1) & 0xFFFF_FFFF
    return _coord_mix32(x)


def _coordinate_random_f32_word(h: int) -> int:
    return (0x3F80_0000, 0xBF80_0000, 0x4000_0000, 0xC000_0000, 0x3F00_0000, 0xBF00_0000)[h % 6]


def _coordinate_random_f16_word(h: int) -> int:
    return (0x3C00, 0xBC00, 0x4000, 0xC000, 0x3800, 0xB800)[h % 6]


def _coordinate_random_i8_word(h: int, operand_format: str) -> int:
    if operand_format == "s8":
        return (-3, -2, -1, 1, 2, 3, 0)[h % 7] & 0xFF
    return (h % 7) + 1


def _coordinate_random_f8f6f4_word(format_name: str, h: int) -> int:
    slot = h % 6
    if format_name == "e5m2":
        return (0x3C, 0xBC, 0x40, 0xC0, 0x38, 0xB8)[slot]
    if format_name == "e2m3":
        return (0x08, 0x28, 0x10, 0x30, 0x04, 0x24)[slot]
    if format_name == "e3m2":
        return (0x0C, 0x2C, 0x10, 0x30, 0x08, 0x28)[slot]
    if format_name == "e2m1":
        return (0x02, 0x0A, 0x04, 0x0C, 0x01, 0x09)[slot]
    return (0x38, 0xB8, 0x40, 0xC0, 0x30, 0xB0)[slot]


def _coordinate_random_operand_word(
    format_name: str,
    operand_format: str,
    seed: int,
    row_or_col: int,
    k: int,
    *,
    matrix_a: bool,
) -> int:
    tag = 0xA341_316C if matrix_a else 0xC801_3EA4
    h = _coord_hash(seed, row_or_col, k, _shape_operand_code(format_name, operand_format), tag)
    if format_name == "i8":
        return _coordinate_random_i8_word(h, operand_format)
    if operand_format in ("tf32", "bf16"):
        return _coordinate_random_f32_word(h)
    if operand_format == "f16":
        return _coordinate_random_f16_word(h)
    return _coordinate_random_f8f6f4_word(operand_format, h)


def _coordinate_random_c_word(d_type: str, seed: int, group: int, thread: int, reg: int) -> int:
    h = _coord_hash(seed, group, thread, reg, 0xD1B5_4A32)
    if d_type == "f16":
        return _coordinate_random_f16_word(h)
    if d_type == "s32":
        return (-17, -9, -3, 0, 5, 11, 19, -23, 29)[h % 9] & 0xFFFF_FFFF
    return _coordinate_random_f32_word(h)


def _shape_operand_code(format_name: str, operand_format: str) -> int:
    if format_name == "i8":
        return {"u8": 0, "s8": 1}[operand_format]
    return {
        "tf32": 0,
        "bf16": 1,
        "f16": 2,
        "fp8": 3,
        "e4m3": 3,
        "e5m2": 4,
        "e2m3": 5,
        "e3m2": 6,
        "e2m1": 7,
    }[operand_format]


def _shape_coordinate_model(
    format_name: str,
    a_format: str,
    b_format: str,
    d_type: str,
    saturate: bool,
    *,
    sparse: bool,
    cta_group: int,
):
    if format_name == "e2m1" and sparse and cta_group == 1:
        return _Tcgen05ShapeSparseFp4E2M1Cg1Model()
    if sparse and cta_group == 2 and format_name in ("fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1"):
        return _Tcgen05ShapeSparseF8F6F4Cg2Model(a_format, b_format, d_type=d_type)
    return _make_tcgen05_shape_model(format_name, a_format, b_format, d_type, saturate=saturate)


def _shape_coordinate_physical_position(thread: int, cta_group: int) -> tuple[int, int]:
    base_threads = CTA_GROUP2_THREADS if cta_group == 2 else THREADS
    return thread // base_threads, thread % base_threads


class _Tcgen05ShapeSparseFp4E2M1Cg1Model:
    def __init__(self):
        self._base = Tcgen05RawWindowFp4E2M1MmaModel()

    @staticmethod
    def _pairs_for_metadata(metadata: int) -> tuple[tuple[int, int], ...]:
        pairs = []
        for chunk in range(8):
            selector = (metadata >> (chunk * 4)) & 0xF
            for selector_index in (selector & 0x3, (selector >> 2) & 0x3):
                a_idx = chunk * 4 + selector_index
                pairs.append((a_idx, a_idx))
                pairs.append((a_idx, a_idx + (16 if chunk == 0 else 32)))
        return tuple(pairs)

    def eval(self, case) -> int:
        pairs = self._pairs_for_metadata(case.metadata)
        return self._base.eval(
            SimpleNamespace(
                a=tuple(case.a[a_idx] for a_idx, _b_idx in pairs),
                b=tuple(case.b[b_idx] for _a_idx, b_idx in pairs),
                c=case.c,
            )
        )

    def matches(self, case, observed: int) -> bool:
        return self.eval(case) == observed


class _Tcgen05ShapeSparseF8F6F4Cg2Model:
    def __init__(self, a_format: str, b_format: str, *, d_type: str):
        self.a_format = a_format
        self.b_format = b_format
        self._base = Tcgen05MixedRawWindowMmaModel(a_format, b_format, d_type=d_type)

    @staticmethod
    def _sparse_index_for_packed(metadata: int, metadata_hi: int, packed_k: int) -> int:
        chunk = packed_k >> 1
        word = metadata if chunk < 8 else metadata_hi
        selector = (word >> ((chunk & 7) * 4)) & 0xF
        selector_index = selector & 0x3 if (packed_k & 1) == 0 else (selector >> 2) & 0x3
        return chunk * 4 + selector_index

    def eval(self, case) -> int:
        a_words = []
        b_words = []
        # Layout A/C sparse cta_group::2 exposes a first-window scalar slice:
        # 32 packed sparse positions, with B indexed by the selected A lane.
        for packed_k in range(32):
            a_idx = self._sparse_index_for_packed(case.metadata, case.metadata_hi, packed_k)
            a_words.append(case.a[a_idx])
            b_words.append(case.b[a_idx])
        return self._base.eval(SimpleNamespace(a=tuple(a_words), b=tuple(b_words), c=case.c))

    def matches(self, case, observed: int) -> bool:
        return self.eval(case) == observed


def _make_tcgen05_mxf8f6f4_model(format_name: str, name: str, a_format: str | None = None, b_format: str | None = None):
    if name == "hardware-oracle":
        return None
    if name != "block-scaled-window-rz":
        raise ValueError(f"unsupported tcgen05 mxf8f6f4 model: {name}")
    a_format = format_name if a_format is None else a_format
    b_format = format_name if b_format is None else b_format
    if a_format != format_name or b_format != format_name:
        return Tcgen05BlockScaledMmaModel(
            input_decoder=_tcgen05_decoder_for_format(a_format),
            input_decoder_b=_tcgen05_decoder_for_format(b_format),
            scale_block_size=32,
            c_merge_group_size=32,
            product_window_fraction_bits=25,
            scale_floor_offset=1000,
            use_c_dominant_merge=False,
        )
    if format_name == "e4m3":
        return Tcgen05BlockScaledMxf8f6f4E4M3MmaModel()
    if format_name == "e5m2":
        return Tcgen05BlockScaledMxf8f6f4E5M2MmaModel()
    if format_name == "e2m3":
        return Tcgen05BlockScaledMxf8f6f4E2M3MmaModel()
    if format_name == "e3m2":
        return Tcgen05BlockScaledMxf8f6f4E3M2MmaModel()
    if format_name == "e2m1":
        return Tcgen05BlockScaledMxf8f6f4E2M1MmaModel()
    raise ValueError(f"unsupported tcgen05 mxf8f6f4 format: {format_name}")


def _make_tcgen05_block_scaled_model(kind: str, name: str):
    if name == "hardware-oracle":
        return None
    if name != "block-scaled-window-rz":
        raise ValueError(f"unsupported tcgen05 block-scaled model: {name}")
    if kind == "mxf4nvf4-4x-ue4m3":
        return Tcgen05RawWindowNvfp4MmaModel()
    if kind == "mxf4":
        return Tcgen05BlockScaledMxf4E2M1MmaModel()
    if kind == "mxf4nvf4-2x":
        return Tcgen05BlockScaledMxf4Nvfp4E2M1Scale2XMmaModel()
    if kind == "mxf4nvf4-4x":
        return Tcgen05BlockScaledMxf4Nvfp4E2M1Scale4XMmaModel()
    raise ValueError(f"unsupported tcgen05 block-scaled kind: {kind}")


def _run_tcgen05_random_validation(args: argparse.Namespace) -> dict:
    harness = Tcgen05Harness(_tcgen05_runner(args), device=args.device)
    rng = random.Random(args.seed)
    cases = [
        Tcgen05ProbeCase(
            tuple(_random_f32_word(rng, finite_only=args.finite_only) for _ in range(8)),
            tuple(_random_f32_word(rng, finite_only=args.finite_only) for _ in range(8)),
            _random_f32_word(rng, finite_only=args.finite_only),
        )
        for _ in range(args.cases)
    ]
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_model(args.model)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = tcgen05_first_word(out)
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff
    model_kind = "known-good TF32 software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    elif args.model != "raw-window-rz":
        model_kind = "software hypothesis"
    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "model": args.model,
        "model_kind": model_kind,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _run_tcgen05_bf16_random_validation(args: argparse.Namespace) -> dict:
    harness = Tcgen05Bf16Harness(_tcgen05_bf16_runner(args), device=args.device)
    rng = random.Random(args.seed)
    cases = [
        Tcgen05Bf16ProbeCase(
            tuple(_random_f32_word(rng, finite_only=args.finite_only) for _ in range(16)),
            tuple(_random_f32_word(rng, finite_only=args.finite_only) for _ in range(16)),
            _random_f32_word(rng, finite_only=args.finite_only),
        )
        for _ in range(args.cases)
    ]
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_bf16_model(args.model)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_bf16_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = tcgen05_first_word(out)
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_bf16_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff
    model_kind = "BF16 software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "model": args.model,
        "model_kind": model_kind,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _random_f16_word(rng: random.Random, *, finite_only: bool) -> int:
    bits = rng.randrange(1 << 16)
    if finite_only and ((bits >> 10) & 0x1F) == 0x1F:
        bits = (bits & ~(0x1F << 10)) | (rng.randrange(0x1F) << 10)
    return bits


def _run_tcgen05_f16_random_validation(args: argparse.Namespace) -> dict:
    harness = Tcgen05F16Harness(_tcgen05_f16_runner(args), device=args.device)
    rng = random.Random(args.seed)
    cases = [
        Tcgen05F16ProbeCase(
            tuple(_random_f16_word(rng, finite_only=args.finite_only) for _ in range(16)),
            tuple(_random_f16_word(rng, finite_only=args.finite_only) for _ in range(16)),
            _random_f32_word(rng, finite_only=args.finite_only),
        )
        for _ in range(args.cases)
    ]
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_f16_model(args.model)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_f16_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_f16_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff
    model_kind = "F16 software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "model": args.model,
        "model_kind": model_kind,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _random_fp8_word(rng: random.Random, *, finite_only: bool) -> int:
    bits = rng.randrange(256)
    if finite_only and (bits & 0x7F) == 0x7F:
        bits = (bits & 0x80) | rng.randrange(0x7F)
    return bits


def _run_tcgen05_fp8_random_validation(args: argparse.Namespace) -> dict:
    harness = Tcgen05Fp8Harness(_tcgen05_fp8_runner(args), device=args.device)
    rng = random.Random(args.seed)
    cases = [
        Tcgen05Fp8ProbeCase(
            tuple(_random_fp8_word(rng, finite_only=args.finite_only) for _ in range(32)),
            tuple(_random_fp8_word(rng, finite_only=args.finite_only) for _ in range(32)),
            _random_f32_word(rng, finite_only=args.finite_only),
        )
        for _ in range(args.cases)
    ]
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_fp8_model(args.model)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_fp8_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = tcgen05_first_word(out)
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_fp8_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff
    model_kind = "FP8 E4M3 software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "model": args.model,
        "model_kind": model_kind,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _random_f8f6f4_word(format_name: str, rng: random.Random, *, finite_only: bool) -> int:
    if format_name == "e4m3":
        return _random_fp8_word(rng, finite_only=finite_only)
    if format_name == "e5m2":
        bits = rng.randrange(256)
        if finite_only and ((bits >> 2) & 0x1F) == 0x1F:
            bits = (bits & ~(0x1F << 2)) | (rng.randrange(0x1F) << 2)
        return bits
    if format_name in ("e2m3", "e3m2"):
        return rng.randrange(1 << 6)
    if format_name == "e2m1":
        return rng.randrange(1 << 4)
    raise ValueError(f"unsupported tcgen05 f8f6f4 format: {format_name}")


def _run_tcgen05_f8f6f4_random_validation(args: argparse.Namespace) -> dict:
    harness = Tcgen05F8f6f4Harness(
        _tcgen05_f8f6f4_runner(args),
        format_name=args.format,
        device=args.device,
    )
    rng = random.Random(args.seed)
    cases = [
        Tcgen05F8f6f4ProbeCase(
            tuple(_random_f8f6f4_word(args.format, rng, finite_only=args.finite_only) for _ in range(32)),
            tuple(_random_f8f6f4_word(args.format, rng, finite_only=args.finite_only) for _ in range(32)),
            _random_f32_word(rng, finite_only=args.finite_only),
        )
        for _ in range(args.cases)
    ]
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_f8f6f4_model(args.format, args.model)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_f8f6f4_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_f8f6f4_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff
    model_kind = f"{args.format.upper()} software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "format": args.format,
        "model": args.model,
        "model_kind": model_kind,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _i8_word_to_s32(bits: int) -> int:
    bits &= 0xFFFF_FFFF
    return bits - 0x1_0000_0000 if bits & 0x8000_0000 else bits


def _random_i8_word(rng: random.Random) -> int:
    return rng.randrange(256)


def _i8_positive_byte(kind: str) -> int:
    return 0x7F if kind == "s8" else 0xFF


def _i8_negative_byte(kind: str) -> int | None:
    return 0x80 if kind == "s8" else None


def _i8_edge_cases(a_type: str, b_type: str) -> list[Tcgen05I8ProbeCase]:
    pos_a = _i8_positive_byte(a_type)
    pos_b = _i8_positive_byte(b_type)
    neg_a = _i8_negative_byte(a_type)
    neg_b = _i8_negative_byte(b_type)
    cases = [
        Tcgen05I8ProbeCase((0,) * 32, (0,) * 32, 0),
        Tcgen05I8ProbeCase((1,) * 32, (1,) * 32, 0),
        Tcgen05I8ProbeCase((pos_a,) * 32, (pos_b,) * 32, 0),
        Tcgen05I8ProbeCase((pos_a,) * 32, (pos_b,) * 32, 0x7FFF_FFF0),
        Tcgen05I8ProbeCase((pos_a,) * 32, (pos_b,) * 32, 0x8000_0010),
    ]
    if neg_a is not None:
        cases.append(Tcgen05I8ProbeCase((neg_a,) * 32, (pos_b,) * 32, 0x8000_0010))
        cases.append(
            Tcgen05I8ProbeCase((pos_a,) * 16 + (neg_a,) * 16, (pos_b,) * 32, 0x7FFF_FFF0)
        )
    if neg_b is not None:
        cases.append(Tcgen05I8ProbeCase((pos_a,) * 32, (neg_b,) * 32, 0x8000_0010))
        cases.append(
            Tcgen05I8ProbeCase((pos_a,) * 32, (pos_b,) * 16 + (neg_b,) * 16, 0x7FFF_FFF0)
        )
    return cases


def _run_tcgen05_i8_random_validation(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    harness = Tcgen05I8Harness(
        _tcgen05_i8_runner(args),
        a_type=args.a_type,
        b_type=args.b_type,
        saturate=args.saturate,
        device=args.device,
    )
    rng = random.Random(args.seed)
    edge_cases = _i8_edge_cases(args.a_type, args.b_type)
    model = None
    if args.model != "hardware-oracle":
        model = Tcgen05I8S32MmaModel(a_type=args.a_type, b_type=args.b_type, saturate=args.saturate)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    valid_words_per_case = 0
    checked_cases = 0
    while checked_cases < args.cases:
        batch_count = min(args.batch_size, args.cases - checked_cases)
        batch_cases: list[Tcgen05I8ProbeCase] = []
        for case_idx in range(checked_cases, checked_cases + batch_count):
            if case_idx < len(edge_cases):
                batch_cases.append(edge_cases[case_idx])
            else:
                batch_cases.append(
                    Tcgen05I8ProbeCase(
                        tuple(_random_i8_word(rng) for _ in range(32)),
                        tuple(_random_i8_word(rng) for _ in range(32)),
                        rng.getrandbits(32),
                    )
                )
        outputs = harness.run(batch_cases)
        model_outputs = harness.run(batch_cases) if args.model == "hardware-oracle" else None
        for batch_idx, (case, out) in enumerate(zip(batch_cases, outputs)):
            idx = checked_cases + batch_idx
            valid_words = tcgen05_i8_flatten_output(out)
            valid_words_per_case = len(valid_words)
            values = set(valid_words)
            if len(values) != 1:
                position_failures += 1
            observed = valid_words[0]
            first_diff = None
            if model_outputs is None:
                assert model is not None
                predicted = model.eval(case)
                matched = all(word == predicted for word in valid_words)
                if not matched:
                    first_diff = next(
                        (
                            {
                                "valid_word_index": word_idx,
                                "observed": tcgen05_bit_hex(observed_word),
                                "model": tcgen05_bit_hex(predicted),
                            }
                            for word_idx, observed_word in enumerate(valid_words)
                            if observed_word != predicted
                        ),
                        None,
                    )
            else:
                model_valid_words = tcgen05_i8_flatten_output(model_outputs[batch_idx])
                predicted = model_valid_words[0]
                if len(set(model_valid_words)) != 1:
                    oracle_position_failures += 1
                matched = model_valid_words == valid_words
                if not matched:
                    first_diff = next(
                        (
                            {
                                "valid_word_index": word_idx,
                                "observed": tcgen05_bit_hex(observed_word),
                                "model": tcgen05_bit_hex(model_word),
                            }
                            for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                            if observed_word != model_word
                        ),
                        None,
                    )
            if not matched:
                mismatch_count += 1
                if len(mismatches) < args.show:
                    mismatches.append(
                        {
                            "case": idx,
                            "observed": tcgen05_bit_hex(observed),
                            "observed_s32": _i8_word_to_s32(observed),
                            "model": tcgen05_bit_hex(predicted),
                            "model_s32": _i8_word_to_s32(predicted),
                            "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                            "a": [tcgen05_bit_hex(x) for x in case.a],
                            "b": [tcgen05_bit_hex(x) for x in case.b],
                            "c": tcgen05_bit_hex(case.c),
                            "c_s32": _i8_word_to_s32(case.c),
                        }
                    )
                    if first_diff is not None:
                        mismatches[-1]["first_valid_word_diff"] = first_diff
        checked_cases += batch_count
        if mismatch_count:
            break
    model_kind = "S32 wrap/clamp software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    report = {
        "a_type": args.a_type,
        "b_type": args.b_type,
        "batch_size": args.batch_size,
        "cases": args.cases,
        "checked_cases": checked_cases,
        "matched_output_words": (checked_cases - mismatch_count) * valid_words_per_case,
        "model": args.model,
        "model_kind": model_kind,
        "saturate": args.saturate,
        "seed": args.seed,
        "valid_words_per_case": valid_words_per_case,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _shape_one_word(format_name: str) -> int:
    if format_name in ("i8", "u8", "s8"):
        return 1
    if format_name in ("tf32", "bf16"):
        return 0x3F80_0000
    if format_name == "f16":
        return 0x3C00
    if format_name in ("fp8", "e4m3"):
        return 0x38
    if format_name == "e5m2":
        return 0x3C
    if format_name == "e2m3":
        return 0x08
    if format_name == "e3m2":
        return 0x0C
    if format_name == "e2m1":
        return 0x02
    raise ValueError(f"unsupported tcgen05 shape format: {format_name}")


def _random_shape_word(format_name: str, rng: random.Random, *, finite_only: bool) -> int:
    if format_name in ("i8", "u8", "s8"):
        return _random_i8_word(rng)
    if format_name in ("tf32", "bf16"):
        return _random_f32_word(rng, finite_only=finite_only)
    if format_name == "f16":
        return _random_f16_word(rng, finite_only=finite_only)
    if format_name == "fp8":
        return _random_fp8_word(rng, finite_only=finite_only)
    return _random_f8f6f4_word(format_name, rng, finite_only=finite_only)


def _shape_k(format_name: str) -> int:
    if format_name == "tf32":
        return 8
    if format_name in ("bf16", "f16"):
        return 16
    return 32


def _shape_sparse_k(format_name: str) -> int:
    if format_name == "tf32":
        return 16
    if format_name in ("bf16", "f16"):
        return 32
    return 64


def _shape_active_sparse_indices(
    format_name: str,
    metadata: int = SPARSE_FIXED_METADATA,
    metadata_hi: int | None = None,
) -> tuple[int, ...]:
    return tuple(sorted(_sparse_active_indices(format_name, metadata, metadata_hi)))


def _shape_logical_k(format_name: str, *, sparse: bool = False) -> int:
    return _shape_sparse_k(format_name) if sparse else _shape_k(format_name)


def _shape_case_from_summands(
    format_name: str,
    m: int,
    n: int,
    summands: list[int],
    *,
    ws: bool = False,
    cta_group: int = 1,
    sparse: bool = False,
    a_format_name: str | None = None,
    b_format_name: str | None = None,
    d_type: str = "f32",
    saturate: bool = False,
    metadata: int = SPARSE_FIXED_METADATA,
    metadata_hi: int = SPARSE_FIXED_METADATA,
) -> Tcgen05ShapeProbeCase:
    a_format_name = a_format_name or ("u8" if format_name == "i8" else format_name)
    b_format_name = b_format_name or ("u8" if format_name == "i8" else format_name)
    k = _shape_logical_k(format_name, sparse=sparse)
    if len(summands) != k + 1:
        raise ValueError(f"expected {k} products plus c")
    one = _shape_one_word(b_format_name)
    a = list(summands[:k])
    if sparse:
        active = set(_shape_active_sparse_indices(format_name, metadata, metadata_hi))
        a = [value if idx in active else 0 for idx, value in enumerate(a)]
    return Tcgen05ShapeProbeCase(
        format_name,
        m,
        n,
        tuple(a),
        (one,) * k,
        summands[k],
        ws,
        cta_group,
        sparse,
        a_format_name,
        b_format_name,
        d_type,
        saturate,
        metadata,
        metadata_hi,
    )


def _shape_valid_positions(
    harness: Tcgen05ShapeHarness,
    format_name: str,
    m: int,
    n: int,
    model,
    *,
    ws: bool = False,
    cta_group: int = 1,
    sparse: bool = False,
    a_format_name: str | None = None,
    b_format_name: str | None = None,
    d_type: str = "f32",
    saturate: bool = False,
    metadata: int = SPARSE_FIXED_METADATA,
    metadata_hi: int = SPARSE_FIXED_METADATA,
) -> tuple[tuple[int, int], ...]:
    a_format_name = a_format_name or ("u8" if format_name == "i8" else format_name)
    b_format_name = b_format_name or ("u8" if format_name == "i8" else format_name)
    k = _shape_logical_k(format_name, sparse=sparse)
    zero = 0
    a_one = _shape_one_word(a_format_name)
    if d_type == "f16":
        c_one, c_two, c_three = 0x3C00, 0x4000, 0x4200
    elif d_type == "s32":
        c_one, c_two, c_three = 1, 2, 3
    else:
        c_one, c_two, c_three = 0x3F80_0000, 0x4000_0000, 0x4040_0000
    active_indices = _shape_active_sparse_indices(format_name, metadata, metadata_hi) if sparse else tuple(range(k))
    second_active = active_indices[1] if len(active_indices) > 1 else active_indices[0]
    calibration_cases = [
        _shape_case_from_summands(
            format_name,
            m,
            n,
            [zero] * k + [c_one],
            ws=ws,
            cta_group=cta_group,
            sparse=sparse,
            a_format_name=a_format_name,
            b_format_name=b_format_name,
            d_type=d_type,
            saturate=saturate,
            metadata=metadata,
            metadata_hi=metadata_hi,
        ),
        _shape_case_from_summands(
            format_name,
            m,
            n,
            [a_one] * k + [zero],
            ws=ws,
            cta_group=cta_group,
            sparse=sparse,
            a_format_name=a_format_name,
            b_format_name=b_format_name,
            d_type=d_type,
            saturate=saturate,
            metadata=metadata,
            metadata_hi=metadata_hi,
        ),
        _shape_case_from_summands(
            format_name,
            m,
            n,
            [a_one] + [zero] * (k - 1) + [c_two],
            ws=ws,
            cta_group=cta_group,
            sparse=sparse,
            a_format_name=a_format_name,
            b_format_name=b_format_name,
            d_type=d_type,
            saturate=saturate,
            metadata=metadata,
            metadata_hi=metadata_hi,
        ),
        _shape_case_from_summands(
            format_name,
            m,
            n,
            [a_one if idx == second_active else zero for idx in range(k)] + [c_three],
            ws=ws,
            cta_group=cta_group,
            sparse=sparse,
            a_format_name=a_format_name,
            b_format_name=b_format_name,
            d_type=d_type,
            saturate=saturate,
            metadata=metadata,
            metadata_hi=metadata_hi,
        ),
    ]
    format_index = {
        "tf32": 0,
        "bf16": 1,
        "f16": 2,
        "fp8": 3,
        "e4m3": 3,
        "e5m2": 4,
        "e2m3": 5,
        "e3m2": 6,
        "e2m1": 7,
        "i8": 8,
    }[format_name]
    rng = random.Random(0x05A9_0000 + format_index * 100_000 + m * 257 + n * 13 + (17 if sparse else 0))
    for _ in range(12):
        a = tuple(_random_shape_word(a_format_name, rng, finite_only=True) for _ in range(k))
        if sparse:
            active = set(active_indices)
            a = tuple(value if idx in active else 0 for idx, value in enumerate(a))
        calibration_cases.append(
            Tcgen05ShapeProbeCase(
                format_name,
                m,
                n,
                a,
                tuple(_random_shape_word(b_format_name, rng, finite_only=True) for _ in range(k)),
                _random_f16_word(rng, finite_only=True)
                if d_type == "f16"
                else (rng.getrandbits(32) if d_type == "s32" else _random_f32_word(rng, finite_only=True)),
                ws,
                cta_group,
                sparse,
                a_format_name,
                b_format_name,
                d_type,
                saturate,
                metadata,
                metadata_hi,
            )
        )
    expected = [model.eval(case) for case in calibration_cases]
    if sparse and ws:
        outputs = [harness.run([case])[0] for case in calibration_cases]
    else:
        outputs = harness.run(calibration_cases)
    positions: list[tuple[int, int]] = []
    for thread in range(len(outputs[0])):
        for reg in range(4):
            if all(out[thread][reg] == want for out, want in zip(outputs, expected)):
                positions.append((thread, reg))
    return tuple(positions)


def _shape_words(case_output: list[list[int]], positions: tuple[tuple[int, int], ...]) -> list[int]:
    return [case_output[thread][reg] for thread, reg in positions]


def _shape_coordinate_case(
    format_name: str,
    m: int,
    n: int,
    *,
    ws: bool,
    cta_group: int,
    sparse: bool,
    a_format_name: str,
    b_format_name: str,
    d_type: str,
    saturate: bool,
    metadata: int,
    metadata_hi: int,
    coordinate: str | None,
    bit: int = 0,
) -> Tcgen05ShapeProbeCase:
    k = _shape_logical_k(format_name, sparse=sparse)
    if coordinate is None:
        a = tuple(_shape_one_word(a_format_name) for _ in range(k))
        b = tuple(_shape_one_word(b_format_name) for _ in range(k))
        c = 0
    else:
        a = (0,) * k
        b = (0,) * k
        c = bit
    return Tcgen05ShapeProbeCase(
        format_name,
        m,
        n,
        a,
        b,
        c,
        ws,
        cta_group,
        sparse,
        a_format_name,
        b_format_name,
        d_type,
        saturate,
        metadata,
        metadata_hi,
        coordinate,
    )


def _run_tcgen05_shape_coordinate_map(args: argparse.Namespace) -> dict:
    harness = Tcgen05ShapeHarness(_tcgen05_shape_runner(args), device=args.device)
    a_format = args.a_format or ("u8" if args.format == "i8" else args.format)
    b_format = args.b_format or ("u8" if args.format == "i8" else args.format)
    d_type = args.d_type or ("s32" if args.format == "i8" else "f32")
    saturate = getattr(args, "saturate", False)
    random_payload_arg = getattr(args, "random_payload_cases", None)
    random_payload_cases = (
        2 if random_payload_arg is None and getattr(args, "require_full", False)
        else (0 if random_payload_arg is None else random_payload_arg)
    )
    if random_payload_cases < 0:
        raise ValueError("--random-payload-cases must be nonnegative")
    if args.format != "i8" and saturate:
        raise ValueError("--saturate is only valid with --format i8")
    if args.format in ("bf16", "f16") and {a_format, b_format} == {"bf16", "f16"}:
        raise ValueError("B200 rejects off-diagonal BF16/F16 kind::f16 A/B descriptors with an illegal instruction")
    if args.format in ("bf16", "f16") and d_type == "f16" and (a_format != "f16" or b_format != "f16"):
        raise ValueError("B200 kind::f16 F16-D shape probes require F16 A/B inputs")
    metadata = SPARSE_FIXED_METADATA
    metadata_hi = SPARSE_FIXED_METADATA
    if args.sparse:
        rng = random.Random(args.seed)
        metadata, metadata_hi = _sparse_metadata_from_args(args, rng, args.format)

    row_bits = max(1, (args.m - 1).bit_length())
    col_bits = max(1, (args.n - 1).bit_length())
    cases = [
        _shape_coordinate_case(
            args.format,
            args.m,
            args.n,
            ws=args.ws,
            cta_group=args.cta_group,
            sparse=args.sparse,
            a_format_name=a_format,
            b_format_name=b_format,
            d_type=d_type,
            saturate=saturate,
            metadata=metadata,
            metadata_hi=metadata_hi,
            coordinate="all",
        )
    ]
    for bit in range(row_bits):
        cases.append(
            _shape_coordinate_case(
                args.format,
                args.m,
                args.n,
                ws=args.ws,
                cta_group=args.cta_group,
                sparse=args.sparse,
                a_format_name=a_format,
                b_format_name=b_format,
                d_type=d_type,
                saturate=saturate,
                metadata=metadata,
                metadata_hi=metadata_hi,
                coordinate="row",
                bit=bit,
            )
        )
    for bit in range(col_bits):
        cases.append(
            _shape_coordinate_case(
                args.format,
                args.m,
                args.n,
                ws=args.ws,
                cta_group=args.cta_group,
                sparse=args.sparse,
                a_format_name=a_format,
                b_format_name=b_format,
                d_type=d_type,
                saturate=saturate,
                metadata=metadata,
                metadata_hi=metadata_hi,
                coordinate="col",
                bit=bit,
            )
        )

    outputs = harness.run(cases)
    active_output = outputs[0]
    row_outputs = outputs[1 : 1 + row_bits]
    col_outputs = outputs[1 + row_bits :]
    position_map: dict[tuple[int, int], tuple[int, int]] = {}
    coords_to_positions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    out_of_range = []
    arithmetic_mismatches = []
    arithmetic_mismatch_count = 0
    expected_cache: dict[tuple[bool, bool], int] = {}

    for thread in range(len(active_output)):
        for reg in range(4):
            if active_output[thread][reg] == 0:
                continue
            row = 0
            for bit, out in enumerate(row_outputs):
                if out[thread][reg] != 0:
                    row |= 1 << bit
            col = 0
            for bit, out in enumerate(col_outputs):
                if out[thread][reg] != 0:
                    col |= 1 << bit
            position = (thread, reg)
            coordinate = (row, col)
            position_map[position] = coordinate
            if row >= args.m or col >= args.n:
                if len(out_of_range) < args.show:
                    out_of_range.append({"position": f"t{thread}:r{reg}", "coordinate": [row, col]})
                continue
            coords_to_positions.setdefault(coordinate, []).append(position)
            probes = [("all", None, active_output, True, True)]
            probes.extend(
                ("row", bit, out, ((row >> bit) & 1) != 0, True)
                for bit, out in enumerate(row_outputs)
            )
            probes.extend(
                ("col", bit, out, True, ((col >> bit) & 1) != 0)
                for bit, out in enumerate(col_outputs)
            )
            for probe_kind, bit, out, a_selected, b_selected in probes:
                key = (a_selected, b_selected)
                if key not in expected_cache:
                    expected_cache[key] = _shape_coordinate_expected_word(
                        args.format,
                        a_format,
                        b_format,
                        d_type,
                        saturate,
                        a_selected=a_selected,
                        b_selected=b_selected,
                    )
                expected = expected_cache[key]
                observed = out[thread][reg]
                if observed != expected:
                    arithmetic_mismatch_count += 1
                    if len(arithmetic_mismatches) < args.show:
                        arithmetic_mismatches.append(
                            {
                                "position": f"t{thread}:r{reg}",
                                "coordinate": [row, col],
                                "probe": probe_kind if bit is None else f"{probe_kind}{bit}",
                                "observed": tcgen05_bit_hex(observed),
                                "expected": tcgen05_bit_hex(expected),
                                "observed_class": _shape_result_class(observed, d_type),
                                "expected_class": _shape_result_class(expected, d_type),
                            }
                        )

    expected_coordinates = args.m * args.n
    unique_coordinates = len(coords_to_positions)
    duplicate_coordinates = {
        coord: positions for coord, positions in coords_to_positions.items() if len(positions) > 1
    }
    missing_sample = []
    if unique_coordinates < expected_coordinates:
        for row in range(args.m):
            for col in range(args.n):
                if (row, col) not in coords_to_positions:
                    missing_sample.append([row, col])
                    if len(missing_sample) >= args.show:
                        break
            if len(missing_sample) >= args.show:
                break
    random_payload_mismatches = []
    random_payload_mismatch_count = 0
    random_payload_checked_words = 0
    random_payload_seeds = [
        (args.seed ^ 0xC001_D00D ^ (case_idx * 0x9E37_79B9)) & 0xFFFF_FFFF
        for case_idx in range(random_payload_cases)
    ]
    if random_payload_seeds:
        random_cases = [
            _shape_coordinate_case(
                args.format,
                args.m,
                args.n,
                ws=args.ws,
                cta_group=args.cta_group,
                sparse=args.sparse,
                a_format_name=a_format,
                b_format_name=b_format,
                d_type=d_type,
                saturate=saturate,
                metadata=metadata,
                metadata_hi=metadata_hi,
                coordinate="random",
                bit=seed,
            )
            for seed in random_payload_seeds
        ]
        random_outputs = harness.run(random_cases)
        random_model = _shape_coordinate_model(
            args.format,
            a_format,
            b_format,
            d_type,
            saturate,
            sparse=args.sparse,
            cta_group=args.cta_group,
        )
        for case_idx, (seed, out) in enumerate(zip(random_payload_seeds, random_outputs)):
            active = set(_shape_active_sparse_indices(args.format, metadata, metadata_hi)) if args.sparse else None
            a_cache: dict[int, tuple[int, ...]] = {}
            b_cache: dict[int, tuple[int, ...]] = {}

            def a_for_row(row: int) -> tuple[int, ...]:
                if row not in a_cache:
                    values = tuple(
                        _coordinate_random_operand_word(args.format, a_format, seed, row, idx, matrix_a=True)
                        for idx in range(_shape_logical_k(args.format, sparse=args.sparse))
                    )
                    if active is not None:
                        values = tuple(value if idx in active else 0 for idx, value in enumerate(values))
                    a_cache[row] = values
                return a_cache[row]

            def b_for_col(col: int) -> tuple[int, ...]:
                if col not in b_cache:
                    b_cache[col] = tuple(
                        _coordinate_random_operand_word(args.format, b_format, seed, col, idx, matrix_a=False)
                        for idx in range(_shape_logical_k(args.format, sparse=args.sparse))
                    )
                return b_cache[col]

            for (thread, reg), (row, col) in position_map.items():
                if row >= args.m or col >= args.n:
                    continue
                group, physical_thread = _shape_coordinate_physical_position(thread, args.cta_group)
                c_word = _coordinate_random_c_word(d_type, seed, group, physical_thread, reg)
                model_case = SimpleNamespace(
                    a=a_for_row(row),
                    b=b_for_col(col),
                    c=c_word,
                    metadata=metadata,
                    metadata_hi=metadata_hi,
                )
                expected = random_model.eval(model_case)
                observed = out[thread][reg]
                random_payload_checked_words += 1
                if observed != expected:
                    random_payload_mismatch_count += 1
                    if len(random_payload_mismatches) < args.show:
                        random_payload_mismatches.append(
                            {
                                "case": case_idx,
                                "seed": tcgen05_bit_hex(seed),
                                "position": f"t{thread}:r{reg}",
                                "coordinate": [row, col],
                                "observed": tcgen05_bit_hex(observed),
                                "expected": tcgen05_bit_hex(expected),
                                "observed_class": _shape_result_class(observed, d_type),
                                "expected_class": _shape_result_class(expected, d_type),
                                "c": tcgen05_bit_hex(c_word),
                            }
                        )
    mapped_sample = [
        {
            "position": f"t{thread}:r{reg}",
            "coordinate": [row, col],
        }
        for (thread, reg), (row, col) in list(position_map.items())[: args.show]
    ]
    duplicate_sample = [
        {
            "coordinate": [coord[0], coord[1]],
            "positions": [f"t{thread}:r{reg}" for thread, reg in positions[: args.show]],
            "count": len(positions),
        }
        for coord, positions in list(duplicate_coordinates.items())[: args.show]
    ]
    if args.cta_group == 2:
        layout = "A" if args.m == 256 else "B"
    else:
        layout = {32: "G", 64: "E", 128: "D"}[args.m] if args.ws else ("F" if args.m == 64 else "D")
    if args.sparse and args.cta_group == 2 and args.m == 128:
        layout = "C"
    basis_checked_words = len(position_map) * (1 + row_bits + col_bits)
    full_coverage = unique_coordinates == expected_coordinates and not duplicate_coordinates and not out_of_range
    full_random_payload = (
        random_payload_cases > 0
        and full_coverage
        and random_payload_mismatch_count == 0
        and random_payload_checked_words == len(position_map) * random_payload_cases
    )
    return {
        "format": args.format,
        "a_format": a_format,
        "b_format": b_format,
        "d_type": d_type,
        "m": args.m,
        "n": args.n,
        "k": _shape_logical_k(args.format, sparse=args.sparse),
        "cta_group": args.cta_group,
        "ws": args.ws,
        "sparse": args.sparse,
        "layout": layout,
        "metadata": tcgen05_bit_hex(metadata) if args.sparse else None,
        "metadata_hi": tcgen05_bit_hex(metadata_hi) if args.sparse else None,
        "row_bits": row_bits,
        "col_bits": col_bits,
        "active_positions": len(position_map),
        "unique_coordinates": unique_coordinates,
        "expected_coordinates": expected_coordinates,
        "full_coordinate_coverage": full_coverage,
        "coordinate_arithmetic_model": (
            "coordinate-all-row-col-plus-random-payload"
            if random_payload_cases
            else "coordinate-all-row-col-exact"
        ),
        "coordinate_arithmetic_kind": (
            "bit-exact all/row-bit/column-bit probes plus deterministic row/column/K/C random payloads"
            if random_payload_cases
            else "bit-exact all/row-bit/column-bit probe values checked at every mapped output coordinate"
        ),
        "coordinate_arithmetic_checked_words": basis_checked_words + random_payload_checked_words,
        "basis_coordinate_arithmetic_checked_words": basis_checked_words,
        "coordinate_arithmetic_mismatches": arithmetic_mismatch_count,
        "random_payload_cases": random_payload_cases,
        "random_payload_checked_words": random_payload_checked_words,
        "random_payload_mismatches": random_payload_mismatch_count,
        "full_coordinate_random_payload": full_random_payload,
        "full_coordinate_arithmetic": (
            full_coverage
            and arithmetic_mismatch_count == 0
            and random_payload_mismatch_count == 0
            and (random_payload_cases == 0 or full_random_payload)
        ),
        "duplicate_coordinate_count": len(duplicate_coordinates),
        "out_of_range_count": len([coord for coord in position_map.values() if coord[0] >= args.m or coord[1] >= args.n]),
        "mapped_sample": mapped_sample,
        "duplicate_sample": duplicate_sample,
        "missing_sample": missing_sample,
        "out_of_range_sample": out_of_range,
        "arithmetic_mismatch_sample": arithmetic_mismatches,
        "random_payload_mismatch_sample": random_payload_mismatches,
    }


def _run_tcgen05_shape_random_validation(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    harness = Tcgen05ShapeHarness(_tcgen05_shape_runner(args), device=args.device)
    rng = random.Random(args.seed)
    a_format = args.a_format or ("u8" if args.format == "i8" else args.format)
    b_format = args.b_format or ("u8" if args.format == "i8" else args.format)
    d_type = args.d_type or ("s32" if args.format == "i8" else "f32")
    saturate = getattr(args, "saturate", False)
    if args.format != "i8" and saturate:
        raise ValueError("--saturate is only valid with --format i8")
    if not args.sparse and (
        args.metadata_word is not None or args.metadata_word_hi is not None or args.random_metadata
    ):
        raise ValueError("shape metadata options require --sparse")
    if args.format in ("bf16", "f16") and {a_format, b_format} == {"bf16", "f16"}:
        raise ValueError("B200 rejects off-diagonal BF16/F16 kind::f16 A/B descriptors with an illegal instruction")
    if args.format in ("bf16", "f16") and d_type == "f16" and (a_format != "f16" or b_format != "f16"):
        raise ValueError("B200 kind::f16 F16-D shape probes require F16 A/B inputs")
    model = _make_tcgen05_shape_model(args.format, a_format, b_format, d_type, saturate=saturate)
    model_name = "raw-window-rz"
    model_kind = "existing scalar tcgen05 software model over runtime-shape hardware path"
    if args.format == "i8":
        model_name = "s32-exact-sat" if saturate else "s32-exact"
        model_kind = "S32 i8 wrap/clamp software model over runtime-shape hardware path"
    elif d_type == "f16":
        model_name = "mixed-raw-window-f16-rne" if (a_format != b_format or a_format != args.format) else "raw-window-f16-rne"
        model_kind = "raw-window model with F16 C/D and RNE half output over runtime-shape hardware path"
    elif a_format != args.format or b_format != args.format:
        model_name = "mixed-raw-window-rz"
        model_kind = "mixed A/B raw-window model over runtime-shape hardware path"
    if args.format == "e2m1" and args.sparse and args.cta_group == 1:
        model = _Tcgen05ShapeSparseFp4E2M1Cg1Model()
        model_name = "fp4-sparse-pair-window-rz"
        model_kind = "runtime-shape FP4 sparse pair-window model over the mapped cta_group::1 output slice"
    elif args.sparse and args.cta_group == 2 and args.format in ("fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1"):
        model = _Tcgen05ShapeSparseF8F6F4Cg2Model(a_format, b_format, d_type=d_type)
        model_name = "f8f6f4-sparse-cg2-first-window"
        model_kind = (
            "runtime-shape f8/f6/f4 sparse-A cta_group::2 first-window model "
            "over the selected output slice"
        )
    position_metadata = SPARSE_FIXED_METADATA
    position_metadata_hi = SPARSE_FIXED_METADATA
    if args.sparse and args.metadata_word is not None:
        position_metadata, position_metadata_hi = _sparse_metadata_from_args(args, rng, args.format)
    valid_positions = _shape_valid_positions(
        harness,
        args.format,
        args.m,
        args.n,
        model,
        ws=args.ws,
        cta_group=args.cta_group,
        sparse=args.sparse,
        a_format_name=a_format,
        b_format_name=b_format,
        d_type=d_type,
        saturate=saturate,
        metadata=position_metadata,
        metadata_hi=position_metadata_hi,
    )
    if not valid_positions:
        raise RuntimeError(
            f"no valid output positions discovered for {args.format} A={a_format} B={b_format} D={d_type} "
            f"m{args.m}n{args.n}"
        )
    k = _shape_logical_k(args.format, sparse=args.sparse)
    metadata_words = set()
    mismatches = []
    mismatch_count = 0
    stateful_replay_count = 0
    position_failures = 0
    case_base = 0
    while case_base < args.cases:
        batch_count = min(args.batch_size, args.cases - case_base)
        cases = []
        for _ in range(batch_count):
            metadata = SPARSE_FIXED_METADATA
            metadata_hi = SPARSE_FIXED_METADATA
            active_sparse = None
            if args.sparse:
                metadata, metadata_hi = _sparse_metadata_from_args(args, rng, args.format)
                metadata_words.add((metadata, metadata_hi))
                active_sparse = set(_shape_active_sparse_indices(args.format, metadata, metadata_hi))
            values = tuple(_random_shape_word(a_format, rng, finite_only=args.finite_only) for _ in range(k))
            if active_sparse is not None:
                values = tuple(value if idx in active_sparse else 0 for idx, value in enumerate(values))
            cases.append(
                Tcgen05ShapeProbeCase(
                    args.format,
                    args.m,
                    args.n,
                    values,
                    tuple(_random_shape_word(b_format, rng, finite_only=args.finite_only) for _ in range(k)),
                    _random_f16_word(rng, finite_only=args.finite_only)
                    if d_type == "f16"
                    else (
                        rng.getrandbits(32)
                        if d_type == "s32"
                        else _random_f32_word(rng, finite_only=args.finite_only)
                    ),
                    args.ws,
                    args.cta_group,
                    args.sparse,
                    a_format,
                    b_format,
                    d_type,
                    saturate,
                    metadata,
                    metadata_hi,
                )
            )
        outputs = harness.run(cases)
        for idx, (case, out) in enumerate(zip(cases, outputs)):
            first_thread, first_reg = valid_positions[0]
            observed = out[first_thread][first_reg]
            positions_match = all(out[thread][reg] == observed for thread, reg in valid_positions[1:])
            if not positions_match:
                position_failures += 1
            predicted = model.eval(case)
            matched = model.matches(case, observed)
            if not matched:
                if args.sparse and args.cta_group == 1 and (args.ws or args.format == "e2m1"):
                    replay_out = harness.run([case])[0]
                    replay_observed = replay_out[first_thread][first_reg]
                    replay_positions_match = all(
                        replay_out[thread][reg] == replay_observed
                        for thread, reg in valid_positions[1:]
                    )
                    if replay_positions_match and model.matches(case, replay_observed):
                        stateful_replay_count += 1
                        continue
                mismatch_count += 1
                if len(mismatches) < args.show:
                    values = {out[thread][reg] for thread, reg in valid_positions}
                    mismatches.append(
                        {
                            "case": case_base + idx,
                            "observed": tcgen05_bit_hex(observed),
                            "observed_class": _shape_result_class(observed, d_type),
                            "model": tcgen05_bit_hex(predicted),
                            "model_class": _shape_result_class(predicted, d_type),
                            "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                            "a": [tcgen05_bit_hex(x) for x in case.a],
                            "b": [tcgen05_bit_hex(x) for x in case.b],
                            "c": tcgen05_bit_hex(case.c),
                            "metadata": tcgen05_bit_hex(case.metadata),
                            "metadata_hi": tcgen05_bit_hex(case.metadata_hi),
                        }
                    )
        case_base += batch_count
    if args.cta_group == 2:
        layout = "A" if args.m == 256 else "B"
    else:
        if args.ws:
            layout = {32: "G", 64: "E", 128: "D"}[args.m]
        else:
            layout = "F" if args.m == 64 else "D"
    if args.sparse and args.cta_group == 2 and args.m == 128:
        layout = "C"
    return {
        "cases": args.cases,
        "cta_group": args.cta_group,
        "dense": not args.sparse,
        "finite_only": args.finite_only,
        "format": args.format,
        "a_format": a_format,
        "b_format": b_format,
        "d_type": d_type,
        "saturate": saturate if args.format == "i8" else None,
        "m": args.m,
        "n": args.n,
        "k": k,
        "layout": layout,
        "batch_size": args.batch_size,
        "sparse": args.sparse,
        "metadata_mode": "random" if args.random_metadata else ("fixed" if args.sparse else None),
        "metadata_words_sample": [
            (
                tcgen05_bit_hex(lo)
                if lo == hi
                else f"{tcgen05_bit_hex(lo)}:{tcgen05_bit_hex(hi)}"
            )
            for lo, hi in sorted(metadata_words)[:8]
        ]
        if args.sparse
        else None,
        "mismatches": mismatch_count,
        "model": model_name,
        "model_kind": model_kind,
        "position_failures": position_failures,
        "seed": args.seed,
        "stateful_replays": stateful_replay_count,
        "ws": args.ws,
        "valid_position_count": len(valid_positions),
        "valid_positions_sample": [f"t{thread}:r{reg}" for thread, reg in valid_positions[:16]],
        "shown_mismatches": mismatches,
    }


def _random_nvfp4_word(rng: random.Random, *, finite_only: bool) -> int:
    return rng.randrange(16)


def _random_ue4m3_scale(rng: random.Random) -> int:
    return rng.randrange(0x80)


def _random_ue8m0_scale(rng: random.Random) -> int:
    return rng.randrange(0x100)


def _count_nan_scale_cases(cases, nan_encoding: int) -> int:
    return sum(
        any(scale == nan_encoding for scale in (*case.scale_a, *case.scale_b))
        for case in cases
    )


def _run_tcgen05_nvfp4_random_validation(args: argparse.Namespace) -> dict:
    harness = Tcgen05Nvfp4Harness(_tcgen05_nvfp4_runner(args), device=args.device)
    rng = random.Random(args.seed)
    random_scales = getattr(args, "random_scales", False)
    active_count = getattr(args, "active_count", None)
    zero_c = getattr(args, "zero_c", False)
    cases = []
    for _ in range(args.cases):
        active = None
        if active_count is not None:
            active = set(rng.sample(range(64), active_count))
        a = tuple(
            _random_nvfp4_word(rng, finite_only=args.finite_only)
            if active is None or k in active
            else 0
            for k in range(64)
        )
        b = tuple(_random_nvfp4_word(rng, finite_only=args.finite_only) for _ in range(64))
        cases.append(
            Tcgen05Nvfp4ProbeCase(
                a,
                b,
                0 if zero_c else _random_f32_word(rng, finite_only=args.finite_only),
                tuple(_random_ue4m3_scale(rng) for _ in range(4)) if random_scales else NVFP4_UNIT_SCALES,
                tuple(_random_ue4m3_scale(rng) for _ in range(4)) if random_scales else NVFP4_UNIT_SCALES,
            )
        )
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_nvfp4_model(args.model)
    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_nvfp4_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_nvfp4_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                        "scale_a": [tcgen05_bit_hex(x) for x in case.scale_a],
                        "scale_b": [tcgen05_bit_hex(x) for x in case.scale_b],
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff
    model_kind = "NVFP4 scaled software model" if random_scales else "NVFP4 unit-scale software model"
    if args.model == "hardware-oracle":
        model_kind = "hardware-backed oracle"
    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "model": args.model,
        "model_kind": model_kind,
        "active_count": active_count,
        "random_scales": random_scales,
        "nan_scale_cases": _count_nan_scale_cases(cases, 0x7F) if random_scales else 0,
        "seed": args.seed,
        "zero_c": zero_c,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _scale_tuple(
    rng: random.Random,
    *,
    count: int,
    random_scales: bool,
    unit_scales: tuple[int, ...],
    random_scale_fn=_random_ue8m0_scale,
) -> tuple[int, ...]:
    if not random_scales:
        return unit_scales
    values = [random_scale_fn(rng) for _ in range(count)]
    values.extend(unit_scales[len(values) :])
    return tuple(values)


def _run_tcgen05_mxf8f6f4_random_validation(args: argparse.Namespace) -> dict:
    op = MXF8F6F4_UE8M0_FORMAT_OPS[args.format]
    harness = Tcgen05BlockScaledHarness(_tcgen05_block_scaled_runner(args), op=op, device=args.device)
    rng = random.Random(args.seed)
    random_scales = getattr(args, "random_scales", False)
    active_count = getattr(args, "active_count", None)
    zero_c = getattr(args, "zero_c", False)
    a_format = getattr(args, "a_format", None) or args.format
    b_format = getattr(args, "b_format", None) or args.format
    cases = []
    for _ in range(args.cases):
        active = None
        if active_count is not None:
            active = set(rng.sample(range(32), active_count))
        a = tuple(
            _random_f8f6f4_word(a_format, rng, finite_only=args.finite_only)
            if active is None or k in active
            else 0
            for k in range(32)
        ) + (0,) * 32
        b = tuple(_random_f8f6f4_word(b_format, rng, finite_only=args.finite_only) for _ in range(32)) + (0,) * 32
        cases.append(
            Tcgen05BlockScaledProbeCase(
                a,
                b,
                0 if zero_c else _random_f32_word(rng, finite_only=args.finite_only),
                _scale_tuple(rng, count=1, random_scales=random_scales, unit_scales=MX_UE8M0_UNIT_SCALES),
                _scale_tuple(rng, count=1, random_scales=random_scales, unit_scales=MX_UE8M0_UNIT_SCALES),
                a_format,
                b_format,
                args.n,
                args.cta_group,
                args.m,
            )
        )
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_mxf8f6f4_model(args.format, args.model, a_format, b_format)

    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_nvfp4_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_nvfp4_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a[:32]],
                        "b": [tcgen05_bit_hex(x) for x in case.b[:32]],
                        "c": tcgen05_bit_hex(case.c),
                        "scale_a": [tcgen05_bit_hex(x) for x in case.scale_a],
                        "scale_b": [tcgen05_bit_hex(x) for x in case.scale_b],
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff

    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "format": args.format,
        "a_format": a_format,
        "b_format": b_format,
        "m": args.m,
        "n": args.n,
        "cta_group": args.cta_group,
        "model": args.model,
        "model_kind": "MXF8F6F4 UE8M0 block-scaled software model"
        if args.model != "hardware-oracle"
        else "hardware-backed oracle",
        "active_count": active_count,
        "random_scales": random_scales,
        "nan_scale_cases": _count_nan_scale_cases(cases, 0xFF) if random_scales else 0,
        "seed": args.seed,
        "zero_c": zero_c,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _run_tcgen05_block_scaled_random_validation(args: argparse.Namespace) -> dict:
    op = BLOCK_SCALED_OPS[args.kind]
    harness = Tcgen05BlockScaledHarness(_tcgen05_block_scaled_runner(args), op=op, device=args.device)
    rng = random.Random(args.seed)
    random_scales = getattr(args, "random_scales", False)
    active_count = getattr(args, "active_count", None)
    zero_c = getattr(args, "zero_c", False)
    ue4m3_scales = args.kind == "mxf4nvf4-4x-ue4m3"
    scale_count = 4 if args.kind == "mxf4nvf4-4x" else 2
    if ue4m3_scales:
        scale_count = 4
    unit_scales = NVFP4_UNIT_SCALES if ue4m3_scales else MX_UE8M0_UNIT_SCALES
    random_scale_fn = _random_ue4m3_scale if ue4m3_scales else _random_ue8m0_scale
    cases = []
    for _ in range(args.cases):
        active = None
        if active_count is not None:
            active = set(rng.sample(range(64), active_count))
        a = tuple(
            _random_nvfp4_word(rng, finite_only=args.finite_only)
            if active is None or k in active
            else 0
            for k in range(64)
        )
        b = tuple(_random_nvfp4_word(rng, finite_only=args.finite_only) for _ in range(64))
        cases.append(
            Tcgen05BlockScaledProbeCase(
                a,
                b,
                0 if zero_c else _random_f32_word(rng, finite_only=args.finite_only),
                _scale_tuple(
                    rng,
                    count=scale_count,
                    random_scales=random_scales,
                    unit_scales=unit_scales,
                    random_scale_fn=random_scale_fn,
                ),
                _scale_tuple(
                    rng,
                    count=scale_count,
                    random_scales=random_scales,
                    unit_scales=unit_scales,
                    random_scale_fn=random_scale_fn,
                ),
                n=args.n,
                cta_group=args.cta_group,
                m=args.m,
            )
        )
    outputs = harness.run(cases)
    if args.model == "hardware-oracle":
        model_outputs = harness.run(cases)
        model = None
    else:
        model_outputs = None
        model = _make_tcgen05_block_scaled_model(args.kind, args.model)

    mismatches = []
    mismatch_count = 0
    position_failures = 0
    oracle_position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_nvfp4_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        first_diff = None
        if model_outputs is None:
            assert model is not None
            predicted = model.eval(case)
            matched = model.matches(case, observed)
        else:
            model_valid_words = tcgen05_nvfp4_flatten_output(model_outputs[idx])
            predicted = model_valid_words[0]
            if len(set(model_valid_words)) != 1:
                oracle_position_failures += 1
            matched = model_valid_words == valid_words
            if not matched:
                first_diff = next(
                    (
                        {
                            "valid_word_index": word_idx,
                            "observed": tcgen05_bit_hex(observed_word),
                            "model": tcgen05_bit_hex(model_word),
                        }
                        for word_idx, (observed_word, model_word) in enumerate(zip(valid_words, model_valid_words))
                        if observed_word != model_word
                    ),
                    None,
                )
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                        "scale_a": [tcgen05_bit_hex(x) for x in case.scale_a],
                        "scale_b": [tcgen05_bit_hex(x) for x in case.scale_b],
                    }
                )
                if first_diff is not None:
                    mismatches[-1]["first_valid_word_diff"] = first_diff

    report = {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "kind": args.kind,
        "m": args.m,
        "n": args.n,
        "cta_group": args.cta_group,
        "model": args.model,
        "model_kind": "NVFP4 UE4M3 block-scaled software model"
        if ue4m3_scales and args.model != "hardware-oracle"
        else "MXF4-family UE8M0 block-scaled software model"
        if args.model != "hardware-oracle"
        else "hardware-backed oracle",
        "active_count": active_count,
        "random_scales": random_scales,
        "nan_scale_cases": _count_nan_scale_cases(cases, 0x7F if ue4m3_scales else 0xFF)
        if random_scales
        else 0,
        "seed": args.seed,
        "zero_c": zero_c,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }
    if args.model == "hardware-oracle":
        report["oracle_comparison"] = "independent second hardware execution over the full valid output slice"
        report["oracle_position_failures"] = oracle_position_failures
    return report


def _block_scaled_sparse_active_indices(kind: str, metadata: int, metadata_hi: int) -> set[int]:
    active = set()
    if kind == "mxf8f6f4":
        for chunk in range(16):
            word = metadata if chunk < 8 else metadata_hi
            selector = (word >> ((chunk & 7) * 4)) & 0xF
            active.add(chunk * 4 + (selector & 0x3))
            active.add(chunk * 4 + ((selector >> 2) & 0x3))
    else:
        for chunk in range(16):
            word = metadata if chunk < 8 else metadata_hi
            selector = (word >> ((chunk & 7) * 4)) & 0xF
            first = selector & 0x3
            second = (selector >> 2) & 0x3
            active.update(
                {
                    chunk * 8 + first * 2,
                    chunk * 8 + first * 2 + 1,
                    chunk * 8 + second * 2,
                    chunk * 8 + second * 2 + 1,
                }
            )
    return active


def _run_tcgen05_block_scaled_sparse_random_validation(args: argparse.Namespace) -> dict:
    if args.kind == "mxf8f6f4":
        op = MXF8F6F4_UE8M0_FORMAT_OPS[args.format]
        a_format = args.a_format or args.format
        b_format = args.b_format or args.format
        logical_k = 64
        scale_count = 2
        metadata_format = "fp8"
    else:
        op = BLOCK_SCALED_SPARSE_OPS[args.kind]
        a_format = "e2m1"
        b_format = "e2m1"
        logical_k = 128
        scale_count = 4
        metadata_format = "nvfp4"
    harness = Tcgen05BlockScaledSparseHarness(
        _tcgen05_block_scaled_sparse_runner(args), op=op, device=args.device
    )
    rng = random.Random(args.seed)
    random_scales = getattr(args, "random_scales", False)
    zero_c = getattr(args, "zero_c", False)
    ue4m3_scales = args.kind == "mxf4nvf4-4x-ue4m3"
    unit_scales = NVFP4_SPARSE_UNIT_SCALES if ue4m3_scales else MX_UE8M0_UNIT_SCALES
    random_scale_fn = _random_ue4m3_scale if ue4m3_scales else _random_ue8m0_scale
    cases = []
    metadata_words = set()
    for _ in range(args.cases):
        metadata, metadata_hi = _sparse_metadata_from_args(args, rng, metadata_format)
        metadata_words.add((metadata, metadata_hi))
        active = _block_scaled_sparse_active_indices(args.kind, metadata, metadata_hi)
        if args.kind == "mxf8f6f4":
            a = tuple(
                _random_f8f6f4_word(a_format, rng, finite_only=args.finite_only)
                if k < logical_k and k in active
                else 0
                for k in range(128)
            )
            b = tuple(
                _random_f8f6f4_word(b_format, rng, finite_only=args.finite_only) if k < logical_k else 0
                for k in range(128)
            )
        else:
            a = tuple(
                _random_nvfp4_word(rng, finite_only=args.finite_only)
                if k < logical_k and k in active
                else 0
                for k in range(128)
            )
            b = tuple(
                _random_nvfp4_word(rng, finite_only=args.finite_only) if k < logical_k else 0
                for k in range(128)
            )
        cases.append(
            Tcgen05BlockScaledSparseProbeCase(
                a,
                b,
                0 if zero_c else _random_f32_word(rng, finite_only=args.finite_only),
                metadata,
                metadata_hi,
                _scale_tuple(
                    rng,
                    count=scale_count,
                    random_scales=random_scales,
                    unit_scales=unit_scales,
                    random_scale_fn=random_scale_fn,
                ),
                _scale_tuple(
                    rng,
                    count=scale_count,
                    random_scales=random_scales,
                    unit_scales=unit_scales,
                    random_scale_fn=random_scale_fn,
                ),
                a_format,
                b_format,
                args.n,
                args.cta_group,
                args.m,
            )
        )
    outputs = harness.run(cases)

    mismatches = []
    mismatch_count = 0
    position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = tcgen05_nvfp4_flatten_output(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        public_kind = "mxf8f6f4" if args.kind == "mxf8f6f4" else "mxf4" if args.kind == "mxf4" else "mxf4nvf4"
        public_scale_vec = 1 if args.kind == "mxf8f6f4" else 2 if args.kind in ("mxf4", "mxf4nvf4-2x") else 4
        predicted = public_mma_dot(
            case.a[:logical_k],
            case.b[:logical_k],
            case.c,
            a_format=a_format,
            b_format=b_format,
            scaling="ue4m3" if ue4m3_scales else "ue8m0",
            scale_vec=public_scale_vec,
            kind=public_kind,
            scale_a=case.scale_a,
            scale_b=case.scale_b,
            sparse_metadata=(case.metadata, case.metadata_hi),
        )
        matched = predicted == observed
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                mismatches.append(
                    {
                        "case": idx,
                        "observed": tcgen05_bit_hex(observed),
                        "observed_class": f32_class(observed),
                        "model": tcgen05_bit_hex(predicted),
                        "model_class": f32_class(predicted),
                        "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                        "metadata": tcgen05_bit_hex(case.metadata),
                        "metadata_hi": tcgen05_bit_hex(case.metadata_hi),
                        "a": [tcgen05_bit_hex(x) for x in case.a],
                        "b": [tcgen05_bit_hex(x) for x in case.b],
                        "c": tcgen05_bit_hex(case.c),
                        "scale_a": [tcgen05_bit_hex(x) for x in case.scale_a],
                        "scale_b": [tcgen05_bit_hex(x) for x in case.scale_b],
                    }
                )

    return {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "kind": args.kind,
        "m": args.m,
        "n": args.n,
        "cta_group": args.cta_group,
        "format": args.format if args.kind == "mxf8f6f4" else "e2m1",
        "a_format": a_format,
        "b_format": b_format,
        "metadata_mode": "random" if args.random_metadata else "fixed",
        "metadata_words_sample": [
            tcgen05_bit_hex(lo) if lo == hi else f"{tcgen05_bit_hex(lo)}:{tcgen05_bit_hex(hi)}"
            for lo, hi in sorted(metadata_words)[:8]
        ],
        "model": args.model,
        "model_kind": "released public full-logical-K sparse block-scaled API",
        "random_scales": random_scales,
        "nan_scale_cases": _count_nan_scale_cases(cases, 0x7F if ue4m3_scales else 0xFF)
        if random_scales
        else 0,
        "seed": args.seed,
        "zero_c": zero_c,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }


def _tcgen05_sparse_runner(args: argparse.Namespace) -> str:
    default_runners = {
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_f16_probe",
        "build/tcgen05_fp8_probe",
        "build/tcgen05_f8f6f4_probe",
        "build/tcgen05_nvfp4_probe",
    }
    sparse_runners = {
        "tf32": "build/tcgen05_tf32_sparse_probe",
        "bf16": "build/tcgen05_bf16_sparse_probe",
        "f16": "build/tcgen05_f16_sparse_probe",
        "fp8": "build/tcgen05_fp8_sparse_probe",
        "e5m2": "build/tcgen05_f8f6f4_sparse_probe",
        "e2m3": "build/tcgen05_f8f6f4_sparse_probe",
        "e3m2": "build/tcgen05_f8f6f4_sparse_probe",
        "e2m1": "build/tcgen05_f8f6f4_sparse_probe",
        "nvfp4": "build/tcgen05_nvfp4_sparse_probe",
    }
    if args.runner in default_runners:
        return sparse_runners[args.format]
    return args.runner


def _sparse_metadata_from_args(args: argparse.Namespace, rng: random.Random, format_name: str) -> tuple[int, int]:
    if args.metadata_word is not None and args.random_metadata:
        raise ValueError("--metadata-word and --random-metadata are mutually exclusive")
    if getattr(args, "metadata_word_hi", None) is not None and args.metadata_word is None:
        raise ValueError("--metadata-word-hi requires --metadata-word")
    if args.random_metadata:
        selectors = SPARSE_TF32_SELECTORS if format_name == "tf32" else SPARSE_2OF4_SELECTORS
        def random_word() -> int:
            word = 0
            for chunk in range(8):
                word |= rng.choice(selectors) << (chunk * 4)
            return word

        lo = random_word()
        hi = (
            random_word()
            if format_name in ("fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1", "i8", "nvfp4")
            else lo
        )
        return lo, hi
    if args.metadata_word is not None:
        lo = parse_hex_word(args.metadata_word)
        hi = parse_hex_word(args.metadata_word_hi) if getattr(args, "metadata_word_hi", None) is not None else lo
        return lo, hi
    return SPARSE_FIXED_METADATA, SPARSE_FIXED_METADATA


def _sparse_active_indices(format_name: str, metadata: int, metadata_hi: int | None = None) -> set[int]:
    if metadata_hi is None:
        metadata_hi = metadata
    if format_name == "tf32":
        active = set()
        for chunk in range(8):
            selector = (metadata >> (chunk * 4)) & 0xF
            active.add(chunk * 2 + (1 if selector == 0xE else 0))
        return active
    if format_name in ("bf16", "f16"):
        chunks = 8
        chunk_size = 4
    elif format_name in ("fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1", "i8"):
        chunks = 16
        chunk_size = 4
    elif format_name == "nvfp4":
        chunks = 16
        chunk_size = 8
    else:
        raise ValueError(f"unsupported sparse format: {format_name}")
    active = set()
    for chunk in range(chunks):
        word = metadata if chunk < 8 else metadata_hi
        selector = (word >> ((chunk & 7) * 4)) & 0xF
        first = selector & 0x3
        second = (selector >> 2) & 0x3
        if format_name == "nvfp4":
            active.update(
                {
                    chunk * chunk_size + first * 2,
                    chunk * chunk_size + first * 2 + 1,
                    chunk * chunk_size + second * 2,
                    chunk * chunk_size + second * 2 + 1,
                }
            )
        else:
            active.update({chunk * chunk_size + first, chunk * chunk_size + second})
    return active


def _sparse_case_and_model(format_name: str):
    if format_name == "tf32":
        return Tcgen05Tf32SparseHarness, Tcgen05Tf32SparseProbeCase, Tcgen05RawWindowTf32MmaModel(), tcgen05_flatten_output
    if format_name == "bf16":
        return Tcgen05Bf16SparseHarness, Tcgen05Bf16SparseProbeCase, Tcgen05RawWindowBf16MmaModel(), tcgen05_bf16_flatten_output
    if format_name == "f16":
        return Tcgen05F16SparseHarness, Tcgen05F16SparseProbeCase, Tcgen05RawWindowF16MmaModel(), tcgen05_f16_flatten_output
    if format_name == "fp8":
        return (
            Tcgen05Fp8SparseHarness,
            Tcgen05Fp8SparseProbeCase,
            Tcgen05RawWindowFp8E4M3MmaModel(),
            tcgen05_fp8_flatten_output,
        )
    if format_name in ("e5m2", "e2m3", "e3m2", "e2m1"):
        model = _make_tcgen05_f8f6f4_model(format_name, "raw-window-rz")
        return (
            lambda runner, *, device=0: Tcgen05F8f6f4SparseHarness(
                runner,
                format_name=format_name,
                device=device,
            ),
            Tcgen05F8f6f4SparseProbeCase,
            model,
            tcgen05_f8f6f4_flatten_output,
        )
    if format_name == "nvfp4":
        return (
            Tcgen05Nvfp4SparseHarness,
            Tcgen05Nvfp4SparseProbeCase,
            Tcgen05RawWindowNvfp4MmaModel(c_merge_group_size=32, scale_block_size=32),
            tcgen05_nvfp4_flatten_output,
        )
    raise ValueError(f"unsupported sparse format: {format_name}")


def _make_sparse_random_case(
    format_name: str,
    rng: random.Random,
    *,
    finite_only: bool,
    random_scales: bool,
    metadata: int,
    metadata_hi: int | None = None,
):
    if metadata_hi is None:
        metadata_hi = metadata
    active = _sparse_active_indices(format_name, metadata, metadata_hi)
    if format_name == "tf32":
        a = tuple(_random_f32_word(rng, finite_only=finite_only) if k in active else 0 for k in range(16))
        b = tuple(_random_f32_word(rng, finite_only=finite_only) for _ in range(16))
        return Tcgen05Tf32SparseProbeCase(a, b, _random_f32_word(rng, finite_only=finite_only), metadata)
    if format_name == "bf16":
        a = tuple(_random_f32_word(rng, finite_only=finite_only) if k in active else 0 for k in range(32))
        b = tuple(_random_f32_word(rng, finite_only=finite_only) for _ in range(32))
        return Tcgen05Bf16SparseProbeCase(a, b, _random_f32_word(rng, finite_only=finite_only), metadata)
    if format_name == "f16":
        a = tuple(_random_f16_word(rng, finite_only=finite_only) if k in active else 0 for k in range(32))
        b = tuple(_random_f16_word(rng, finite_only=finite_only) for _ in range(32))
        return Tcgen05F16SparseProbeCase(a, b, _random_f32_word(rng, finite_only=finite_only), metadata)
    if format_name == "fp8":
        a = tuple(_random_fp8_word(rng, finite_only=finite_only) if k in active else 0 for k in range(64))
        b = tuple(_random_fp8_word(rng, finite_only=finite_only) for _ in range(64))
        return Tcgen05Fp8SparseProbeCase(a, b, _random_f32_word(rng, finite_only=finite_only), metadata, metadata_hi)
    if format_name in ("e5m2", "e2m3", "e3m2", "e2m1"):
        a = tuple(
            _random_f8f6f4_word(format_name, rng, finite_only=finite_only) if k in active else 0
            for k in range(64)
        )
        b = tuple(_random_f8f6f4_word(format_name, rng, finite_only=finite_only) for _ in range(64))
        return Tcgen05F8f6f4SparseProbeCase(a, b, _random_f32_word(rng, finite_only=finite_only), metadata, metadata_hi)
    if format_name == "nvfp4":
        a = tuple(_random_nvfp4_word(rng, finite_only=finite_only) if k in active else 0 for k in range(128))
        b = tuple(_random_nvfp4_word(rng, finite_only=finite_only) for _ in range(128))
        scale_a = tuple(_random_ue4m3_scale(rng) for _ in range(4)) if random_scales else NVFP4_SPARSE_UNIT_SCALES
        scale_b = tuple(_random_ue4m3_scale(rng) for _ in range(4)) if random_scales else NVFP4_SPARSE_UNIT_SCALES
        return Tcgen05Nvfp4SparseProbeCase(
            a,
            b,
            _random_f32_word(rng, finite_only=finite_only),
            metadata,
            metadata_hi,
            scale_a,
            scale_b,
        )
    raise ValueError(f"unsupported sparse format: {format_name}")


def _run_tcgen05_sparse_random_validation(args: argparse.Namespace) -> dict:
    harness_cls, _case_cls, model, flattener = _sparse_case_and_model(args.format)
    harness = harness_cls(_tcgen05_sparse_runner(args), device=args.device)
    rng = random.Random(args.seed)
    cases = []
    metadata_words = set()
    for _ in range(args.cases):
        metadata, metadata_hi = _sparse_metadata_from_args(args, rng, args.format)
        metadata_words.add((metadata, metadata_hi))
        cases.append(
            _make_sparse_random_case(
                args.format,
                rng,
                finite_only=args.finite_only,
                random_scales=getattr(args, "random_scales", False),
                metadata=metadata,
                metadata_hi=metadata_hi,
            )
        )
    outputs = harness.run(cases)

    mismatches = []
    mismatch_count = 0
    position_failures = 0
    for idx, (case, out) in enumerate(zip(cases, outputs)):
        valid_words = flattener(out)
        values = set(valid_words)
        if len(values) != 1:
            position_failures += 1
        observed = valid_words[0]
        # Sparse MMA never consumes B values at metadata-disabled logical K
        # positions. Mask them in the scalar-model view as well. Leaving a
        # disabled NaN/Inf in B made the old harness evaluate 0 * NaN/Inf and
        # report false mismatches even though that pair is absent in hardware.
        active = _sparse_active_indices(
            args.format,
            case.metadata,
            getattr(case, "metadata_hi", case.metadata),
        )
        model_case = SimpleNamespace(
            a=case.a,
            b=tuple(value if k in active else 0 for k, value in enumerate(case.b)),
            c=case.c,
            **(
                {"scale_a": case.scale_a, "scale_b": case.scale_b}
                if hasattr(case, "scale_a")
                else {}
            ),
        )
        predicted = model.eval(model_case)
        matched = model.matches(model_case, observed)
        if not matched:
            mismatch_count += 1
            if len(mismatches) < args.show:
                entry = {
                    "case": idx,
                    "observed": tcgen05_bit_hex(observed),
                    "observed_class": f32_class(observed),
                    "model": tcgen05_bit_hex(predicted),
                    "model_class": f32_class(predicted),
                    "valid_unique_words": [tcgen05_bit_hex(x) for x in sorted(values)[:8]],
                    "a": [tcgen05_bit_hex(x) for x in case.a],
                    "b": [tcgen05_bit_hex(x) for x in case.b],
                    "c": tcgen05_bit_hex(case.c),
                    "metadata": tcgen05_bit_hex(case.metadata),
                }
                if hasattr(case, "metadata_hi"):
                    entry["metadata_hi"] = tcgen05_bit_hex(case.metadata_hi)
                if hasattr(case, "scale_a"):
                    entry["scale_a"] = [tcgen05_bit_hex(x) for x in case.scale_a]
                    entry["scale_b"] = [tcgen05_bit_hex(x) for x in case.scale_b]
                mismatches.append(entry)

    model_kind = "expanded logical-K software model over the true tcgen05.mma.sp hardware path"

    return {
        "cases": args.cases,
        "finite_only": args.finite_only,
        "format": args.format,
        "metadata_mode": "random" if args.random_metadata else "fixed",
        "metadata_words_sample": [
            (
                tcgen05_bit_hex(lo)
                if lo == hi
                else f"{tcgen05_bit_hex(lo)}:{tcgen05_bit_hex(hi)}"
            )
            for lo, hi in sorted(metadata_words)[:8]
        ],
        "model_kind": model_kind,
        "random_scales": getattr(args, "random_scales", False) if args.format == "nvfp4" else False,
        "nan_scale_cases": _count_nan_scale_cases(cases, 0x7F)
        if args.format == "nvfp4" and getattr(args, "random_scales", False)
        else 0,
        "seed": args.seed,
        "mismatches": mismatch_count,
        "position_failures": position_failures,
        "shown_mismatches": mismatches,
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    runner = Path(args.runner)
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [line.strip() for line in result.stdout.splitlines() if "MMA" in line or "HMMA" in line]
    print("\n".join(lines))
    return 0


def _tcgen05_runner(args: argparse.Namespace) -> str:
    if args.runner == "build/mma_tf32_probe":
        return "build/tcgen05_tf32_probe"
    return args.runner


def _tcgen05_bf16_runner(args: argparse.Namespace) -> str:
    if args.runner in ("build/mma_tf32_probe", "build/tcgen05_tf32_probe"):
        return "build/tcgen05_bf16_probe"
    return args.runner


def _tcgen05_f16_runner(args: argparse.Namespace) -> str:
    if args.runner in ("build/mma_tf32_probe", "build/tcgen05_tf32_probe", "build/tcgen05_bf16_probe"):
        return "build/tcgen05_f16_probe"
    return args.runner


def _tcgen05_fp8_runner(args: argparse.Namespace) -> str:
    if args.runner in ("build/mma_tf32_probe", "build/tcgen05_tf32_probe", "build/tcgen05_bf16_probe"):
        return "build/tcgen05_fp8_probe"
    return args.runner


def _tcgen05_f8f6f4_runner(args: argparse.Namespace) -> str:
    if args.runner in (
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_f16_probe",
        "build/tcgen05_fp8_probe",
    ):
        return "build/tcgen05_f8f6f4_probe"
    return args.runner


def _tcgen05_i8_runner(args: argparse.Namespace) -> str:
    if args.runner in (
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_f16_probe",
        "build/tcgen05_fp8_probe",
        "build/tcgen05_f8f6f4_probe",
    ):
        return "build/tcgen05_i8_probe"
    return args.runner


def _tcgen05_nvfp4_runner(args: argparse.Namespace) -> str:
    if args.runner in (
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_fp8_probe",
    ):
        return "build/tcgen05_nvfp4_probe"
    return args.runner


def _tcgen05_block_scaled_runner(args: argparse.Namespace) -> str:
    if args.runner in (
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_f16_probe",
        "build/tcgen05_fp8_probe",
        "build/tcgen05_f8f6f4_probe",
        "build/tcgen05_i8_probe",
        "build/tcgen05_nvfp4_probe",
    ):
        return "build/tcgen05_block_scaled_probe"
    return args.runner


def _tcgen05_block_scaled_sparse_runner(args: argparse.Namespace) -> str:
    if args.runner in (
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_f16_probe",
        "build/tcgen05_fp8_probe",
        "build/tcgen05_f8f6f4_probe",
        "build/tcgen05_i8_probe",
        "build/tcgen05_nvfp4_probe",
        "build/tcgen05_block_scaled_probe",
    ):
        return "build/tcgen05_block_scaled_sparse_probe"
    return args.runner


def _tcgen05_shape_runner(args: argparse.Namespace) -> str:
    if args.runner in (
        "build/mma_tf32_probe",
        "build/tcgen05_tf32_probe",
        "build/tcgen05_bf16_probe",
        "build/tcgen05_f16_probe",
        "build/tcgen05_fp8_probe",
        "build/tcgen05_f8f6f4_probe",
        "build/tcgen05_i8_probe",
        "build/tcgen05_nvfp4_probe",
        "build/tcgen05_block_scaled_probe",
    ):
        return "build/tcgen05_shape_probe"
    return args.runner


def cmd_tcgen05_quick(args: argparse.Namespace) -> int:
    harness = Tcgen05Harness(_tcgen05_runner(args), device=args.device)
    report = run_tcgen05_quick_probes(harness)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Runner:", report["runner_info"])
        print("Deterministic:", report["determinism"]["deterministic"])
        print("Position independent:", report["position_independence"]["position_independent"])
        print("Valid output slice:", report["output_layout"]["valid_threads"], "threads")
        print("A ignored mantissa bits:", report["pretruncation"]["a_ignored_mantissa_bits"])
        print("B ignored mantissa bits:", report["pretruncation"]["b_ignored_mantissa_bits"])
        print("Precision:", report["precision"])
        print("Rounding:", json.dumps(report["rounding"]["observed"], indent=2, sort_keys=True))
        print("Special values:", json.dumps(report["special_values"], indent=2, sort_keys=True))
    return 0


def cmd_tcgen05_deep(args: argparse.Namespace) -> int:
    harness = Tcgen05Harness(_tcgen05_runner(args), device=args.device)
    report = run_tcgen05_deep_probes(harness, max_precision_exp=args.max_exp)
    if args.random_cases:
        random_args = argparse.Namespace(
            runner=args.runner,
            device=args.device,
            cases=args.random_cases,
            seed=args.seed,
            show=args.show,
            model=args.model,
            finite_only=args.finite_only,
        )
        report["random_validation"] = _run_tcgen05_random_validation(random_args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Runner:", report["runner_info"])
        print("Valid output slice:", report["output_layout"]["valid_threads"], "threads")
        print("Position independent:", report["position_independence"]["position_independent"])
        print("Swamping pairs:", len(report["swamping_order"]["pairs"]))
        print("Precision pairs:", len(report["precision_matrix"]["thresholds"]))
        print("Cancellation:", json.dumps(report["cancellation_precision"]["thresholds"], indent=2, sort_keys=True))
        print("Rounding:", json.dumps(report["rounding"]["observed"], indent=2, sort_keys=True))
        print("Special values:", json.dumps(report["special_values"], indent=2, sort_keys=True))
        if "random_validation" in report:
            print("Random validation:", json.dumps(report["random_validation"], indent=2, sort_keys=True))
    return 0


def _emit_validation_report(report: dict) -> int:
    for field in ("cases", "mismatches", "position_failures"):
        if field not in report:
            raise ValueError(f"validation report is missing required field {field!r}")
        if type(report[field]) is not int or report[field] < 0:
            raise ValueError(f"validation report field {field!r} must be a nonnegative integer")
    if "oracle_position_failures" in report and (
        type(report["oracle_position_failures"]) is not int or report["oracle_position_failures"] < 0
    ):
        raise ValueError("validation report field 'oracle_position_failures' must be a nonnegative integer")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if any(report.get(field, 0) != 0 for field in (
        "mismatches", "position_failures", "oracle_position_failures"
    )) else 0


def cmd_tcgen05_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_random_validation(args))


def cmd_tcgen05_bf16_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_bf16_random_validation(args))


def cmd_tcgen05_f16_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_f16_random_validation(args))


def cmd_tcgen05_fp8_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_fp8_random_validation(args))


def cmd_tcgen05_f8f6f4_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_f8f6f4_random_validation(args))


def cmd_tcgen05_i8_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_i8_random_validation(args))


def cmd_tcgen05_nvfp4_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_nvfp4_random_validation(args))


def cmd_tcgen05_mxf8f6f4_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_mxf8f6f4_random_validation(args))


def cmd_tcgen05_block_scaled_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_block_scaled_random_validation(args))


def cmd_tcgen05_block_scaled_sparse_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_block_scaled_sparse_random_validation(args))


def cmd_tcgen05_shape_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_shape_random_validation(args))


def cmd_tcgen05_shape_coordinate_map(args: argparse.Namespace) -> int:
    report = _run_tcgen05_shape_coordinate_map(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_full and not report["full_coordinate_arithmetic"]:
        return 1
    return 0


def cmd_tcgen05_sparse_random(args: argparse.Namespace) -> int:
    return _emit_validation_report(_run_tcgen05_sparse_random_validation(args))


def cmd_tcgen05_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05Harness(_tcgen05_runner(args), device=args.device)
    case = Tcgen05ProbeCase(parse_hex_words(args.a, 8), parse_hex_words(args.b, 8), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(tcgen05_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_bf16_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05Bf16Harness(_tcgen05_bf16_runner(args), device=args.device)
    case = Tcgen05Bf16ProbeCase(parse_hex_words(args.a, 16), parse_hex_words(args.b, 16), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(tcgen05_bf16_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_f16_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05F16Harness(_tcgen05_f16_runner(args), device=args.device)
    case = Tcgen05F16ProbeCase(parse_hex_words(args.a, 16), parse_hex_words(args.b, 16), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(tcgen05_f16_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_fp8_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05Fp8Harness(_tcgen05_fp8_runner(args), device=args.device)
    case = Tcgen05Fp8ProbeCase(parse_hex_words(args.a, 32), parse_hex_words(args.b, 32), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(tcgen05_fp8_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_f8f6f4_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05F8f6f4Harness(
        _tcgen05_f8f6f4_runner(args),
        format_name=args.format,
        device=args.device,
    )
    case = Tcgen05F8f6f4ProbeCase(parse_hex_words(args.a, 32), parse_hex_words(args.b, 32), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(tcgen05_f8f6f4_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_i8_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05I8Harness(
        _tcgen05_i8_runner(args),
        a_type=args.a_type,
        b_type=args.b_type,
        saturate=args.saturate,
        device=args.device,
    )
    case = Tcgen05I8ProbeCase(parse_hex_words(args.a, 32), parse_hex_words(args.b, 32), parse_hex_word(args.c))
    out = harness.run([case])[0]
    values = sorted(tcgen05_i8_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} s32={_i8_word_to_s32(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_nvfp4_run_case(args: argparse.Namespace) -> int:
    harness = Tcgen05Nvfp4Harness(_tcgen05_nvfp4_runner(args), device=args.device)
    scale_a = parse_ue4m3_words(args.scale_a, 4) if args.scale_a else NVFP4_UNIT_SCALES
    scale_b = parse_ue4m3_words(args.scale_b, 4) if args.scale_b else NVFP4_UNIT_SCALES
    case = Tcgen05Nvfp4ProbeCase(
        parse_hex_words(args.a, 64),
        parse_hex_words(args.b, 64),
        parse_hex_word(args.c),
        scale_a,
        scale_b,
    )
    out = harness.run([case])[0]
    values = sorted(tcgen05_nvfp4_unique_words(out))
    print(f"unique output words: {len(values)}")
    for value in values:
        print(f"{tcgen05_bit_hex(value)} {f32_class(value)}")
    if args.dump_all:
        for lane, regs in enumerate(out):
            print(f"thread {lane:03d}: " + " ".join(tcgen05_bit_hex(x) for x in regs))
    return 0


def cmd_tcgen05_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def cmd_tcgen05_bf16_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_bf16_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def cmd_tcgen05_f16_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_f16_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def cmd_tcgen05_fp8_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_fp8_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def cmd_tcgen05_f8f6f4_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_f8f6f4_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def cmd_tcgen05_i8_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_i8_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def cmd_tcgen05_nvfp4_inspect(args: argparse.Namespace) -> int:
    runner = Path(_tcgen05_nvfp4_runner(args))
    if not runner.exists():
        subprocess.run(["make", str(runner)], check=True)
    result = subprocess.run(["cuobjdump", "--dump-sass", str(runner)], check=True, text=True, stdout=subprocess.PIPE)
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in ("TCGEN05", "UMMA", "MMA", "TMEM", "UTMALDG", "UTMASTG"))
    ]
    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe NVIDIA TF32 m16n8k4 MMA behavior")
    parser.add_argument("--runner", default="build/mma_tf32_probe")
    parser.add_argument("--device", type=int, default=0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    quick = sub.add_parser("quick", help="run the prompt.md probe suite")
    quick.add_argument("--json", action="store_true")
    quick.set_defaults(func=cmd_quick)

    run_case = sub.add_parser("run-case", help="run one raw-bit case")
    run_case.add_argument("--a", required=True, help="four comma-separated uint32 hex words")
    run_case.add_argument("--b", required=True, help="four comma-separated uint32 hex words")
    run_case.add_argument("--c", required=True, help="one uint32 hex word")
    run_case.add_argument("--dump-all", action="store_true")
    run_case.set_defaults(func=cmd_run_case)

    random_cmd = sub.add_parser("random", help="random raw-bit validation against a selected model")
    random_cmd.add_argument("--cases", type=int, default=1000)
    random_cmd.add_argument("--seed", type=int, default=1)
    random_cmd.add_argument("--show", type=int, default=8)
    random_cmd.add_argument("--model", choices=["simple", "fixed"], default="simple")
    random_cmd.set_defaults(func=cmd_random)

    inspect = sub.add_parser("inspect", help="print SASS lines containing MMA")
    inspect.set_defaults(func=cmd_inspect)

    tcgen05_quick = sub.add_parser("tcgen05-quick", help="run quick probes for tcgen05.mma TF32 m64n8k8")
    tcgen05_quick.add_argument("--json", action="store_true")
    tcgen05_quick.set_defaults(func=cmd_tcgen05_quick)

    tcgen05_deep = sub.add_parser("tcgen05-deep", help="run detailed probes for tcgen05.mma TF32 m64n8k8")
    tcgen05_deep.add_argument("--json", action="store_true")
    tcgen05_deep.add_argument("--max-exp", type=int, default=80)
    tcgen05_deep.add_argument("--random-cases", type=int, default=0)
    tcgen05_deep.add_argument("--seed", type=int, default=1)
    tcgen05_deep.add_argument("--show", type=int, default=8)
    tcgen05_deep.add_argument("--finite-only", action="store_true")
    tcgen05_deep.add_argument(
        "--model",
        choices=[
            "exact-rne",
            "exact-rz",
            "fixed-rne",
            "fixed-rz",
            "hardware-oracle",
            "raw-window-rz",
            "twostage-c23-rne",
            "twostage-c24-rne",
            "twostage-c24-rz",
            "twostage-p25-c25-rz",
            "twostage-p26-c25-rz",
            "twostage-p26-c26-rz",
        ],
        default="raw-window-rz",
    )
    tcgen05_deep.set_defaults(func=cmd_tcgen05_deep)

    tcgen05_random = sub.add_parser("tcgen05-random", help="random raw-bit validation for tcgen05 models")
    tcgen05_random.add_argument("--cases", type=int, default=1000)
    tcgen05_random.add_argument("--seed", type=int, default=1)
    tcgen05_random.add_argument("--show", type=int, default=8)
    tcgen05_random.add_argument("--finite-only", action="store_true")
    tcgen05_random.add_argument(
        "--model",
        choices=[
            "exact-rne",
            "exact-rz",
            "fixed-rne",
            "fixed-rz",
            "hardware-oracle",
            "raw-window-rz",
            "twostage-c23-rne",
            "twostage-c24-rne",
            "twostage-c24-rz",
            "twostage-p25-c25-rz",
            "twostage-p26-c25-rz",
            "twostage-p26-c26-rz",
        ],
        default="raw-window-rz",
    )
    tcgen05_random.set_defaults(func=cmd_tcgen05_random)

    tcgen05_case = sub.add_parser("tcgen05-run-case", help="run one raw-bit tcgen05.mma m64n8k8 case")
    tcgen05_case.add_argument("--a", required=True, help="eight comma-separated uint32 hex words")
    tcgen05_case.add_argument("--b", required=True, help="eight comma-separated uint32 hex words")
    tcgen05_case.add_argument("--c", required=True, help="one uint32 hex word")
    tcgen05_case.add_argument("--dump-all", action="store_true")
    tcgen05_case.set_defaults(func=cmd_tcgen05_run_case)

    tcgen05_inspect = sub.add_parser("tcgen05-inspect", help="print SASS lines for the tcgen05 runner")
    tcgen05_inspect.set_defaults(func=cmd_tcgen05_inspect)

    tcgen05_bf16_random = sub.add_parser("tcgen05-bf16-random", help="random raw-bit validation for tcgen05 BF16 models")
    tcgen05_bf16_random.add_argument("--cases", type=int, default=1000)
    tcgen05_bf16_random.add_argument("--seed", type=int, default=1)
    tcgen05_bf16_random.add_argument("--show", type=int, default=8)
    tcgen05_bf16_random.add_argument("--finite-only", action="store_true")
    tcgen05_bf16_random.add_argument(
        "--model",
        choices=[
            "bf16-raw-window-rz",
            "hardware-oracle",
        ],
        default="bf16-raw-window-rz",
    )
    tcgen05_bf16_random.set_defaults(func=cmd_tcgen05_bf16_random)

    tcgen05_bf16_case = sub.add_parser("tcgen05-bf16-run-case", help="run one raw-bit tcgen05.mma BF16 m64n8k16 case")
    tcgen05_bf16_case.add_argument("--a", required=True, help="sixteen comma-separated uint32 BF16-container words")
    tcgen05_bf16_case.add_argument("--b", required=True, help="sixteen comma-separated uint32 BF16-container words")
    tcgen05_bf16_case.add_argument("--c", required=True, help="one uint32 F32 accumulator word")
    tcgen05_bf16_case.add_argument("--dump-all", action="store_true")
    tcgen05_bf16_case.set_defaults(func=cmd_tcgen05_bf16_run_case)

    tcgen05_bf16_inspect = sub.add_parser("tcgen05-bf16-inspect", help="print SASS lines for the tcgen05 BF16 runner")
    tcgen05_bf16_inspect.set_defaults(func=cmd_tcgen05_bf16_inspect)

    tcgen05_f16_random = sub.add_parser("tcgen05-f16-random", help="random raw-bit validation for tcgen05 F16 models")
    tcgen05_f16_random.add_argument("--cases", type=int, default=1000)
    tcgen05_f16_random.add_argument("--seed", type=int, default=1)
    tcgen05_f16_random.add_argument("--show", type=int, default=8)
    tcgen05_f16_random.add_argument("--finite-only", action="store_true")
    tcgen05_f16_random.add_argument(
        "--model",
        choices=[
            "f16-raw-window-rz",
            "hardware-oracle",
        ],
        default="f16-raw-window-rz",
    )
    tcgen05_f16_random.set_defaults(func=cmd_tcgen05_f16_random)

    tcgen05_f16_case = sub.add_parser("tcgen05-f16-run-case", help="run one raw-bit tcgen05.mma F16 m64n8k16 case")
    tcgen05_f16_case.add_argument("--a", required=True, help="sixteen comma-separated uint32 F16-container words")
    tcgen05_f16_case.add_argument("--b", required=True, help="sixteen comma-separated uint32 F16-container words")
    tcgen05_f16_case.add_argument("--c", required=True, help="one uint32 F32 accumulator word")
    tcgen05_f16_case.add_argument("--dump-all", action="store_true")
    tcgen05_f16_case.set_defaults(func=cmd_tcgen05_f16_run_case)

    tcgen05_f16_inspect = sub.add_parser("tcgen05-f16-inspect", help="print SASS lines for the tcgen05 F16 runner")
    tcgen05_f16_inspect.set_defaults(func=cmd_tcgen05_f16_inspect)

    tcgen05_fp8_random = sub.add_parser("tcgen05-fp8-random", help="random raw-bit validation for tcgen05 FP8 E4M3 models")
    tcgen05_fp8_random.add_argument("--cases", type=int, default=1000)
    tcgen05_fp8_random.add_argument("--seed", type=int, default=1)
    tcgen05_fp8_random.add_argument("--show", type=int, default=8)
    tcgen05_fp8_random.add_argument("--finite-only", action="store_true")
    tcgen05_fp8_random.add_argument(
        "--model",
        choices=[
            "fp8e4m3-raw-window-rz",
            "hardware-oracle",
        ],
        default="fp8e4m3-raw-window-rz",
    )
    tcgen05_fp8_random.set_defaults(func=cmd_tcgen05_fp8_random)

    tcgen05_fp8_case = sub.add_parser("tcgen05-fp8-run-case", help="run one raw-bit tcgen05.mma FP8 E4M3 m64n8k32 case")
    tcgen05_fp8_case.add_argument("--a", required=True, help="thirty-two comma-separated uint32 FP8-container words")
    tcgen05_fp8_case.add_argument("--b", required=True, help="thirty-two comma-separated uint32 FP8-container words")
    tcgen05_fp8_case.add_argument("--c", required=True, help="one uint32 F32 accumulator word")
    tcgen05_fp8_case.add_argument("--dump-all", action="store_true")
    tcgen05_fp8_case.set_defaults(func=cmd_tcgen05_fp8_run_case)

    tcgen05_fp8_inspect = sub.add_parser("tcgen05-fp8-inspect", help="print SASS lines for the tcgen05 FP8 runner")
    tcgen05_fp8_inspect.set_defaults(func=cmd_tcgen05_fp8_inspect)

    tcgen05_f8f6f4_random = sub.add_parser("tcgen05-f8f6f4-random", help="random raw-bit validation for tcgen05 f8/f6/f4 models")
    tcgen05_f8f6f4_random.add_argument("--format", choices=["e5m2", "e2m3", "e3m2", "e2m1"], required=True)
    tcgen05_f8f6f4_random.add_argument("--cases", type=int, default=1000)
    tcgen05_f8f6f4_random.add_argument("--seed", type=int, default=1)
    tcgen05_f8f6f4_random.add_argument("--show", type=int, default=8)
    tcgen05_f8f6f4_random.add_argument("--finite-only", action="store_true")
    tcgen05_f8f6f4_random.add_argument(
        "--model",
        choices=[
            "hardware-oracle",
            "raw-window-rz",
        ],
        default="raw-window-rz",
    )
    tcgen05_f8f6f4_random.set_defaults(func=cmd_tcgen05_f8f6f4_random)

    tcgen05_f8f6f4_case = sub.add_parser("tcgen05-f8f6f4-run-case", help="run one raw-bit tcgen05.mma f8/f6/f4 m64n8k32 case")
    tcgen05_f8f6f4_case.add_argument("--format", choices=["e5m2", "e2m3", "e3m2", "e2m1"], required=True)
    tcgen05_f8f6f4_case.add_argument("--a", required=True, help="thirty-two comma-separated uint32 element-container words")
    tcgen05_f8f6f4_case.add_argument("--b", required=True, help="thirty-two comma-separated uint32 element-container words")
    tcgen05_f8f6f4_case.add_argument("--c", required=True, help="one uint32 F32 accumulator word")
    tcgen05_f8f6f4_case.add_argument("--dump-all", action="store_true")
    tcgen05_f8f6f4_case.set_defaults(func=cmd_tcgen05_f8f6f4_run_case)

    tcgen05_f8f6f4_inspect = sub.add_parser("tcgen05-f8f6f4-inspect", help="print SASS lines for the tcgen05 f8/f6/f4 runner")
    tcgen05_f8f6f4_inspect.set_defaults(func=cmd_tcgen05_f8f6f4_inspect)

    tcgen05_i8_random = sub.add_parser("tcgen05-i8-random", help="random raw-bit validation for tcgen05 i8 S32 models")
    tcgen05_i8_random.add_argument("--a-type", choices=["u8", "s8"], required=True)
    tcgen05_i8_random.add_argument("--b-type", choices=["u8", "s8"], required=True)
    tcgen05_i8_random.add_argument("--saturate", action="store_true")
    tcgen05_i8_random.add_argument("--cases", type=int, default=1000)
    tcgen05_i8_random.add_argument("--batch-size", type=int, default=8192)
    tcgen05_i8_random.add_argument("--seed", type=int, default=1)
    tcgen05_i8_random.add_argument("--show", type=int, default=8)
    tcgen05_i8_random.add_argument(
        "--model",
        choices=[
            "hardware-oracle",
            "s32-exact",
        ],
        default="s32-exact",
    )
    tcgen05_i8_random.set_defaults(func=cmd_tcgen05_i8_random)

    tcgen05_i8_case = sub.add_parser("tcgen05-i8-run-case", help="run one raw-bit tcgen05.mma i8 m64n8k32 case")
    tcgen05_i8_case.add_argument("--a-type", choices=["u8", "s8"], required=True)
    tcgen05_i8_case.add_argument("--b-type", choices=["u8", "s8"], required=True)
    tcgen05_i8_case.add_argument("--saturate", action="store_true")
    tcgen05_i8_case.add_argument("--a", required=True, help="thirty-two comma-separated uint32 i8-container words")
    tcgen05_i8_case.add_argument("--b", required=True, help="thirty-two comma-separated uint32 i8-container words")
    tcgen05_i8_case.add_argument("--c", required=True, help="one uint32 S32 accumulator word")
    tcgen05_i8_case.add_argument("--dump-all", action="store_true")
    tcgen05_i8_case.set_defaults(func=cmd_tcgen05_i8_run_case)

    tcgen05_i8_inspect = sub.add_parser("tcgen05-i8-inspect", help="print SASS lines for the tcgen05 i8 runner")
    tcgen05_i8_inspect.set_defaults(func=cmd_tcgen05_i8_inspect)

    tcgen05_nvfp4_random = sub.add_parser("tcgen05-nvfp4-random", help="random raw-bit validation for tcgen05 NVFP4 models")
    tcgen05_nvfp4_random.add_argument("--cases", type=int, default=1000)
    tcgen05_nvfp4_random.add_argument("--seed", type=int, default=1)
    tcgen05_nvfp4_random.add_argument("--show", type=int, default=8)
    tcgen05_nvfp4_random.add_argument("--finite-only", action="store_true")
    tcgen05_nvfp4_random.add_argument("--random-scales", action="store_true", help="randomize raw UE4M3 A/B scale vectors, including NaN encodings, instead of using unit scales")
    tcgen05_nvfp4_random.add_argument("--zero-c", action="store_true", help="force the F32 accumulator input to +0")
    tcgen05_nvfp4_random.add_argument("--active-count", type=int, choices=range(65), metavar="N", help="randomly keep N active K lanes and zero the rest")
    tcgen05_nvfp4_random.add_argument(
        "--model",
        choices=[
            "hardware-oracle",
            "nvfp4-block16-window-rz",
            "nvfp4-raw-window-rz",
        ],
        default="nvfp4-block16-window-rz",
    )
    tcgen05_nvfp4_random.set_defaults(func=cmd_tcgen05_nvfp4_random)

    tcgen05_mxf8f6f4_random = sub.add_parser(
        "tcgen05-mxf8f6f4-random",
        help="random raw-bit validation for tcgen05 MXF8/F6/F4 UE8M0 block-scaled models",
    )
    tcgen05_mxf8f6f4_random.add_argument("--format", choices=["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"], required=True)
    tcgen05_mxf8f6f4_random.add_argument("--a-format", choices=["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"])
    tcgen05_mxf8f6f4_random.add_argument("--b-format", choices=["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"])
    tcgen05_mxf8f6f4_random.add_argument("--cases", type=int, default=1000)
    tcgen05_mxf8f6f4_random.add_argument("--seed", type=int, default=1)
    tcgen05_mxf8f6f4_random.add_argument("--m", type=int, choices=[128, 256], default=128)
    tcgen05_mxf8f6f4_random.add_argument("--n", type=int, default=8, help="runtime N shape, 8..256 step 8")
    tcgen05_mxf8f6f4_random.add_argument("--cta-group", type=int, choices=[1, 2], default=1)
    tcgen05_mxf8f6f4_random.add_argument("--show", type=int, default=8)
    tcgen05_mxf8f6f4_random.add_argument("--finite-only", action="store_true")
    tcgen05_mxf8f6f4_random.add_argument("--random-scales", action="store_true", help="randomize raw UE8M0 A/B scale bytes, including NaN, instead of using unit scales")
    tcgen05_mxf8f6f4_random.add_argument("--zero-c", action="store_true", help="force the F32 accumulator input to +0")
    tcgen05_mxf8f6f4_random.add_argument("--active-count", type=int, choices=range(33), metavar="N", help="randomly keep N active K lanes and zero the rest")
    tcgen05_mxf8f6f4_random.add_argument(
        "--model",
        choices=[
            "hardware-oracle",
            "block-scaled-window-rz",
        ],
        default="block-scaled-window-rz",
    )
    tcgen05_mxf8f6f4_random.set_defaults(func=cmd_tcgen05_mxf8f6f4_random)

    tcgen05_block_scaled_random = sub.add_parser(
        "tcgen05-block-scaled-random",
        help="random raw-bit validation for tcgen05 MXF4-family UE8M0 block-scaled models",
    )
    tcgen05_block_scaled_random.add_argument(
        "--kind",
        choices=["mxf4", "mxf4nvf4-2x", "mxf4nvf4-4x", "mxf4nvf4-4x-ue4m3"],
        required=True,
    )
    tcgen05_block_scaled_random.add_argument("--cases", type=int, default=1000)
    tcgen05_block_scaled_random.add_argument("--seed", type=int, default=1)
    tcgen05_block_scaled_random.add_argument("--m", type=int, choices=[128, 256], default=128)
    tcgen05_block_scaled_random.add_argument("--n", type=int, default=8, help="runtime N shape, 8..256 step 8")
    tcgen05_block_scaled_random.add_argument("--cta-group", type=int, choices=[1, 2], default=1)
    tcgen05_block_scaled_random.add_argument("--show", type=int, default=8)
    tcgen05_block_scaled_random.add_argument("--finite-only", action="store_true")
    tcgen05_block_scaled_random.add_argument("--random-scales", action="store_true", help="randomize raw UE8M0 A/B scale bytes, including NaN, instead of using unit scales")
    tcgen05_block_scaled_random.add_argument("--zero-c", action="store_true", help="force the F32 accumulator input to +0")
    tcgen05_block_scaled_random.add_argument("--active-count", type=int, choices=range(65), metavar="N", help="randomly keep N active K lanes and zero the rest")
    tcgen05_block_scaled_random.add_argument(
        "--model",
        choices=[
            "hardware-oracle",
            "block-scaled-window-rz",
        ],
        default="block-scaled-window-rz",
    )
    tcgen05_block_scaled_random.set_defaults(func=cmd_tcgen05_block_scaled_random)

    tcgen05_block_scaled_sparse_random = sub.add_parser(
        "tcgen05-block-scaled-sparse-random",
        help="random validation for sparse tcgen05 UE8M0 block-scaled models",
    )
    tcgen05_block_scaled_sparse_random.add_argument(
        "--kind",
        choices=["mxf8f6f4", "mxf4", "mxf4nvf4-2x", "mxf4nvf4-4x", "mxf4nvf4-4x-ue4m3"],
        required=True,
    )
    tcgen05_block_scaled_sparse_random.add_argument(
        "--format",
        choices=["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"],
        default="e4m3",
        help="MXF8/F6/F4 descriptor tag; ignored for MXF4-family kinds",
    )
    tcgen05_block_scaled_sparse_random.add_argument("--a-format", choices=["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"])
    tcgen05_block_scaled_sparse_random.add_argument("--b-format", choices=["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"])
    tcgen05_block_scaled_sparse_random.add_argument("--cases", type=int, default=1000)
    tcgen05_block_scaled_sparse_random.add_argument("--seed", type=int, default=1)
    tcgen05_block_scaled_sparse_random.add_argument("--m", type=int, choices=[128, 256], default=128)
    tcgen05_block_scaled_sparse_random.add_argument("--n", type=int, default=8, help="runtime N shape, 8..256 step 8")
    tcgen05_block_scaled_sparse_random.add_argument("--cta-group", type=int, choices=[1, 2], default=1)
    tcgen05_block_scaled_sparse_random.add_argument("--show", type=int, default=8)
    tcgen05_block_scaled_sparse_random.add_argument("--finite-only", action="store_true")
    tcgen05_block_scaled_sparse_random.add_argument("--random-scales", action="store_true", help="randomize raw UE8M0 A/B scale bytes, including NaN, instead of using unit scales")
    tcgen05_block_scaled_sparse_random.add_argument("--zero-c", action="store_true", help="force the F32 accumulator input to +0")
    tcgen05_block_scaled_sparse_random.add_argument("--metadata-word", help="low sparse metadata word")
    tcgen05_block_scaled_sparse_random.add_argument("--metadata-word-hi", help="high sparse metadata word")
    tcgen05_block_scaled_sparse_random.add_argument("--random-metadata", action="store_true", help="randomize legal sparse metadata selectors")
    tcgen05_block_scaled_sparse_random.add_argument(
        "--model",
        choices=[
            "block-scaled-window-rz",
        ],
        default="block-scaled-window-rz",
    )
    tcgen05_block_scaled_sparse_random.set_defaults(func=cmd_tcgen05_block_scaled_sparse_random)

    tcgen05_shape_random = sub.add_parser(
        "tcgen05-shape-random",
        help="random validation for runtime M/N tcgen05 shapes",
    )
    shape_format_choices = ["tf32", "bf16", "f16", "fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1", "i8"]
    shape_operand_choices = shape_format_choices + ["u8", "s8"]
    tcgen05_shape_random.add_argument("--format", choices=shape_format_choices, required=True)
    tcgen05_shape_random.add_argument("--a-format", choices=shape_operand_choices, help="override Matrix A descriptor/input type")
    tcgen05_shape_random.add_argument("--b-format", choices=shape_operand_choices, help="override Matrix B descriptor/input type")
    tcgen05_shape_random.add_argument("--d-type", choices=["f32", "f16", "s32"], help="override D/C descriptor type")
    tcgen05_shape_random.add_argument("--m", type=int, choices=[32, 64, 128, 256], required=True)
    tcgen05_shape_random.add_argument("--n", type=int, choices=range(8, 257, 8), metavar="N", required=True)
    tcgen05_shape_random.add_argument("--cta-group", type=int, choices=[1, 2], default=1)
    tcgen05_shape_random.add_argument("--cases", type=int, default=1000)
    tcgen05_shape_random.add_argument("--batch-size", type=int, default=8192)
    tcgen05_shape_random.add_argument("--seed", type=int, default=1)
    tcgen05_shape_random.add_argument("--show", type=int, default=8)
    tcgen05_shape_random.add_argument("--finite-only", action="store_true")
    tcgen05_shape_random.add_argument("--ws", action="store_true", help="issue tcgen05.mma.ws instead of tcgen05.mma")
    tcgen05_shape_random.add_argument("--sparse", action="store_true", help="issue sparse-A tcgen05.mma.sp or tcgen05.mma.ws.sp")
    tcgen05_shape_random.add_argument("--saturate", action="store_true", help="i8 only: request saturating S32 accumulation")
    tcgen05_shape_random.add_argument("--metadata-word", help="sparse only: fixed low 32-bit metadata word")
    tcgen05_shape_random.add_argument(
        "--metadata-word-hi",
        help="sparse only: fixed high 32-bit metadata word for K64 paths; defaults to --metadata-word",
    )
    tcgen05_shape_random.add_argument(
        "--random-metadata",
        action="store_true",
        help="sparse only: randomize legal sparse metadata selectors per case",
    )
    tcgen05_shape_random.set_defaults(func=cmd_tcgen05_shape_random)

    tcgen05_shape_coordinate = sub.add_parser(
        "tcgen05-shape-coordinate-map",
        help="derive row/column coordinates for runtime-shape tcgen05 output positions",
    )
    tcgen05_shape_coordinate.add_argument("--format", choices=shape_format_choices, required=True)
    tcgen05_shape_coordinate.add_argument("--a-format", choices=shape_operand_choices, help="override Matrix A descriptor/input type")
    tcgen05_shape_coordinate.add_argument("--b-format", choices=shape_operand_choices, help="override Matrix B descriptor/input type")
    tcgen05_shape_coordinate.add_argument("--d-type", choices=["f32", "f16", "s32"], help="override D/C descriptor type")
    tcgen05_shape_coordinate.add_argument("--m", type=int, choices=[32, 64, 128, 256], required=True)
    tcgen05_shape_coordinate.add_argument("--n", type=int, choices=range(8, 257, 8), metavar="N", required=True)
    tcgen05_shape_coordinate.add_argument("--cta-group", type=int, choices=[1, 2], default=1)
    tcgen05_shape_coordinate.add_argument("--seed", type=int, default=1)
    tcgen05_shape_coordinate.add_argument("--show", type=int, default=16)
    tcgen05_shape_coordinate.add_argument(
        "--require-full",
        action="store_true",
        help="exit nonzero unless full coordinate coverage and exact coordinate-probe arithmetic both pass",
    )
    tcgen05_shape_coordinate.add_argument(
        "--random-payload-cases",
        type=int,
        default=None,
        help=(
            "also validate this many deterministic full-surface random A(row,k), B(col,k), and C payloads; "
            "--require-full defaults this to 2 unless this option is set explicitly"
        ),
    )
    tcgen05_shape_coordinate.add_argument("--ws", action="store_true", help="issue tcgen05.mma.ws instead of tcgen05.mma")
    tcgen05_shape_coordinate.add_argument("--sparse", action="store_true", help="issue sparse-A tcgen05.mma.sp or tcgen05.mma.ws.sp")
    tcgen05_shape_coordinate.add_argument("--saturate", action="store_true", help="i8 only: request saturating S32 accumulation")
    tcgen05_shape_coordinate.add_argument("--metadata-word", help="sparse only: fixed low 32-bit metadata word")
    tcgen05_shape_coordinate.add_argument(
        "--metadata-word-hi",
        help="sparse only: fixed high 32-bit metadata word for K64 paths; defaults to --metadata-word",
    )
    tcgen05_shape_coordinate.add_argument(
        "--random-metadata",
        action="store_true",
        help="sparse only: randomize legal sparse metadata selectors for the coordinate probe",
    )
    tcgen05_shape_coordinate.set_defaults(func=cmd_tcgen05_shape_coordinate_map)

    tcgen05_sparse_random = sub.add_parser("tcgen05-sparse-random", help="random validation for true tcgen05.mma.sp paths")
    tcgen05_sparse_random.add_argument(
        "--format",
        choices=["tf32", "bf16", "f16", "fp8", "e5m2", "e2m3", "e3m2", "e2m1", "nvfp4"],
        required=True,
    )
    tcgen05_sparse_random.add_argument("--cases", type=int, default=1000)
    tcgen05_sparse_random.add_argument("--seed", type=int, default=1)
    tcgen05_sparse_random.add_argument("--show", type=int, default=8)
    tcgen05_sparse_random.add_argument("--finite-only", action="store_true")
    tcgen05_sparse_random.add_argument("--metadata-word", help="fixed 32-bit sparse metadata word; default is 0x44444444")
    tcgen05_sparse_random.add_argument(
        "--metadata-word-hi",
        help="fixed high 32-bit sparse metadata word for K64/K128 paths; defaults to --metadata-word",
    )
    tcgen05_sparse_random.add_argument("--random-metadata", action="store_true", help="randomize legal sparse metadata selectors per case")
    tcgen05_sparse_random.add_argument("--random-scales", action="store_true", help="NVFP4 only: randomize UE4M3 A/B scale vectors instead of using unit scales")
    tcgen05_sparse_random.set_defaults(func=cmd_tcgen05_sparse_random)

    tcgen05_nvfp4_case = sub.add_parser("tcgen05-nvfp4-run-case", help="run one raw-bit tcgen05.mma NVFP4 m128n8k64 case")
    tcgen05_nvfp4_case.add_argument("--a", required=True, help="sixty-four comma-separated uint32 NVFP4-container words")
    tcgen05_nvfp4_case.add_argument("--b", required=True, help="sixty-four comma-separated uint32 NVFP4-container words")
    tcgen05_nvfp4_case.add_argument("--c", required=True, help="one uint32 F32 accumulator word")
    tcgen05_nvfp4_case.add_argument("--scale-a", help="four comma-separated 7-bit UE4M3 hex scale values")
    tcgen05_nvfp4_case.add_argument("--scale-b", help="four comma-separated 7-bit UE4M3 hex scale values")
    tcgen05_nvfp4_case.add_argument("--dump-all", action="store_true")
    tcgen05_nvfp4_case.set_defaults(func=cmd_tcgen05_nvfp4_run_case)

    tcgen05_nvfp4_inspect = sub.add_parser("tcgen05-nvfp4-inspect", help="print SASS lines for the tcgen05 NVFP4 runner")
    tcgen05_nvfp4_inspect.set_defaults(func=cmd_tcgen05_nvfp4_inspect)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

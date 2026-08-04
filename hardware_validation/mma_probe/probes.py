from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable

from .formats import (
    F32_NEG_INF,
    F32_NEG_ONE,
    F32_NEG_ZERO,
    F32_POS_INF,
    F32_POS_ONE,
    F32_POS_ZERO,
    bits_to_f32,
    f32_class,
    f32_neg,
    f32_pow2_bits,
    f32_to_bits,
)
from .harness import MmaHarness, ProbeCase


ONE = F32_POS_ONE
ZERO = F32_POS_ZERO


def make_case(a: Iterable[int], b: Iterable[int], c: int) -> ProbeCase:
    return ProbeCase(tuple(a), tuple(b), c)


def case_from_summands(summands: list[int]) -> ProbeCase:
    if len(summands) != 5:
        raise ValueError("expected four products plus c")
    return ProbeCase(tuple(summands[:4]), (ONE, ONE, ONE, ONE), summands[4])


def flatten_output(case_output: list[list[int]]) -> list[int]:
    return [word for lane in case_output for word in lane]


def first_word(case_output: list[list[int]]) -> int:
    return case_output[0][0]


def unique_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_output(case_output))


def bit_hex(bits: int) -> str:
    return f"0x{bits:08x}"


@dataclass
class ProbeReport:
    data: dict


def probe_determinism(harness: MmaHarness, repeats: int = 32) -> dict:
    case = case_from_summands([f32_pow2_bits(0), f32_pow2_bits(-8), f32_pow2_bits(-16), ZERO, f32_pow2_bits(-4)])
    outputs = harness.run([case] * repeats)
    first = flatten_output(outputs[0])
    deterministic = all(flatten_output(out) == first for out in outputs[1:])
    return {
        "repeats": repeats,
        "deterministic": deterministic,
        "first": bit_hex(first[0]),
        "unique_words": len(set(first)),
    }


def probe_position_independence(harness: MmaHarness, cases: int = 64, seed: int = 1) -> dict:
    rng = random.Random(seed)
    probes: list[ProbeCase] = []
    for _ in range(cases):
        exps = [rng.randint(-16, 16) for _ in range(4)]
        signs = [F32_NEG_ZERO if rng.randrange(2) else F32_POS_ZERO for _ in range(4)]
        products = [f32_pow2_bits(e) ^ (signs[i] & 0x8000_0000) for i, e in enumerate(exps)]
        c = f32_pow2_bits(rng.randint(-16, 16))
        probes.append(case_from_summands([*products, c]))
    outputs = harness.run(probes)
    failures = []
    for idx, out in enumerate(outputs):
        values = unique_words(out)
        if len(values) != 1:
            failures.append({"case": idx, "unique": [bit_hex(x) for x in sorted(values)[:8]]})
    return {
        "cases": cases,
        "position_independent": not failures,
        "failures": failures[:8],
    }


def probe_pretruncation(harness: MmaHarness) -> dict:
    base = F32_POS_ONE
    base_case_a = case_from_summands([base, ZERO, ZERO, ZERO, ZERO])
    base_case_b = make_case([base, ZERO, ZERO, ZERO], [base, ONE, ONE, ONE], ZERO)
    base_a_out, base_b_out = [first_word(x) for x in harness.run([base_case_a, base_case_b])]

    a_ignored: list[int] = []
    b_ignored: list[int] = []
    a_changed: list[int] = []
    b_changed: list[int] = []
    cases: list[ProbeCase] = []
    tags: list[tuple[str, int]] = []
    for bit in range(23):
        varied = base ^ (1 << bit)
        cases.append(case_from_summands([varied, ZERO, ZERO, ZERO, ZERO]))
        tags.append(("a", bit))
        cases.append(make_case([base, ZERO, ZERO, ZERO], [varied, ONE, ONE, ONE], ZERO))
        tags.append(("b", bit))
    outs = harness.run(cases)
    for tag, out in zip(tags, outs):
        kind, bit = tag
        changed = first_word(out) != (base_a_out if kind == "a" else base_b_out)
        if kind == "a":
            (a_changed if changed else a_ignored).append(bit)
        else:
            (b_changed if changed else b_ignored).append(bit)

    nan_low_payload = 0x7F80_0001
    hazard_cases = [
        case_from_summands([nan_low_payload, ZERO, ZERO, ZERO, ZERO]),
        make_case([base, ZERO, ZERO, ZERO], [nan_low_payload, ONE, ONE, ONE], ZERO),
    ]
    hazard_out = [first_word(x) for x in harness.run(hazard_cases)]
    return {
        "base_a": bit_hex(base_a_out),
        "base_b": bit_hex(base_b_out),
        "a_ignored_mantissa_bits": a_ignored,
        "a_changed_mantissa_bits": a_changed,
        "b_ignored_mantissa_bits": b_ignored,
        "b_changed_mantissa_bits": b_changed,
        "nan_low_payload_a": {"bits": bit_hex(hazard_out[0]), "class": f32_class(hazard_out[0])},
        "nan_low_payload_b": {"bits": bit_hex(hazard_out[1]), "class": f32_class(hazard_out[1])},
    }


def probe_swamping_order(harness: MmaHarness, x_exp: int = 30, y_exp: int = 0) -> dict:
    x = f32_pow2_bits(x_exp)
    neg_x = f32_neg(x)
    y = f32_pow2_bits(y_exp)
    cases: list[ProbeCase] = []
    pairs: list[tuple[int, int]] = []
    for i, j in itertools.permutations(range(5), 2):
        summands = [y] * 5
        summands[i] = x
        summands[j] = neg_x
        cases.append(case_from_summands(summands))
        pairs.append((i, j))
    outputs = harness.run(cases)
    observations = {}
    y_value = bits_to_f32(y)
    for pair, out in zip(pairs, outputs):
        bits = first_word(out)
        value = bits_to_f32(bits)
        count = None
        if y_value != 0 and value == value and abs(value / y_value) < 1000:
            count = round(value / y_value)
        observations[f"{pair[0]},{pair[1]}"] = {
            "bits": bit_hex(bits),
            "class": f32_class(bits),
            "surviving_y_count": count,
        }
    return {
        "summand_order": ["p0", "p1", "p2", "p3", "c"],
        "X": f"2^{x_exp}",
        "y": f"2^{y_exp}",
        "pairs": observations,
    }


def probe_precision(harness: MmaHarness, max_exp: int = 80) -> dict:
    cases: list[ProbeCase] = []
    eps_exps = list(range(1, max_exp + 1))
    for e in eps_exps:
        eps = f32_pow2_bits(-e)
        cases.append(case_from_summands([ONE, eps, ZERO, ZERO, ZERO]))
    outputs = harness.run(cases)
    one = F32_POS_ONE
    changed = []
    vanished = []
    for e, out in zip(eps_exps, outputs):
        bits = first_word(out)
        (changed if bits != one else vanished).append(e)
    last_surviving = max(changed) if changed else None
    first_vanished = min(vanished) if vanished else None
    return {
        "test": "1 + epsilon via p0 + p1",
        "last_surviving_epsilon": f"2^-{last_surviving}" if last_surviving is not None else None,
        "first_vanished_epsilon": f"2^-{first_vanished}" if first_vanished is not None else None,
        "changed_exponents": changed,
    }


def _offset_case(sign: int, terms: list[int]) -> ProbeCase:
    base = F32_POS_ONE if sign > 0 else F32_NEG_ONE
    signed_terms = [t if sign > 0 else f32_neg(t) for t in terms]
    products = [*signed_terms, ZERO, ZERO, ZERO][:4]
    return case_from_summands([*products, base])


def _scaled_pow2_bits(multiplier: float, exp: int) -> int:
    return f32_to_bits(math.ldexp(multiplier, exp))


def _single_offset_case(sign: int, offset_bits: int) -> ProbeCase:
    base = F32_POS_ONE if sign > 0 else F32_NEG_ONE
    term = offset_bits if sign > 0 else f32_neg(offset_bits)
    return case_from_summands([term, ZERO, ZERO, ZERO, base])


def probe_rounding(harness: MmaHarness, ulp_exp: int = -23) -> dict:
    # Encode each offset as one TF32-representable product. Splitting 0.75u
    # into 0.5u + 0.25u would probe per-summand swamping instead.
    u = _scaled_pow2_bits(1.0, ulp_exp)
    quarter = _scaled_pow2_bits(1.0, ulp_exp - 2)
    half = _scaled_pow2_bits(1.0, ulp_exp - 1)
    three_quarters = _scaled_pow2_bits(1.5, ulp_exp - 1)
    one_and_half = _scaled_pow2_bits(1.5, ulp_exp)
    cases = {
        "+0.75u": _single_offset_case(+1, three_quarters),
        "+0.25u": _single_offset_case(+1, quarter),
        "-0.75u": _single_offset_case(-1, three_quarters),
        "-0.25u": _single_offset_case(-1, quarter),
        "+0.5u": _single_offset_case(+1, half),
        "+1.5u": _single_offset_case(+1, one_and_half),
        "-0.5u": _single_offset_case(-1, half),
        "-1.5u": _single_offset_case(-1, one_and_half),
    }
    outputs = harness.run(cases.values())
    observed = {
        name: {"bits": bit_hex(first_word(out)), "class": f32_class(first_word(out))}
        for name, out in zip(cases, outputs)
    }
    return {
        "test": "c = +/-1 plus one TF32 product at an exact fractional ULP offset",
        "ulp": f"2^{ulp_exp}",
        "offset_encoding": "single TF32 product plus c",
        "note": "This isolates the rounding/alignment behavior visible at this summand scale; full per-site mode inference still requires site-specific constructions.",
        "observed": observed,
    }


def probe_special_values(harness: MmaHarness) -> dict:
    half_min_normal = 0x0040_0000
    min_normal = 0x0080_0000
    cases = {
        "subnormal_c_half_min_normal": case_from_summands([ZERO, ZERO, ZERO, ZERO, half_min_normal]),
        "subnormal_a_half_min_normal": case_from_summands([half_min_normal, ZERO, ZERO, ZERO, ZERO]),
        "subnormal_product_half_times_min_normal": make_case(
            [f32_to_bits(0.5), ZERO, ZERO, ZERO], [min_normal, ONE, ONE, ONE], ZERO
        ),
        "negative_zero_generation": make_case(
            [F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO],
            [F32_NEG_ZERO, F32_NEG_ZERO, F32_NEG_ZERO, F32_NEG_ZERO],
            F32_NEG_ZERO,
        ),
        "positive_inf_product": case_from_summands([F32_POS_INF, ZERO, ZERO, ZERO, ZERO]),
        "negative_inf_product": case_from_summands([F32_NEG_INF, ZERO, ZERO, ZERO, ZERO]),
        "zero_times_inf": make_case([F32_POS_ZERO, ZERO, ZERO, ZERO], [F32_POS_INF, ONE, ONE, ONE], ZERO),
        "pos_inf_plus_neg_inf": case_from_summands([F32_POS_INF, ZERO, ZERO, ZERO, F32_NEG_INF]),
        "quiet_nan_product": case_from_summands([0x7FC1_2345, ZERO, ZERO, ZERO, ZERO]),
        "signaling_nan_product": case_from_summands([0x7F81_2345, ZERO, ZERO, ZERO, ZERO]),
    }
    outputs = harness.run(cases.values())
    return {
        name: {"bits": bit_hex(first_word(out)), "class": f32_class(first_word(out))}
        for name, out in zip(cases, outputs)
    }


def run_quick_probes(harness: MmaHarness) -> dict:
    return {
        "runner_info": harness.info(),
        "determinism": probe_determinism(harness),
        "position_independence": probe_position_independence(harness),
        "pretruncation": probe_pretruncation(harness),
        "swamping_order": probe_swamping_order(harness),
        "precision": probe_precision(harness),
        "rounding": probe_rounding(harness),
        "special_values": probe_special_values(harness),
    }

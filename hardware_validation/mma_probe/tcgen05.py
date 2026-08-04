from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
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


INPUT_MAGIC = b"MMAPRB1\0"
OUTPUT_MAGIC = b"MMAPRO1\0"
VERSION = 1
OP_TCGEN05_TF32_M64N8K8 = 2
OP_TCGEN05_BF16_M64N8K16 = 3
OP_TCGEN05_FP8_M64N8K32 = 4
OP_TCGEN05_NVFP4_M128N8K64 = 5
OP_TCGEN05_TF32_SPARSE_M64N8K16 = 6
OP_TCGEN05_BF16_SPARSE_M64N8K32 = 7
OP_TCGEN05_FP8_SPARSE_M64N8K64 = 8
OP_TCGEN05_NVFP4_SPARSE_M128N8K128 = 9
OP_TCGEN05_F16_M64N8K16 = 10
OP_TCGEN05_F16_SPARSE_M64N8K32 = 11
OP_TCGEN05_F8F6F4_E5M2_M64N8K32 = 12
OP_TCGEN05_F8F6F4_E2M3_M64N8K32 = 13
OP_TCGEN05_F8F6F4_E3M2_M64N8K32 = 14
OP_TCGEN05_F8F6F4_E2M1_M64N8K32 = 15
OP_TCGEN05_F8F6F4_E5M2_SPARSE_M64N8K64 = 16
OP_TCGEN05_F8F6F4_E2M3_SPARSE_M64N8K64 = 17
OP_TCGEN05_F8F6F4_E3M2_SPARSE_M64N8K64 = 18
OP_TCGEN05_F8F6F4_E2M1_SPARSE_M64N8K64 = 19
OP_TCGEN05_MXF8F6F4_E4M3_UE8M0_M128N8K32 = 20
OP_TCGEN05_MXF8F6F4_E5M2_UE8M0_M128N8K32 = 21
OP_TCGEN05_MXF8F6F4_E2M3_UE8M0_M128N8K32 = 22
OP_TCGEN05_MXF8F6F4_E3M2_UE8M0_M128N8K32 = 23
OP_TCGEN05_MXF8F6F4_E2M1_UE8M0_M128N8K32 = 24
OP_TCGEN05_MXF4_E2M1_UE8M0_M128N8K64 = 25
OP_TCGEN05_MXF4NVF4_E2M1_UE8M0_M128N8K64_SCALE2X = 26
OP_TCGEN05_MXF4NVF4_E2M1_UE8M0_M128N8K64_SCALE4X = 27
OP_TCGEN05_SHAPE = 28
OP_TCGEN05_I8_U8U8_M64N8K32 = 29
OP_TCGEN05_I8_S8U8_M64N8K32 = 30
OP_TCGEN05_I8_U8S8_M64N8K32 = 31
OP_TCGEN05_I8_S8S8_M64N8K32 = 32
OP_TCGEN05_I8_U8U8_SAT_M64N8K32 = 33
OP_TCGEN05_I8_S8U8_SAT_M64N8K32 = 34
OP_TCGEN05_I8_U8S8_SAT_M64N8K32 = 35
OP_TCGEN05_I8_S8S8_SAT_M64N8K32 = 36
OP_TCGEN05_MXF4NVF4_E2M1_UE4M3_M128N8K64_SCALE4X = 37
OP_TCGEN05_MXF4NVF4_E2M1_UE4M3_SPARSE_M128N8K128_SCALE4X = 38
THREADS = 128
CTA_GROUP2_THREADS = THREADS * 2
REGS_PER_THREAD = 4
VALID_THREADS = 32
BF16_VALID_POSITIONS = tuple(
    [(thread, reg) for thread in range(16) for reg in range(4)]
    + [(thread, reg) for thread in range(16, 32) for reg in range(2)]
)
FP8_VALID_POSITIONS = tuple(
    [(thread, reg) for thread in (0, 1, 4, 5) for reg in range(4)]
    + [(thread, reg) for thread in (8, 9, 12, 13) for reg in range(2)]
)
NVFP4_VALID_POSITIONS = ((0, 0),)
NVFP4_UNIT_SCALES = (0x38, 0x38, 0x38, 0x38)
NVFP4_SPARSE_UNIT_SCALES = (0x38,) * 4
MX_UE8M0_UNIT_SCALES = (0x7F,) * 4
SPARSE_FIXED_METADATA = 0x4444_4444
SPARSE_TF32_SELECTORS = (0x4, 0xE)
SPARSE_2OF4_SELECTORS = (0x4, 0x8, 0xC, 0x9, 0xD, 0x6, 0xE)
SHAPE_COORDINATE_FLAGS = {
    None: 0,
    "row": 16,
    "col": 32,
    "all": 64,
    "random": 128,
}
F8F6F4_FORMAT_OPS = {
    "e5m2": OP_TCGEN05_F8F6F4_E5M2_M64N8K32,
    "e2m3": OP_TCGEN05_F8F6F4_E2M3_M64N8K32,
    "e3m2": OP_TCGEN05_F8F6F4_E3M2_M64N8K32,
    "e2m1": OP_TCGEN05_F8F6F4_E2M1_M64N8K32,
}
F8F6F4_SPARSE_FORMAT_OPS = {
    "e5m2": OP_TCGEN05_F8F6F4_E5M2_SPARSE_M64N8K64,
    "e2m3": OP_TCGEN05_F8F6F4_E2M3_SPARSE_M64N8K64,
    "e3m2": OP_TCGEN05_F8F6F4_E3M2_SPARSE_M64N8K64,
    "e2m1": OP_TCGEN05_F8F6F4_E2M1_SPARSE_M64N8K64,
}
I8_FORMAT_OPS = {
    ("u8", "u8", False): OP_TCGEN05_I8_U8U8_M64N8K32,
    ("s8", "u8", False): OP_TCGEN05_I8_S8U8_M64N8K32,
    ("u8", "s8", False): OP_TCGEN05_I8_U8S8_M64N8K32,
    ("s8", "s8", False): OP_TCGEN05_I8_S8S8_M64N8K32,
    ("u8", "u8", True): OP_TCGEN05_I8_U8U8_SAT_M64N8K32,
    ("s8", "u8", True): OP_TCGEN05_I8_S8U8_SAT_M64N8K32,
    ("u8", "s8", True): OP_TCGEN05_I8_U8S8_SAT_M64N8K32,
    ("s8", "s8", True): OP_TCGEN05_I8_S8S8_SAT_M64N8K32,
}
MXF8F6F4_UE8M0_FORMAT_OPS = {
    "e4m3": OP_TCGEN05_MXF8F6F4_E4M3_UE8M0_M128N8K32,
    "e5m2": OP_TCGEN05_MXF8F6F4_E5M2_UE8M0_M128N8K32,
    "e2m3": OP_TCGEN05_MXF8F6F4_E2M3_UE8M0_M128N8K32,
    "e3m2": OP_TCGEN05_MXF8F6F4_E3M2_UE8M0_M128N8K32,
    "e2m1": OP_TCGEN05_MXF8F6F4_E2M1_UE8M0_M128N8K32,
}
MXF8F6F4_FORMAT_CODES = {
    "e4m3": 0,
    "e5m2": 1,
    "e2m3": 3,
    "e3m2": 4,
    "e2m1": 5,
}
BLOCK_SCALED_OPS = {
    "mxf4": OP_TCGEN05_MXF4_E2M1_UE8M0_M128N8K64,
    "mxf4nvf4-2x": OP_TCGEN05_MXF4NVF4_E2M1_UE8M0_M128N8K64_SCALE2X,
    "mxf4nvf4-4x": OP_TCGEN05_MXF4NVF4_E2M1_UE8M0_M128N8K64_SCALE4X,
    "mxf4nvf4-4x-ue4m3": OP_TCGEN05_MXF4NVF4_E2M1_UE4M3_M128N8K64_SCALE4X,
}
BLOCK_SCALED_SPARSE_OPS = {
    "mxf4": OP_TCGEN05_MXF4_E2M1_UE8M0_M128N8K64,
    "mxf4nvf4-2x": OP_TCGEN05_MXF4NVF4_E2M1_UE8M0_M128N8K64_SCALE2X,
    "mxf4nvf4-4x": OP_TCGEN05_MXF4NVF4_E2M1_UE8M0_M128N8K64_SCALE4X,
    "mxf4nvf4-4x-ue4m3": OP_TCGEN05_MXF4NVF4_E2M1_UE4M3_SPARSE_M128N8K128_SCALE4X,
}
SHAPE_FORMATS = {
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
}
SHAPE_I8_TYPES = {
    "u8": 0,
    "s8": 1,
}
SHAPE_D_TYPES = {
    "f16": 0,
    "f32": 1,
    "s32": 2,
}
SHAPE_FORMAT_K = {
    "tf32": 8,
    "bf16": 16,
    "f16": 16,
    "fp8": 32,
    "e4m3": 32,
    "e5m2": 32,
    "e2m3": 32,
    "e3m2": 32,
    "e2m1": 32,
    "i8": 32,
}
SHAPE_FORMAT_SPARSE_K = {
    "tf32": 16,
    "bf16": 32,
    "f16": 32,
    "fp8": 64,
    "e4m3": 64,
    "e5m2": 64,
    "e2m3": 64,
    "e3m2": 64,
    "e2m1": 64,
    "i8": 64,
}
CASE_STRUCT = struct.Struct("<17I")
BF16_CASE_STRUCT = struct.Struct("<33I")
F16_CASE_STRUCT = struct.Struct("<33I")
FP8_CASE_STRUCT = struct.Struct("<65I")
F8F6F4_CASE_STRUCT = struct.Struct("<65I")
I8_CASE_STRUCT = struct.Struct("<65I")
NVFP4_CASE_STRUCT = struct.Struct("<137I")
BLOCK_SCALED_CASE_STRUCT = struct.Struct("<142I")
BLOCK_SCALED_SPARSE_CASE_STRUCT = struct.Struct("<272I")
TF32_SPARSE_CASE_STRUCT = struct.Struct("<34I")
BF16_SPARSE_CASE_STRUCT = struct.Struct("<66I")
F16_SPARSE_CASE_STRUCT = struct.Struct("<66I")
FP8_SPARSE_CASE_STRUCT = struct.Struct("<131I")
F8F6F4_SPARSE_CASE_STRUCT = struct.Struct("<131I")
NVFP4_SPARSE_CASE_STRUCT = struct.Struct("<267I")
SHAPE_CASE_STRUCT = struct.Struct("<138I")
INPUT_HEADER = struct.Struct("<8sIIQ")
OUTPUT_HEADER = struct.Struct("<8sIIIIQ")


@dataclass(frozen=True)
class Tcgen05ProbeCase:
    a: tuple[int, int, int, int, int, int, int, int]
    b: tuple[int, int, int, int, int, int, int, int]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 8 or len(self.b) != 8:
            raise ValueError("Tcgen05ProbeCase requires eight A and eight B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05Bf16ProbeCase:
    a: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    b: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 16 or len(self.b) != 16:
            raise ValueError("Tcgen05Bf16ProbeCase requires sixteen A and sixteen B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05F16ProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 16 or len(self.b) != 16:
            raise ValueError("Tcgen05F16ProbeCase requires sixteen A and sixteen B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05Fp8ProbeCase:
    a: tuple[
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    ]
    b: tuple[
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    ]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 32 or len(self.b) != 32:
            raise ValueError("Tcgen05Fp8ProbeCase requires thirty-two A and thirty-two B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05F8f6f4ProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 32 or len(self.b) != 32:
            raise ValueError("Tcgen05F8f6f4ProbeCase requires thirty-two A and thirty-two B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05I8ProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 32 or len(self.b) != 32:
            raise ValueError("Tcgen05I8ProbeCase requires thirty-two A and thirty-two B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05Nvfp4ProbeCase:
    a: tuple[
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    ]
    b: tuple[
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    ]
    c: int
    scale_a: tuple[int, int, int, int] = NVFP4_UNIT_SCALES
    scale_b: tuple[int, int, int, int] = NVFP4_UNIT_SCALES

    def __post_init__(self) -> None:
        if len(self.a) != 64 or len(self.b) != 64:
            raise ValueError("Tcgen05Nvfp4ProbeCase requires sixty-four A and sixty-four B values")
        if len(self.scale_a) != 4 or len(self.scale_b) != 4:
            raise ValueError("Tcgen05Nvfp4ProbeCase requires four A and four B UE4M3 scale values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")
        for value in (*self.scale_a, *self.scale_b):
            if value < 0 or value > 0x7F:
                raise ValueError(f"UE4M3 scale value out of 7-bit range: {value!r}")


@dataclass(frozen=True)
class Tcgen05BlockScaledProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    scale_a: tuple[int, ...] = MX_UE8M0_UNIT_SCALES
    scale_b: tuple[int, ...] = MX_UE8M0_UNIT_SCALES
    a_format: str = "e2m1"
    b_format: str = "e2m1"
    n: int = 8
    cta_group: int = 1
    m: int = 128

    def __post_init__(self) -> None:
        if len(self.a) != 64 or len(self.b) != 64:
            raise ValueError("Tcgen05BlockScaledProbeCase requires sixty-four A and B values")
        if self.m not in (128, 256):
            raise ValueError(f"unsupported block-scaled M shape: {self.m}")
        if self.cta_group == 1 and self.m != 128:
            raise ValueError("cta_group::1 block-scaled probes require M128")
        if self.cta_group not in (1, 2):
            raise ValueError(f"unsupported block-scaled cta_group: {self.cta_group}")
        if self.n < 8 or self.n > 256 or self.n % 8 != 0:
            raise ValueError(f"unsupported block-scaled N shape: {self.n}")
        if self.cta_group == 2 and (self.n < 16 or self.n % 16 != 0):
            raise ValueError(f"unsupported cta_group::2 block-scaled N shape: {self.n}")
        if len(self.scale_a) != 4 or len(self.scale_b) != 4:
            raise ValueError("Tcgen05BlockScaledProbeCase requires four A and four B scale values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")
        for value in (*self.scale_a, *self.scale_b):
            if value < 0 or value > 0xFF:
                raise ValueError(f"scale value out of 8-bit range: {value!r}")
        if self.a_format not in MXF8F6F4_FORMAT_CODES:
            raise ValueError(f"unsupported MXF8/F6/F4 A format: {self.a_format!r}")
        if self.b_format not in MXF8F6F4_FORMAT_CODES:
            raise ValueError(f"unsupported MXF8/F6/F4 B format: {self.b_format!r}")


@dataclass(frozen=True)
class Tcgen05BlockScaledSparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA
    metadata_hi: int = SPARSE_FIXED_METADATA
    scale_a: tuple[int, ...] = MX_UE8M0_UNIT_SCALES
    scale_b: tuple[int, ...] = MX_UE8M0_UNIT_SCALES
    a_format: str = "e2m1"
    b_format: str = "e2m1"
    n: int = 8
    cta_group: int = 1
    m: int = 128

    def __post_init__(self) -> None:
        if len(self.a) != 128 or len(self.b) != 128:
            raise ValueError("Tcgen05BlockScaledSparseProbeCase requires 128 A and B values")
        if self.m not in (128, 256):
            raise ValueError(f"unsupported sparse block-scaled M shape: {self.m}")
        if self.cta_group == 1 and self.m != 128:
            raise ValueError("cta_group::1 sparse block-scaled probes require M128")
        if self.cta_group not in (1, 2):
            raise ValueError(f"unsupported sparse block-scaled cta_group: {self.cta_group}")
        if self.cta_group == 2 and self.m != 256:
            raise ValueError("sparse block-scaled cta_group::2 probes require M256")
        if self.n < 8 or self.n > 256 or self.n % 8 != 0:
            raise ValueError(f"unsupported sparse block-scaled N shape: {self.n}")
        if self.cta_group == 2 and (self.n < 16 or self.n % 16 != 0):
            raise ValueError(f"unsupported cta_group::2 sparse block-scaled N shape: {self.n}")
        if len(self.scale_a) != 4 or len(self.scale_b) != 4:
            raise ValueError("Tcgen05BlockScaledSparseProbeCase requires four A and four B scale values")
        _validate_sparse_metadata(self.metadata, SPARSE_2OF4_SELECTORS)
        _validate_sparse_metadata(self.metadata_hi, SPARSE_2OF4_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")
        for value in (*self.scale_a, *self.scale_b):
            if value < 0 or value > 0xFF:
                raise ValueError(f"scale value out of 8-bit range: {value!r}")
        if self.a_format not in MXF8F6F4_FORMAT_CODES:
            raise ValueError(f"unsupported MXF8/F6/F4 A format: {self.a_format!r}")
        if self.b_format not in MXF8F6F4_FORMAT_CODES:
            raise ValueError(f"unsupported MXF8/F6/F4 B format: {self.b_format!r}")


def _validate_sparse_metadata(metadata: int, legal_selectors: tuple[int, ...]) -> None:
    if metadata < 0 or metadata > 0xFFFF_FFFF:
        raise ValueError(f"sparse metadata out of uint32 range: {metadata!r}")
    legal = set(legal_selectors)
    for chunk in range(8):
        selector = (metadata >> (chunk * 4)) & 0xF
        if selector not in legal:
            raise ValueError(f"illegal sparse metadata selector 0x{selector:x} in word 0x{metadata:08x}")


@dataclass(frozen=True)
class Tcgen05Tf32SparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA

    def __post_init__(self) -> None:
        if len(self.a) != 16 or len(self.b) != 16:
            raise ValueError("Tcgen05Tf32SparseProbeCase requires sixteen A and sixteen B values")
        _validate_sparse_metadata(self.metadata, SPARSE_TF32_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05Bf16SparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA

    def __post_init__(self) -> None:
        if len(self.a) != 32 or len(self.b) != 32:
            raise ValueError("Tcgen05Bf16SparseProbeCase requires thirty-two A and thirty-two B values")
        _validate_sparse_metadata(self.metadata, SPARSE_2OF4_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05F16SparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA

    def __post_init__(self) -> None:
        if len(self.a) != 32 or len(self.b) != 32:
            raise ValueError("Tcgen05F16SparseProbeCase requires thirty-two A and thirty-two B values")
        _validate_sparse_metadata(self.metadata, SPARSE_2OF4_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05Fp8SparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA
    metadata_hi: int = SPARSE_FIXED_METADATA

    def __post_init__(self) -> None:
        if len(self.a) != 64 or len(self.b) != 64:
            raise ValueError("Tcgen05Fp8SparseProbeCase requires sixty-four A and sixty-four B values")
        _validate_sparse_metadata(self.metadata, SPARSE_2OF4_SELECTORS)
        _validate_sparse_metadata(self.metadata_hi, SPARSE_2OF4_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05F8f6f4SparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA
    metadata_hi: int = SPARSE_FIXED_METADATA

    def __post_init__(self) -> None:
        if len(self.a) != 64 or len(self.b) != 64:
            raise ValueError("Tcgen05F8f6f4SparseProbeCase requires sixty-four A and sixty-four B values")
        _validate_sparse_metadata(self.metadata, SPARSE_2OF4_SELECTORS)
        _validate_sparse_metadata(self.metadata_hi, SPARSE_2OF4_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


@dataclass(frozen=True)
class Tcgen05Nvfp4SparseProbeCase:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    metadata: int = SPARSE_FIXED_METADATA
    metadata_hi: int = SPARSE_FIXED_METADATA
    scale_a: tuple[int, ...] = NVFP4_SPARSE_UNIT_SCALES
    scale_b: tuple[int, ...] = NVFP4_SPARSE_UNIT_SCALES

    def __post_init__(self) -> None:
        if len(self.a) != 128 or len(self.b) != 128:
            raise ValueError("Tcgen05Nvfp4SparseProbeCase requires one hundred twenty-eight A and B values")
        if len(self.scale_a) != 4 or len(self.scale_b) != 4:
            raise ValueError("Tcgen05Nvfp4SparseProbeCase requires four A and four B UE4M3 scale values")
        _validate_sparse_metadata(self.metadata, SPARSE_2OF4_SELECTORS)
        _validate_sparse_metadata(self.metadata_hi, SPARSE_2OF4_SELECTORS)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")
        for value in (*self.scale_a, *self.scale_b):
            if value < 0 or value > 0x7F:
                raise ValueError(f"UE4M3 scale value out of 7-bit range: {value!r}")


@dataclass(frozen=True)
class Tcgen05ShapeProbeCase:
    format_name: str
    m: int
    n: int
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: int
    ws: bool = False
    cta_group: int = 1
    sparse: bool = False
    a_format_name: str | None = None
    b_format_name: str | None = None
    d_type: str = "f32"
    saturate: bool = False
    metadata: int = SPARSE_FIXED_METADATA
    metadata_hi: int = SPARSE_FIXED_METADATA
    coordinate: str | None = None

    def __post_init__(self) -> None:
        if self.format_name not in SHAPE_FORMATS:
            raise ValueError(f"unsupported shape format: {self.format_name}")
        a_format_name = self.a_format_name or ("u8" if self.format_name == "i8" else self.format_name)
        b_format_name = self.b_format_name or ("u8" if self.format_name == "i8" else self.format_name)
        if self.d_type not in SHAPE_D_TYPES:
            raise ValueError(f"unsupported D shape type: {self.d_type}")
        if self.coordinate not in SHAPE_COORDINATE_FLAGS:
            raise ValueError(f"unsupported coordinate probe kind: {self.coordinate!r}")
        if self.coordinate in ("row", "col") and self.c > 31:
            raise ValueError("coordinate probe bit index must fit in 0..31")
        if self.format_name == "i8":
            if a_format_name not in SHAPE_I8_TYPES:
                raise ValueError(f"unsupported i8 A type: {a_format_name}")
            if b_format_name not in SHAPE_I8_TYPES:
                raise ValueError(f"unsupported i8 B type: {b_format_name}")
            if self.d_type != "s32":
                raise ValueError("kind::i8 shape probes require S32 D")
        else:
            if self.saturate:
                raise ValueError("shape saturation flag is only valid for kind::i8")
            if self.d_type == "s32":
                raise ValueError("S32 D shape probes are only valid for kind::i8")
            if a_format_name not in SHAPE_FORMATS:
                raise ValueError(f"unsupported A shape format: {a_format_name}")
            if b_format_name not in SHAPE_FORMATS:
                raise ValueError(f"unsupported B shape format: {b_format_name}")
        f16_family = {"bf16", "f16"}
        f8f6f4_family = {"fp8", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1"}
        if self.format_name == "i8":
            pass
        elif self.format_name == "tf32":
            if a_format_name != "tf32" or b_format_name != "tf32" or self.d_type != "f32":
                raise ValueError("TF32 shape probes require TF32 A/B and F32 D")
        elif self.format_name in f16_family:
            if a_format_name not in f16_family or b_format_name not in f16_family:
                raise ValueError("kind::f16 shape probes require BF16/F16 A/B formats")
            if a_format_name != b_format_name:
                raise ValueError(
                    "B200 rejects off-diagonal BF16/F16 kind::f16 A/B descriptors with an illegal instruction"
                )
            if self.d_type == "f16" and a_format_name != "f16":
                raise ValueError("B200 kind::f16 F16-D shape probes require F16 A/B inputs")
        elif self.format_name in f8f6f4_family:
            if a_format_name not in f8f6f4_family or b_format_name not in f8f6f4_family:
                raise ValueError("kind::f8f6f4 shape probes require f8/f6/f4 A/B formats")
        if self.cta_group not in (1, 2):
            raise ValueError(f"unsupported cta_group: {self.cta_group}")
        if self.format_name == "i8":
            def valid_i8_n(n: int) -> bool:
                return n in (8, 16, 24, 32) or (48 <= n <= 256 and n % 16 == 0)

            if self.sparse:
                if self.cta_group == 2:
                    if self.ws:
                        raise ValueError("cta_group::2 sparse i8 shape probes do not support .ws")
                    if self.m not in (128, 256):
                        raise ValueError(f"unsupported cta_group::2 sparse i8 M shape: {self.m}")
                    if self.n < 32 or self.n > 256 or self.n % 32 != 0:
                        raise ValueError(f"unsupported cta_group::2 sparse i8 N shape: {self.n}")
                elif self.ws:
                    if self.m not in (32, 64, 128):
                        raise ValueError(f"unsupported cta_group::1 sparse i8 .ws M shape: {self.m}")
                    if self.n not in (64, 128):
                        raise ValueError(f"unsupported cta_group::1 sparse i8 .ws N shape: {self.n}")
                else:
                    if self.m not in (64, 128):
                        raise ValueError(f"unsupported cta_group::1 sparse i8 non-.ws M shape: {self.m}")
                    if not valid_i8_n(self.n):
                        raise ValueError(f"unsupported cta_group::1 sparse i8 N shape: {self.n}")
            elif self.cta_group == 2:
                if self.ws:
                    raise ValueError("cta_group::2 i8 shape probes do not support .ws")
                if self.m not in (128, 256):
                    raise ValueError(f"unsupported cta_group::2 dense i8 M shape: {self.m}")
                if self.n < 32 or self.n > 256 or self.n % 32 != 0:
                    raise ValueError(f"unsupported cta_group::2 dense i8 N shape: {self.n}")
            else:
                if self.ws:
                    if self.m not in (32, 64, 128):
                        raise ValueError(f"unsupported cta_group::1 dense i8 .ws M shape: {self.m}")
                    if self.n not in (64, 128, 256):
                        raise ValueError(f"unsupported cta_group::1 dense i8 .ws N shape: {self.n}")
                else:
                    if self.m not in (64, 128):
                        raise ValueError(f"unsupported cta_group::1 dense i8 non-.ws M shape: {self.m}")
                    if not valid_i8_n(self.n):
                        raise ValueError(f"unsupported cta_group::1 dense i8 N shape: {self.n}")
        elif self.sparse:
            if self.cta_group == 2:
                if self.ws:
                    raise ValueError("cta_group::2 sparse shape probes do not support .ws")
                if self.m not in (128, 256):
                    raise ValueError(f"unsupported cta_group::2 sparse M shape: {self.m}")
                if self.n < 16 or self.n > 256 or self.n % 16 != 0:
                    raise ValueError(f"unsupported cta_group::2 sparse N shape: {self.n}")
            elif self.ws:
                if self.m not in (32, 64, 128):
                    raise ValueError(f"unsupported cta_group::1 sparse .ws M shape: {self.m}")
                if self.n not in (64, 128):
                    raise ValueError(f"unsupported cta_group::1 sparse .ws N shape: {self.n}")
            else:
                if self.m not in (64, 128):
                    raise ValueError(f"unsupported cta_group::1 sparse non-.ws M shape: {self.m}")
                if self.n < 8 or self.n > 256 or self.n % 8 != 0:
                    raise ValueError(f"unsupported cta_group::1 sparse non-.ws N shape: {self.n}")
        elif self.cta_group == 2:
            if self.ws:
                raise ValueError("cta_group::2 shape probes do not support .ws")
            if self.m not in (128, 256):
                raise ValueError(f"unsupported cta_group::2 dense M shape: {self.m}")
            if self.n < 16 or self.n > 256 or self.n % 16 != 0:
                raise ValueError(f"unsupported cta_group::2 dense N shape: {self.n}")
        else:
            if self.ws:
                if self.m not in (32, 64, 128):
                    raise ValueError(f"unsupported cta_group::1 dense .ws M shape: {self.m}")
                if self.n not in (64, 128, 256):
                    raise ValueError(f"unsupported cta_group::1 dense .ws N shape: {self.n}")
            else:
                if self.m not in (64, 128):
                    raise ValueError(f"unsupported cta_group::1 dense non-.ws M shape: {self.m}")
                if self.n < 8 or self.n > 256 or self.n % 8 != 0:
                    raise ValueError(f"unsupported cta_group::1 dense non-.ws N shape: {self.n}")
        k = SHAPE_FORMAT_SPARSE_K[self.format_name] if self.sparse else SHAPE_FORMAT_K[self.format_name]
        if len(self.a) != k or len(self.b) != k:
            raise ValueError(f"Tcgen05ShapeProbeCase requires {k} A and B values")
        if self.sparse:
            selectors = SPARSE_TF32_SELECTORS if self.format_name == "tf32" else SPARSE_2OF4_SELECTORS
            _validate_sparse_metadata(self.metadata, selectors)
            _validate_sparse_metadata(self.metadata_hi, selectors)
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


class Tcgen05Harness:
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_tf32_probe",
        *,
        device: int = 0,
    ):
        self.runner = Path(runner)
        self.device = device

    def build(self) -> None:
        if self.runner.exists():
            return
        subprocess.run(["make", str(self.runner)], check=True)

    def info(self) -> str:
        self.build()
        result = subprocess.run(
            [str(self.runner), "--info", "--device", str(self.device)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout.strip()

    def run(self, cases: Iterable[Tcgen05ProbeCase]) -> list[list[list[int]]]:
        self.build()
        case_list = list(cases)
        with tempfile.TemporaryDirectory(prefix="tcgen05-probe-") as tmp:
            input_path = Path(tmp) / "cases.bin"
            output_path = Path(tmp) / "outputs.bin"
            self._write_cases(input_path, case_list)
            subprocess.run(
                [
                    str(self.runner),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--device",
                    str(self.device),
                ],
                check=True,
            )
            outputs = self._read_outputs(output_path)
            if len(outputs) != len(case_list):
                raise RuntimeError(
                    f"output case count mismatch: got {len(outputs)}, expected {len(case_list)}"
                )
            return outputs

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_TF32_M64N8K8, len(cases)))
            for case in cases:
                f.write(CASE_STRUCT.pack(*case.a, *case.b, case.c))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        with path.open("rb") as f:
            header = f.read(OUTPUT_HEADER.size)
            if len(header) != OUTPUT_HEADER.size:
                raise RuntimeError("short output header")
            magic, version, op, threads, regs_per_thread, count = OUTPUT_HEADER.unpack(header)
            if magic != OUTPUT_MAGIC:
                raise RuntimeError("bad output magic")
            if version != VERSION or op != OP_TCGEN05_TF32_M64N8K8:
                raise RuntimeError("unsupported output file")
            if threads != THREADS or regs_per_thread != REGS_PER_THREAD:
                raise RuntimeError("unexpected output shape")
            raw = f.read()
        expected = count * threads * regs_per_thread * 4
        if len(raw) != expected:
            raise RuntimeError(f"short output payload: got {len(raw)}, expected {expected}")
        words = struct.unpack(f"<{count * threads * regs_per_thread}I", raw)
        outputs: list[list[list[int]]] = []
        offset = 0
        for _ in range(count):
            threads_out: list[list[int]] = []
            for _thread in range(threads):
                threads_out.append(list(words[offset : offset + regs_per_thread]))
                offset += regs_per_thread
            outputs.append(threads_out)
        return outputs


class Tcgen05Bf16Harness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_bf16_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Bf16ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_BF16_M64N8K16, len(cases)))
            for case in cases:
                f.write(BF16_CASE_STRUCT.pack(*case.a, *case.b, case.c))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        with path.open("rb") as f:
            header = f.read(OUTPUT_HEADER.size)
            if len(header) != OUTPUT_HEADER.size:
                raise RuntimeError("short output header")
            magic, version, op, threads, regs_per_thread, count = OUTPUT_HEADER.unpack(header)
            if magic != OUTPUT_MAGIC:
                raise RuntimeError("bad output magic")
            if version != VERSION or op != OP_TCGEN05_BF16_M64N8K16:
                raise RuntimeError("unsupported output file")
            if threads != THREADS or regs_per_thread != REGS_PER_THREAD:
                raise RuntimeError("unexpected output shape")
            raw = f.read()
        expected = count * threads * regs_per_thread * 4
        if len(raw) != expected:
            raise RuntimeError(f"short output payload: got {len(raw)}, expected {expected}")
        words = struct.unpack(f"<{count * threads * regs_per_thread}I", raw)
        outputs: list[list[list[int]]] = []
        offset = 0
        for _ in range(count):
            threads_out: list[list[int]] = []
            for _thread in range(threads):
                threads_out.append(list(words[offset : offset + regs_per_thread]))
                offset += regs_per_thread
            outputs.append(threads_out)
        return outputs


class Tcgen05F16Harness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_f16_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05F16ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_F16_M64N8K16, len(cases)))
            for case in cases:
                f.write(F16_CASE_STRUCT.pack(*case.a, *case.b, case.c))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_F16_M64N8K16)


class Tcgen05Fp8Harness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_fp8_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Fp8ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_FP8_M64N8K32, len(cases)))
            for case in cases:
                f.write(FP8_CASE_STRUCT.pack(*case.a, *case.b, case.c))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        with path.open("rb") as f:
            header = f.read(OUTPUT_HEADER.size)
            if len(header) != OUTPUT_HEADER.size:
                raise RuntimeError("short output header")
            magic, version, op, threads, regs_per_thread, count = OUTPUT_HEADER.unpack(header)
            if magic != OUTPUT_MAGIC:
                raise RuntimeError("bad output magic")
            if version != VERSION or op != OP_TCGEN05_FP8_M64N8K32:
                raise RuntimeError("unsupported output file")
            if threads != THREADS or regs_per_thread != REGS_PER_THREAD:
                raise RuntimeError("unexpected output shape")
            raw = f.read()
        expected = count * threads * regs_per_thread * 4
        if len(raw) != expected:
            raise RuntimeError(f"short output payload: got {len(raw)}, expected {expected}")
        words = struct.unpack(f"<{count * threads * regs_per_thread}I", raw)
        outputs: list[list[list[int]]] = []
        offset = 0
        for _ in range(count):
            threads_out: list[list[int]] = []
            for _thread in range(threads):
                threads_out.append(list(words[offset : offset + regs_per_thread]))
                offset += regs_per_thread
            outputs.append(threads_out)
        return outputs


class Tcgen05F8f6f4Harness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_f8f6f4_probe",
        *,
        format_name: str = "e5m2",
        device: int = 0,
    ):
        super().__init__(runner, device=device)
        if format_name not in F8F6F4_FORMAT_OPS:
            raise ValueError(f"unsupported f8f6f4 format: {format_name}")
        self.format_name = format_name
        self.op = F8F6F4_FORMAT_OPS[format_name]

    def _write_cases(self, path: Path, cases: list[Tcgen05F8f6f4ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, self.op, len(cases)))
            for case in cases:
                f.write(F8F6F4_CASE_STRUCT.pack(*case.a, *case.b, case.c))

    def _read_outputs(self, path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, self.op, expected_threads=(1, THREADS, CTA_GROUP2_THREADS))


class Tcgen05I8Harness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_i8_probe",
        *,
        a_type: str = "u8",
        b_type: str = "u8",
        saturate: bool = False,
        device: int = 0,
    ):
        super().__init__(runner, device=device)
        key = (a_type, b_type, saturate)
        if key not in I8_FORMAT_OPS:
            raise ValueError(f"unsupported i8 mode: a_type={a_type!r}, b_type={b_type!r}, saturate={saturate!r}")
        self.a_type = a_type
        self.b_type = b_type
        self.saturate = saturate
        self.op = I8_FORMAT_OPS[key]

    def _write_cases(self, path: Path, cases: list[Tcgen05I8ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, self.op, len(cases)))
            for case in cases:
                f.write(I8_CASE_STRUCT.pack(*case.a, *case.b, case.c))

    def _read_outputs(self, path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, self.op, expected_threads=(1, THREADS, CTA_GROUP2_THREADS))


class Tcgen05Nvfp4Harness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_nvfp4_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Nvfp4ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_NVFP4_M128N8K64, len(cases)))
            for case in cases:
                f.write(NVFP4_CASE_STRUCT.pack(*case.a, *case.b, case.c, *case.scale_a, *case.scale_b))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        with path.open("rb") as f:
            header = f.read(OUTPUT_HEADER.size)
            if len(header) != OUTPUT_HEADER.size:
                raise RuntimeError("short output header")
            magic, version, op, threads, regs_per_thread, count = OUTPUT_HEADER.unpack(header)
            if magic != OUTPUT_MAGIC:
                raise RuntimeError("bad output magic")
            if version != VERSION or op != OP_TCGEN05_NVFP4_M128N8K64:
                raise RuntimeError("unsupported output file")
            if threads != THREADS or regs_per_thread != REGS_PER_THREAD:
                raise RuntimeError("unexpected output shape")
            raw = f.read()
        expected = count * threads * regs_per_thread * 4
        if len(raw) != expected:
            raise RuntimeError(f"short output payload: got {len(raw)}, expected {expected}")
        words = struct.unpack(f"<{count * threads * regs_per_thread}I", raw)
        outputs: list[list[list[int]]] = []
        offset = 0
        for _ in range(count):
            threads_out: list[list[int]] = []
            for _thread in range(threads):
                threads_out.append(list(words[offset : offset + regs_per_thread]))
                offset += regs_per_thread
            outputs.append(threads_out)
        return outputs


class Tcgen05BlockScaledHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_block_scaled_probe",
        *,
        op: int,
        device: int = 0,
    ):
        super().__init__(runner, device=device)
        self.op = op

    def _write_cases(self, path: Path, cases: list[Tcgen05BlockScaledProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, self.op, len(cases)))
            for case in cases:
                f.write(
                    BLOCK_SCALED_CASE_STRUCT.pack(
                        *case.a,
                        *case.b,
                        case.c,
                        *case.scale_a,
                        *case.scale_b,
                        MXF8F6F4_FORMAT_CODES[case.a_format],
                        MXF8F6F4_FORMAT_CODES[case.b_format],
                        case.n,
                        case.cta_group,
                        case.m,
                    )
                )

    def _read_outputs(self, path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, self.op, expected_threads=(1, THREADS, CTA_GROUP2_THREADS))


class Tcgen05BlockScaledSparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_block_scaled_sparse_probe",
        *,
        op: int,
        device: int = 0,
    ):
        super().__init__(runner, device=device)
        self.op = op

    def _write_cases(self, path: Path, cases: list[Tcgen05BlockScaledSparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, self.op, len(cases)))
            for case in cases:
                f.write(
                    BLOCK_SCALED_SPARSE_CASE_STRUCT.pack(
                        *case.a,
                        *case.b,
                        case.c,
                        case.metadata,
                        case.metadata_hi,
                        *case.scale_a,
                        *case.scale_b,
                        MXF8F6F4_FORMAT_CODES[case.a_format],
                        MXF8F6F4_FORMAT_CODES[case.b_format],
                        case.n,
                        case.cta_group,
                        case.m,
                    )
                )

    def _read_outputs(self, path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, self.op, expected_threads=(1, THREADS, CTA_GROUP2_THREADS))


def _read_outputs_for_op(
    path: Path,
    expected_op: int,
    expected_threads: int | tuple[int, ...] | None = THREADS,
) -> list[list[list[int]]]:
    with path.open("rb") as f:
        header = f.read(OUTPUT_HEADER.size)
        if len(header) != OUTPUT_HEADER.size:
            raise RuntimeError("short output header")
        magic, version, op, threads, regs_per_thread, count = OUTPUT_HEADER.unpack(header)
        if magic != OUTPUT_MAGIC:
            raise RuntimeError("bad output magic")
        if version != VERSION or op != expected_op:
            raise RuntimeError("unsupported output file")
        valid_threads = (
            None
            if expected_threads is None
            else ((expected_threads,) if isinstance(expected_threads, int) else expected_threads)
        )
        if (valid_threads is not None and threads not in valid_threads) or regs_per_thread != REGS_PER_THREAD:
            raise RuntimeError("unexpected output shape")
        raw = f.read()
    expected = count * threads * regs_per_thread * 4
    if len(raw) != expected:
        raise RuntimeError(f"short output payload: got {len(raw)}, expected {expected}")
    words = struct.unpack(f"<{count * threads * regs_per_thread}I", raw)
    outputs: list[list[list[int]]] = []
    offset = 0
    for _ in range(count):
        threads_out: list[list[int]] = []
        for _thread in range(threads):
            threads_out.append(list(words[offset : offset + regs_per_thread]))
            offset += regs_per_thread
        outputs.append(threads_out)
    return outputs


class Tcgen05Tf32SparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_tf32_sparse_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Tf32SparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_TF32_SPARSE_M64N8K16, len(cases)))
            for case in cases:
                f.write(TF32_SPARSE_CASE_STRUCT.pack(*case.a, *case.b, case.c, case.metadata))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_TF32_SPARSE_M64N8K16)


class Tcgen05Bf16SparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_bf16_sparse_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Bf16SparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_BF16_SPARSE_M64N8K32, len(cases)))
            for case in cases:
                f.write(BF16_SPARSE_CASE_STRUCT.pack(*case.a, *case.b, case.c, case.metadata))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_BF16_SPARSE_M64N8K32)


class Tcgen05F16SparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_f16_sparse_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05F16SparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_F16_SPARSE_M64N8K32, len(cases)))
            for case in cases:
                f.write(F16_SPARSE_CASE_STRUCT.pack(*case.a, *case.b, case.c, case.metadata))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_F16_SPARSE_M64N8K32)


class Tcgen05Fp8SparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_fp8_sparse_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Fp8SparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_FP8_SPARSE_M64N8K64, len(cases)))
            for case in cases:
                f.write(FP8_SPARSE_CASE_STRUCT.pack(*case.a, *case.b, case.c, case.metadata, case.metadata_hi))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_FP8_SPARSE_M64N8K64)


class Tcgen05F8f6f4SparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_f8f6f4_sparse_probe",
        *,
        format_name: str = "e5m2",
        device: int = 0,
    ):
        super().__init__(runner, device=device)
        if format_name not in F8F6F4_SPARSE_FORMAT_OPS:
            raise ValueError(f"unsupported f8f6f4 sparse format: {format_name}")
        self.format_name = format_name
        self.op = F8F6F4_SPARSE_FORMAT_OPS[format_name]

    def _write_cases(self, path: Path, cases: list[Tcgen05F8f6f4SparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, self.op, len(cases)))
            for case in cases:
                f.write(F8F6F4_SPARSE_CASE_STRUCT.pack(*case.a, *case.b, case.c, case.metadata, case.metadata_hi))

    def _read_outputs(self, path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, self.op)


class Tcgen05Nvfp4SparseHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_nvfp4_sparse_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05Nvfp4SparseProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_NVFP4_SPARSE_M128N8K128, len(cases)))
            for case in cases:
                f.write(
                    NVFP4_SPARSE_CASE_STRUCT.pack(
                        *case.a,
                        *case.b,
                        case.c,
                        case.metadata,
                        case.metadata_hi,
                        *case.scale_a,
                        *case.scale_b,
                    )
                )

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_NVFP4_SPARSE_M128N8K128)


class Tcgen05ShapeHarness(Tcgen05Harness):
    def __init__(
        self,
        runner: str | os.PathLike[str] = "build/tcgen05_shape_probe",
        *,
        device: int = 0,
    ):
        super().__init__(runner, device=device)

    @staticmethod
    def _write_cases(path: Path, cases: list[Tcgen05ShapeProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TCGEN05_SHAPE, len(cases)))
            for case in cases:
                k = SHAPE_FORMAT_K[case.format_name]
                if case.sparse:
                    k = SHAPE_FORMAT_SPARSE_K[case.format_name]
                padded_a = tuple(case.a) + (0,) * (64 - k)
                padded_b = tuple(case.b) + (0,) * (64 - k)
                a_format_name = case.a_format_name or ("u8" if case.format_name == "i8" else case.format_name)
                b_format_name = case.b_format_name or ("u8" if case.format_name == "i8" else case.format_name)
                a_format = SHAPE_I8_TYPES[a_format_name] if case.format_name == "i8" else SHAPE_FORMATS[a_format_name]
                b_format = SHAPE_I8_TYPES[b_format_name] if case.format_name == "i8" else SHAPE_FORMATS[b_format_name]
                f.write(
                    SHAPE_CASE_STRUCT.pack(
                        SHAPE_FORMATS[case.format_name],
                        a_format,
                        b_format,
                        SHAPE_D_TYPES[case.d_type],
                        case.m,
                        case.n,
                        (1 if case.ws else 0)
                        | (2 if case.cta_group == 2 else 0)
                        | (4 if case.sparse else 0)
                        | (8 if case.saturate else 0)
                        | SHAPE_COORDINATE_FLAGS[case.coordinate],
                        *padded_a,
                        *padded_b,
                        case.c,
                        case.metadata,
                        case.metadata_hi,
                    )
                )

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        return _read_outputs_for_op(path, OP_TCGEN05_SHAPE, expected_threads=None)


def bit_hex(bits: int) -> str:
    return f"0x{bits:08x}"


def flatten_output(case_output: list[list[int]]) -> list[int]:
    return [word for lane in case_output[:VALID_THREADS] for word in lane]


def flatten_bf16_output(case_output: list[list[int]]) -> list[int]:
    return [case_output[thread][reg] for thread, reg in BF16_VALID_POSITIONS]


def flatten_f16_output(case_output: list[list[int]]) -> list[int]:
    return flatten_bf16_output(case_output)


def flatten_fp8_output(case_output: list[list[int]]) -> list[int]:
    return [case_output[thread][reg] for thread, reg in FP8_VALID_POSITIONS]


def flatten_f8f6f4_output(case_output: list[list[int]]) -> list[int]:
    return flatten_fp8_output(case_output)


def flatten_i8_output(case_output: list[list[int]]) -> list[int]:
    return flatten_fp8_output(case_output)


def flatten_nvfp4_output(case_output: list[list[int]]) -> list[int]:
    return [case_output[thread][reg] for thread, reg in NVFP4_VALID_POSITIONS]


def first_word(case_output: list[list[int]]) -> int:
    return case_output[0][0]


def unique_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_output(case_output))


def unique_bf16_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_bf16_output(case_output))


def unique_f16_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_f16_output(case_output))


def unique_fp8_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_fp8_output(case_output))


def unique_f8f6f4_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_f8f6f4_output(case_output))


def unique_i8_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_i8_output(case_output))


def unique_nvfp4_words(case_output: list[list[int]]) -> set[int]:
    return set(flatten_nvfp4_output(case_output))


def all_words(case_output: list[list[int]]) -> list[int]:
    return [word for lane in case_output for word in lane]


def unique_all_words(case_output: list[list[int]]) -> set[int]:
    return set(all_words(case_output))


def make_case(a: Iterable[int], b: Iterable[int], c: int) -> Tcgen05ProbeCase:
    return Tcgen05ProbeCase(tuple(a), tuple(b), c)


def make_bf16_case(a: Iterable[int], b: Iterable[int], c: int) -> Tcgen05Bf16ProbeCase:
    return Tcgen05Bf16ProbeCase(tuple(a), tuple(b), c)


def make_f16_case(a: Iterable[int], b: Iterable[int], c: int) -> Tcgen05F16ProbeCase:
    return Tcgen05F16ProbeCase(tuple(a), tuple(b), c)


def make_fp8_case(a: Iterable[int], b: Iterable[int], c: int) -> Tcgen05Fp8ProbeCase:
    return Tcgen05Fp8ProbeCase(tuple(a), tuple(b), c)


def make_f8f6f4_case(a: Iterable[int], b: Iterable[int], c: int) -> Tcgen05F8f6f4ProbeCase:
    return Tcgen05F8f6f4ProbeCase(tuple(a), tuple(b), c)


def make_i8_case(a: Iterable[int], b: Iterable[int], c: int) -> Tcgen05I8ProbeCase:
    return Tcgen05I8ProbeCase(tuple(a), tuple(b), c)


def make_nvfp4_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    scale_a: Iterable[int] = NVFP4_UNIT_SCALES,
    scale_b: Iterable[int] = NVFP4_UNIT_SCALES,
) -> Tcgen05Nvfp4ProbeCase:
    return Tcgen05Nvfp4ProbeCase(tuple(a), tuple(b), c, tuple(scale_a), tuple(scale_b))


def make_block_scaled_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    scale_a: Iterable[int] = MX_UE8M0_UNIT_SCALES,
    scale_b: Iterable[int] = MX_UE8M0_UNIT_SCALES,
    a_format: str = "e2m1",
    b_format: str = "e2m1",
    n: int = 8,
    cta_group: int = 1,
    m: int = 128,
) -> Tcgen05BlockScaledProbeCase:
    return Tcgen05BlockScaledProbeCase(
        tuple(a), tuple(b), c, tuple(scale_a), tuple(scale_b), a_format, b_format, n, cta_group, m
    )


def make_tf32_sparse_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    metadata: int = SPARSE_FIXED_METADATA,
) -> Tcgen05Tf32SparseProbeCase:
    return Tcgen05Tf32SparseProbeCase(tuple(a), tuple(b), c, metadata)


def make_bf16_sparse_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    metadata: int = SPARSE_FIXED_METADATA,
) -> Tcgen05Bf16SparseProbeCase:
    return Tcgen05Bf16SparseProbeCase(tuple(a), tuple(b), c, metadata)


def make_f16_sparse_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    metadata: int = SPARSE_FIXED_METADATA,
) -> Tcgen05F16SparseProbeCase:
    return Tcgen05F16SparseProbeCase(tuple(a), tuple(b), c, metadata)


def make_fp8_sparse_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    metadata: int = SPARSE_FIXED_METADATA,
    metadata_hi: int = SPARSE_FIXED_METADATA,
) -> Tcgen05Fp8SparseProbeCase:
    return Tcgen05Fp8SparseProbeCase(tuple(a), tuple(b), c, metadata, metadata_hi)


def make_f8f6f4_sparse_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    metadata: int = SPARSE_FIXED_METADATA,
    metadata_hi: int = SPARSE_FIXED_METADATA,
) -> Tcgen05F8f6f4SparseProbeCase:
    return Tcgen05F8f6f4SparseProbeCase(tuple(a), tuple(b), c, metadata, metadata_hi)


def make_nvfp4_sparse_case(
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    scale_a: Iterable[int] = NVFP4_SPARSE_UNIT_SCALES,
    scale_b: Iterable[int] = NVFP4_SPARSE_UNIT_SCALES,
    metadata: int = SPARSE_FIXED_METADATA,
    metadata_hi: int = SPARSE_FIXED_METADATA,
) -> Tcgen05Nvfp4SparseProbeCase:
    return Tcgen05Nvfp4SparseProbeCase(tuple(a), tuple(b), c, metadata, metadata_hi, tuple(scale_a), tuple(scale_b))


def make_shape_case(
    format_name: str,
    m: int,
    n: int,
    a: Iterable[int],
    b: Iterable[int],
    c: int,
    *,
    ws: bool = False,
    cta_group: int = 1,
    sparse: bool = False,
    a_format_name: str | None = None,
    b_format_name: str | None = None,
    d_type: str = "f32",
    saturate: bool = False,
) -> Tcgen05ShapeProbeCase:
    return Tcgen05ShapeProbeCase(
        format_name,
        m,
        n,
        tuple(a),
        tuple(b),
        c,
        ws,
        cta_group,
        sparse,
        a_format_name,
        b_format_name,
        d_type,
        saturate,
    )


def case_from_summands(summands: list[int]) -> Tcgen05ProbeCase:
    if len(summands) != 9:
        raise ValueError("expected eight products plus c")
    return Tcgen05ProbeCase(tuple(summands[:8]), (F32_POS_ONE,) * 8, summands[8])


def bf16_case_from_summands(summands: list[int]) -> Tcgen05Bf16ProbeCase:
    if len(summands) != 17:
        raise ValueError("expected sixteen products plus c")
    return Tcgen05Bf16ProbeCase(tuple(summands[:16]), (F32_POS_ONE,) * 16, summands[16])


def fp8_case_from_summands(summands: list[int]) -> Tcgen05Fp8ProbeCase:
    if len(summands) != 33:
        raise ValueError("expected thirty-two products plus c")
    return Tcgen05Fp8ProbeCase(tuple(summands[:32]), (0x38,) * 32, summands[32])


def nvfp4_case_from_summands(summands: list[int]) -> Tcgen05Nvfp4ProbeCase:
    if len(summands) != 65:
        raise ValueError("expected sixty-four products plus c")
    return Tcgen05Nvfp4ProbeCase(tuple(summands[:64]), (0x2,) * 64, summands[64])


def _word_report(bits: int) -> dict:
    return {"bits": bit_hex(bits), "class": f32_class(bits)}


def _unique_report(case_output: list[list[int]]) -> dict:
    valid = sorted(unique_words(case_output))
    all_values = sorted(unique_all_words(case_output))
    return {
        "first": _word_report(first_word(case_output)),
        "valid_unique_count": len(valid),
        "valid_unique_words": [bit_hex(x) for x in valid[:16]],
        "all_unique_count": len(all_values),
        "all_unique_words": [bit_hex(x) for x in all_values[:16]],
    }


def _summand_names() -> list[str]:
    return [f"p{i}" for i in range(8)] + ["c"]


def _scaled_pow2_bits(multiplier: float, exp: int) -> int:
    return f32_to_bits(math.ldexp(multiplier, exp))


def probe_output_layout(harness: Tcgen05Harness) -> dict:
    zero = F32_POS_ZERO
    one = F32_POS_ONE
    cases = {
        "c_only_one": case_from_summands([zero, zero, zero, zero, zero, zero, zero, zero, one]),
        "sum_8_products": case_from_summands([one, one, one, one, one, one, one, one, zero]),
        "sum_8_products_plus_c": case_from_summands([one, one, one, one, one, one, one, one, one]),
    }
    outputs = harness.run(cases.values())
    return {
        "threads": THREADS,
        "regs_per_thread": REGS_PER_THREAD,
        "valid_threads": VALID_THREADS,
        "valid_slice_note": "The scalar probes compare the first 32 threads x 4 registers, which is the updated uniform slice in this runner.",
        "cases": {name: _unique_report(out) for name, out in zip(cases, outputs)},
    }


def probe_position_independence(harness: Tcgen05Harness) -> dict:
    zero = F32_POS_ZERO
    one = F32_POS_ONE
    cases = [
        case_from_summands([one if i == j else zero for i in range(8)] + [zero])
        for j in range(8)
    ]
    cases.append(case_from_summands([zero, zero, zero, zero, zero, zero, zero, zero, one]))
    outputs = harness.run(cases)
    failures = []
    for name, out in zip(_summand_names(), outputs):
        values = unique_words(out)
        if len(values) != 1:
            failures.append({"summand": name, "unique": [bit_hex(x) for x in sorted(values)[:8]]})
    return {
        "cases": len(cases),
        "position_independent": not failures,
        "summand_order": _summand_names(),
        "failures": failures,
    }


def run_quick_probes(harness: Tcgen05Harness) -> dict:
    zero = F32_POS_ZERO
    one = F32_POS_ONE

    determinism_case = case_from_summands(
        [f32_pow2_bits(0), f32_pow2_bits(-8), f32_pow2_bits(-16), zero, zero, zero, zero, zero, f32_pow2_bits(-4)]
    )
    det_outputs = harness.run([determinism_case] * 16)
    det_first = flatten_output(det_outputs[0])

    pre_cases = []
    tags = []
    for bit in range(23):
        varied = one ^ (1 << bit)
        pre_cases.append(case_from_summands([varied, zero, zero, zero, zero, zero, zero, zero, zero]))
        tags.append(("a", bit))
        pre_cases.append(Tcgen05ProbeCase((one,) + (zero,) * 7, (varied,) + (one,) * 7, zero))
        tags.append(("b", bit))
    base_a, base_b = [
        first_word(out)
        for out in harness.run(
            [
                case_from_summands([one, zero, zero, zero, zero, zero, zero, zero, zero]),
                Tcgen05ProbeCase((one,) + (zero,) * 7, (one,) * 8, zero),
            ]
        )
    ]
    pre_outputs = harness.run(pre_cases)
    ignored = {"a": [], "b": []}
    changed = {"a": [], "b": []}
    for (kind, bit), out in zip(tags, pre_outputs):
        base = base_a if kind == "a" else base_b
        (changed if first_word(out) != base else ignored)[kind].append(bit)

    precision_cases = [
        case_from_summands([one, f32_pow2_bits(-e), zero, zero, zero, zero, zero, zero, zero])
        for e in range(1, 81)
    ]
    precision_outputs = harness.run(precision_cases)
    changed_eps = [e for e, out in zip(range(1, 81), precision_outputs) if first_word(out) != one]
    vanished_eps = [e for e, out in zip(range(1, 81), precision_outputs) if first_word(out) == one]

    return {
        "runner_info": harness.info(),
        "output_layout": probe_output_layout(harness),
        "determinism": {
            "repeats": 16,
            "deterministic": all(flatten_output(out) == det_first for out in det_outputs[1:]),
            "first": bit_hex(det_first[0]),
            "unique_words": len(set(det_first)),
        },
        "position_independence": probe_position_independence(harness),
        "pretruncation": {
            "base_a": bit_hex(base_a),
            "base_b": bit_hex(base_b),
            "a_ignored_mantissa_bits": ignored["a"],
            "a_changed_mantissa_bits": changed["a"],
            "b_ignored_mantissa_bits": ignored["b"],
            "b_changed_mantissa_bits": changed["b"],
        },
        "precision": {
            "test": "1 + epsilon via p0 + p1",
            "last_surviving_epsilon": f"2^-{max(changed_eps)}" if changed_eps else None,
            "first_vanished_epsilon": f"2^-{min(vanished_eps)}" if vanished_eps else None,
        },
        "rounding": probe_rounding(harness),
        "special_values": probe_special_values(harness),
    }


def probe_swamping_order(harness: Tcgen05Harness, x_exp: int = 30, y_exp: int = 0) -> dict:
    x = f32_pow2_bits(x_exp)
    neg_x = f32_neg(x)
    y = f32_pow2_bits(y_exp)
    cases: list[Tcgen05ProbeCase] = []
    pairs: list[tuple[int, int]] = []
    for i, j in itertools.permutations(range(9), 2):
        summands = [y] * 9
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
        "summand_order": _summand_names(),
        "X": f"2^{x_exp}",
        "y": f"2^{y_exp}",
        "pairs": observations,
    }


def probe_precision_matrix(harness: Tcgen05Harness, max_exp: int = 80) -> dict:
    zero = F32_POS_ZERO
    one = F32_POS_ONE
    names = _summand_names()
    cases: list[Tcgen05ProbeCase] = []
    tags: list[tuple[int, int, int]] = []
    for base_idx in range(9):
        for eps_idx in range(9):
            if base_idx == eps_idx:
                continue
            for eps_exp in range(1, max_exp + 1):
                summands = [zero] * 9
                summands[base_idx] = one
                summands[eps_idx] = f32_pow2_bits(-eps_exp)
                cases.append(case_from_summands(summands))
                tags.append((base_idx, eps_idx, eps_exp))
    outputs = harness.run(cases)

    by_pair: dict[str, dict[str, list[int]]] = {}
    for (base_idx, eps_idx, eps_exp), out in zip(tags, outputs):
        key = f"{names[base_idx]}+eps@{names[eps_idx]}"
        pair = by_pair.setdefault(key, {"changed": [], "vanished": []})
        (pair["changed"] if first_word(out) != one else pair["vanished"]).append(eps_exp)

    thresholds = {}
    for key, pair in by_pair.items():
        changed = pair["changed"]
        vanished = pair["vanished"]
        thresholds[key] = {
            "last_surviving_epsilon": f"2^-{max(changed)}" if changed else None,
            "first_vanished_epsilon": f"2^-{min(vanished)}" if vanished else None,
        }
    return {
        "test": "base summand = 1 plus epsilon at each other summand",
        "max_exponent": max_exp,
        "summand_order": names,
        "thresholds": thresholds,
    }


def probe_cancellation_precision(harness: Tcgen05Harness, max_exp: int = 100) -> dict:
    zero = F32_POS_ZERO
    names = _summand_names()
    cases: list[Tcgen05ProbeCase] = []
    tags: list[tuple[int, int]] = []
    for eps_idx in range(2, 9):
        for eps_exp in range(1, max_exp + 1):
            summands = [zero] * 9
            summands[0] = F32_NEG_ONE
            summands[1] = F32_POS_ONE
            summands[eps_idx] = f32_pow2_bits(-eps_exp)
            cases.append(case_from_summands(summands))
            tags.append((eps_idx, eps_exp))
    outputs = harness.run(cases)

    results: dict[str, dict[str, list[int]]] = {}
    for (eps_idx, eps_exp), out in zip(tags, outputs):
        key = f"-p0+p1+eps@{names[eps_idx]}"
        expected = f32_pow2_bits(-eps_exp)
        bucket = results.setdefault(key, {"exact": [], "changed": []})
        (bucket["exact"] if first_word(out) == expected else bucket["changed"]).append(eps_exp)

    thresholds = {}
    for key, bucket in results.items():
        exact = bucket["exact"]
        changed = bucket["changed"]
        thresholds[key] = {
            "last_exact_epsilon": f"2^-{max(exact)}" if exact else None,
            "first_changed_epsilon": f"2^-{min(changed)}" if changed else None,
        }
    return {
        "test": "-1 + 1 + epsilon exposes product-group cancellation precision",
        "max_exponent": max_exp,
        "thresholds": thresholds,
    }


def _single_offset_case(sign: int, offset_bits: int) -> Tcgen05ProbeCase:
    base = F32_POS_ONE if sign > 0 else F32_NEG_ONE
    term = offset_bits if sign > 0 else f32_neg(offset_bits)
    return case_from_summands([term, F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO, base])


def probe_rounding(harness: Tcgen05Harness, ulp_exp: int = -23) -> dict:
    quarter = _scaled_pow2_bits(1.0, ulp_exp - 2)
    half = _scaled_pow2_bits(1.0, ulp_exp - 1)
    three_quarters = _scaled_pow2_bits(1.5, ulp_exp - 1)
    one_and_half = _scaled_pow2_bits(1.5, ulp_exp)
    cases = {
        "+0.25u": _single_offset_case(+1, quarter),
        "+0.5u": _single_offset_case(+1, half),
        "+0.75u": _single_offset_case(+1, three_quarters),
        "+1.5u": _single_offset_case(+1, one_and_half),
        "-0.25u": _single_offset_case(-1, quarter),
        "-0.5u": _single_offset_case(-1, half),
        "-0.75u": _single_offset_case(-1, three_quarters),
        "-1.5u": _single_offset_case(-1, one_and_half),
    }
    outputs = harness.run(cases.values())
    return {
        "test": "c = +/-1 plus one TF32 product at an exact fractional ULP offset",
        "ulp": f"2^{ulp_exp}",
        "observed": {name: _word_report(first_word(out)) for name, out in zip(cases, outputs)},
    }


def probe_special_values(harness: Tcgen05Harness) -> dict:
    zero = F32_POS_ZERO
    one = F32_POS_ONE
    half_min_normal = 0x0040_0000
    min_normal = 0x0080_0000
    max_finite = 0x7F7F_FFFF
    cases = {
        "subnormal_c_half_min_normal": case_from_summands([zero, zero, zero, zero, zero, zero, zero, zero, half_min_normal]),
        "subnormal_a_half_min_normal": case_from_summands([half_min_normal, zero, zero, zero, zero, zero, zero, zero, zero]),
        "subnormal_b_half_min_normal": make_case((one,) + (zero,) * 7, (half_min_normal,) + (one,) * 7, zero),
        "subnormal_product_half_times_min_normal": make_case(
            (f32_to_bits(0.5),) + (zero,) * 7, (min_normal,) + (one,) * 7, zero
        ),
        "negative_zero_generation": make_case((F32_POS_ZERO,) * 8, (F32_NEG_ZERO,) * 8, F32_NEG_ZERO),
        "positive_inf_product": case_from_summands([F32_POS_INF, zero, zero, zero, zero, zero, zero, zero, zero]),
        "negative_inf_product": case_from_summands([F32_NEG_INF, zero, zero, zero, zero, zero, zero, zero, zero]),
        "zero_times_inf": make_case((F32_POS_ZERO,) + (zero,) * 7, (F32_POS_INF,) + (one,) * 7, zero),
        "pos_inf_plus_neg_inf": case_from_summands([F32_POS_INF, zero, zero, zero, zero, zero, zero, zero, F32_NEG_INF]),
        "quiet_nan_product": case_from_summands([0x7FC1_2345, zero, zero, zero, zero, zero, zero, zero, zero]),
        "signaling_nan_product": case_from_summands([0x7F81_2345, zero, zero, zero, zero, zero, zero, zero, zero]),
        "nan_low_payload_a": case_from_summands([0x7F80_0001, zero, zero, zero, zero, zero, zero, zero, zero]),
        "nan_low_payload_b": make_case((one,) + (zero,) * 7, (0x7F80_0001,) + (one,) * 7, zero),
        "finite_overflow_edge": case_from_summands([max_finite, max_finite, zero, zero, zero, zero, zero, zero, F32_NEG_INF]),
    }
    outputs = harness.run(cases.values())
    return {name: _word_report(first_word(out)) for name, out in zip(cases, outputs)}


def run_deep_probes(harness: Tcgen05Harness, *, max_precision_exp: int = 80) -> dict:
    return {
        "runner_info": harness.info(),
        "output_layout": probe_output_layout(harness),
        "position_independence": probe_position_independence(harness),
        "pretruncation": run_quick_probes(harness)["pretruncation"],
        "swamping_order": probe_swamping_order(harness),
        "precision_matrix": probe_precision_matrix(harness, max_exp=max_precision_exp),
        "cancellation_precision": probe_cancellation_precision(harness, max_exp=max_precision_exp),
        "rounding": probe_rounding(harness),
        "special_values": probe_special_values(harness),
    }

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INPUT_MAGIC = b"MMAPRB1\0"
OUTPUT_MAGIC = b"MMAPRO1\0"
VERSION = 1
OP_TF32_M16N8K4 = 1
LANES = 32
ACC_REGS = 4
CASE_STRUCT = struct.Struct("<9I")
INPUT_HEADER = struct.Struct("<8sIIQ")
OUTPUT_HEADER = struct.Struct("<8sIIIIQ")


@dataclass(frozen=True)
class ProbeCase:
    a: tuple[int, int, int, int]
    b: tuple[int, int, int, int]
    c: int

    def __post_init__(self) -> None:
        if len(self.a) != 4 or len(self.b) != 4:
            raise ValueError("ProbeCase requires four A and four B values")
        for value in (*self.a, *self.b, self.c):
            if value < 0 or value > 0xFFFF_FFFF:
                raise ValueError(f"value out of uint32 range: {value!r}")


class MmaHarness:
    def __init__(self, runner: str | os.PathLike[str] = "build/mma_tf32_probe", *, device: int = 0):
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

    def run(self, cases: Iterable[ProbeCase]) -> list[list[list[int]]]:
        self.build()
        case_list = list(cases)
        with tempfile.TemporaryDirectory(prefix="mma-probe-") as tmp:
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
    def _write_cases(path: Path, cases: list[ProbeCase]) -> None:
        with path.open("wb") as f:
            f.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, OP_TF32_M16N8K4, len(cases)))
            for case in cases:
                f.write(CASE_STRUCT.pack(*case.a, *case.b, case.c))

    @staticmethod
    def _read_outputs(path: Path) -> list[list[list[int]]]:
        with path.open("rb") as f:
            header = f.read(OUTPUT_HEADER.size)
            if len(header) != OUTPUT_HEADER.size:
                raise RuntimeError("short output header")
            magic, version, op, lanes, acc_regs, count = OUTPUT_HEADER.unpack(header)
            if magic != OUTPUT_MAGIC:
                raise RuntimeError("bad output magic")
            if version != VERSION or op != OP_TF32_M16N8K4:
                raise RuntimeError("unsupported output file")
            if lanes != LANES or acc_regs != ACC_REGS:
                raise RuntimeError("unexpected output shape")
            raw = f.read()
        expected = count * lanes * acc_regs * 4
        if len(raw) != expected:
            raise RuntimeError(f"short output payload: got {len(raw)}, expected {expected}")
        words = struct.unpack(f"<{count * lanes * acc_regs}I", raw)
        outputs: list[list[list[int]]] = []
        offset = 0
        for _ in range(count):
            lanes_out: list[list[int]] = []
            for _lane in range(lanes):
                lanes_out.append(list(words[offset : offset + acc_regs]))
                offset += acc_regs
            outputs.append(lanes_out)
        return outputs

#!/usr/bin/env python3
"""Reproduce the full B200 hardware validation matrix for tcgen05-model."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Run:
    args: tuple[str, ...]


@dataclass(frozen=True)
class PathValidation:
    name: str
    runs: tuple[Run, ...]


def _split_runs(command: tuple[str, ...], cases: int, seed: int) -> tuple[Run, Run]:
    raw_cases = cases // 2
    finite_cases = cases - raw_cases
    common = (*command, "--show", "4")
    return (
        Run((*common, "--cases", str(raw_cases), "--seed", str(seed))),
        Run((*common, "--cases", str(finite_cases), "--seed", str(seed + 1), "--finite-only")),
    )


def validation_matrix(cases: int) -> tuple[PathValidation, ...]:
    paths: list[PathValidation] = []

    dense = (
        ("dense-tf32", ("tcgen05-random", "--model", "raw-window-rz")),
        ("dense-bf16", ("tcgen05-bf16-random", "--model", "bf16-raw-window-rz")),
        ("dense-f16", ("tcgen05-f16-random", "--model", "f16-raw-window-rz")),
        ("dense-e4m3", ("tcgen05-fp8-random", "--model", "fp8e4m3-raw-window-rz")),
        ("dense-e5m2", ("tcgen05-f8f6f4-random", "--format", "e5m2")),
        ("dense-e2m3", ("tcgen05-f8f6f4-random", "--format", "e2m3")),
        ("dense-e3m2", ("tcgen05-f8f6f4-random", "--format", "e3m2")),
        ("dense-e2m1", ("tcgen05-f8f6f4-random", "--format", "e2m1")),
    )
    for index, (name, command) in enumerate(dense):
        paths.append(PathValidation(name, _split_runs(command, cases, 92101 + index * 2)))

    i8_pairs = (("u8", "u8"), ("s8", "u8"), ("u8", "s8"), ("s8", "s8"))
    seed = 92201
    for saturate in (False, True):
        for a_type, b_type in i8_pairs:
            args = (
                "tcgen05-i8-random",
                "--a-type",
                a_type,
                "--b-type",
                b_type,
                "--model",
                "s32-exact",
                "--cases",
                str(cases),
                "--batch-size",
                "20000",
                "--seed",
                str(seed),
                "--show",
                "4",
            )
            if saturate:
                args = (*args, "--saturate")
            paths.append(
                PathValidation(
                    f"i8-{a_type}-{b_type}-{'saturate' if saturate else 'wrap'}",
                    (Run(args),),
                )
            )
            seed += 1

    seed = 92301
    for format_name in ("e4m3", "e5m2", "e2m3", "e3m2", "e2m1"):
        command = (
            "tcgen05-mxf8f6f4-random",
            "--format",
            format_name,
            "--model",
            "block-scaled-window-rz",
            "--random-scales",
        )
        paths.append(PathValidation(f"dense-mx-{format_name}", _split_runs(command, cases, seed)))
        seed += 2

    for kind in ("mxf4", "mxf4nvf4-2x", "mxf4nvf4-4x", "mxf4nvf4-4x-ue4m3"):
        command = (
            "tcgen05-block-scaled-random",
            "--kind",
            kind,
            "--model",
            "block-scaled-window-rz",
            "--random-scales",
        )
        paths.append(PathValidation(f"dense-{kind}", _split_runs(command, cases, seed)))
        seed += 2

    sparse_seed = {
        "tf32": 92401,
        "bf16": 92403,
        "f16": 92405,
        "fp8": 92407,
        "e5m2": 92409,
        "e2m3": 92501,
        "e3m2": 92503,
        "e2m1": 92505,
    }
    for format_name, format_seed in sparse_seed.items():
        command = (
            "tcgen05-sparse-random",
            "--format",
            format_name,
            "--random-metadata",
        )
        paths.append(PathValidation(f"sparse-{format_name}", _split_runs(command, cases, format_seed)))

    sparse_nvfp4 = (
        "tcgen05-sparse-random",
        "--format",
        "nvfp4",
        "--random-metadata",
        "--random-scales",
    )
    paths.append(PathValidation("sparse-nvfp4", _split_runs(sparse_nvfp4, cases, 92507)))

    seed = 92601
    for format_name in ("e4m3", "e5m2", "e2m3", "e3m2", "e2m1"):
        command = (
            "tcgen05-block-scaled-sparse-random",
            "--kind",
            "mxf8f6f4",
            "--format",
            format_name,
            "--random-scales",
            "--random-metadata",
        )
        paths.append(PathValidation(f"sparse-mx-{format_name}", _split_runs(command, cases, seed)))
        seed += 2

    for kind in ("mxf4", "mxf4nvf4-2x", "mxf4nvf4-4x", "mxf4nvf4-4x-ue4m3"):
        command = (
            "tcgen05-block-scaled-sparse-random",
            "--kind",
            kind,
            "--random-scales",
            "--random-metadata",
        )
        paths.append(PathValidation(f"sparse-{kind}", _split_runs(command, cases, seed)))
        seed += 2

    mixed = (
        (
            "mixed-e4m3-e2m1-f32",
            (
                "tcgen05-shape-random",
                "--format",
                "e4m3",
                "--a-format",
                "e4m3",
                "--b-format",
                "e2m1",
                "--d-type",
                "f32",
                "--m",
                "64",
                "--n",
                "8",
                "--batch-size",
                "20000",
            ),
            92703,
        ),
        (
            "mixed-e2m1-e4m3-f16",
            (
                "tcgen05-shape-random",
                "--format",
                "e2m1",
                "--a-format",
                "e2m1",
                "--b-format",
                "e4m3",
                "--d-type",
                "f16",
                "--m",
                "64",
                "--n",
                "8",
                "--batch-size",
                "20000",
            ),
            92705,
        ),
    )
    for name, command, mixed_seed in mixed:
        paths.append(PathValidation(name, _split_runs(command, cases, mixed_seed)))

    nvfp4 = (
        "tcgen05-nvfp4-random",
        "--model",
        "nvfp4-block16-window-rz",
        "--random-scales",
    )
    paths.append(PathValidation("dense-nvfp4", _split_runs(nvfp4, cases, 92801)))

    assert len(paths) == 46, len(paths)
    return tuple(paths)


def _json_report(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"validator did not return JSON:\n{stdout[-2000:]}")
    return json.loads(stdout[start : end + 1])


def _report_failed(report: dict) -> bool:
    failure_fields = (
        "mismatches",
        "position_failures",
        "oracle_position_failures",
    )
    return any(report.get(field, 0) != 0 for field in failure_fields)


def _run_path(spec: PathValidation, device: int) -> dict:
    run_reports = []
    failed = False
    for index, run in enumerate(spec.runs, 1):
        command = [sys.executable, "mma_probe.py", "--device", str(device), *run.args]
        print(f"START device={device} path={spec.name} run={index}/{len(spec.runs)}", flush=True)
        completed = subprocess.run(
            command,
            cwd=HERE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            report = _json_report(completed.stdout)
        except Exception as error:
            report = {"parse_error": str(error), "output": completed.stdout[-8000:]}
        run_failed = completed.returncode != 0 or "parse_error" in report or _report_failed(report)
        failed |= run_failed
        run_reports.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "failed": run_failed,
                "report": report,
            }
        )
        print(
            f"{'FAIL' if run_failed else 'PASS'} device={device} path={spec.name} "
            f"run={index}/{len(spec.runs)} cases={report.get('cases', '?')} "
            f"mismatches={report.get('mismatches', '?')} "
            f"position_failures={report.get('position_failures', '?')}",
            flush=True,
        )
    return {"path": spec.name, "device": device, "failed": failed, "runs": run_reports}


def _parse_devices(text: str) -> tuple[int, ...]:
    devices = tuple(int(item) for item in text.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("expected unique comma-separated nonnegative device IDs")
    return devices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-path", type=int, default=1_000_000)
    parser.add_argument("--devices", type=_parse_devices, default=(0,))
    parser.add_argument("--only", action="append", default=[], help="run path names containing this text")
    parser.add_argument("--list", action="store_true", help="list path names without building or running")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cases_per_path < 2:
        parser.error("--cases-per-path must be at least 2")

    matrix = validation_matrix(args.cases_per_path)
    if args.only:
        matrix = tuple(spec for spec in matrix if any(term in spec.name for term in args.only))
    if args.list:
        for spec in matrix:
            print(spec.name)
        return 0
    if not matrix:
        parser.error("--only did not match a validation path")

    if not args.skip_build:
        subprocess.run(["make", "-j", "all"], cwd=HERE, check=True)

    pending: queue.Queue[PathValidation] = queue.Queue()
    for spec in matrix:
        pending.put(spec)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(device: int) -> None:
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            result = _run_path(spec, device)
            with lock:
                results.append(result)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    results.sort(key=lambda result: result["path"])
    failed_paths = [result["path"] for result in results if result["failed"]]
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = HERE / "results" / f"validation-{stamp}.json"
    if not output.is_absolute():
        output = HERE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_per_path": args.cases_per_path,
        "path_count": len(results),
        "total_cases": args.cases_per_path * len(results),
        "devices": args.devices,
        "failed_paths": failed_paths,
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"SUMMARY paths={len(results)} cases={summary['total_cases']} "
        f"failed={len(failed_paths)} output={output}",
        flush=True,
    )
    return 1 if failed_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())

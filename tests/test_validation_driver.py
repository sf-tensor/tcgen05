import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hardware_validation import run_full_validation


ROOT = Path(__file__).resolve().parents[1]
HARDWARE_VALIDATION = ROOT / "hardware_validation"


def test_report_schema_rejects_missing_failure_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        run_full_validation._report_failed({})


def test_worker_exception_fails_the_run(monkeypatch, tmp_path):
    def explode(spec, device):
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(run_full_validation, "_run_path", explode)
    output = tmp_path / "report.json"
    rc = run_full_validation.main(
        ["--skip-build", "--cases-per-path", "2", "--only", "dense-tf32", "--output", str(output)]
    )
    report = json.loads(output.read_text())
    assert rc == 1
    assert report["requested_path_count"] == 1
    assert report["path_count"] == 1
    assert report["total_cases"] == 0
    assert report["failed_paths"] == ["dense-tf32"]
    assert "injected worker failure" in report["results"][0]["worker_error"]


def test_total_cases_is_summed_from_completed_reports(monkeypatch, tmp_path):
    def one_case(spec, device):
        return {
            "path": spec.name,
            "device": device,
            "failed": False,
            "runs": [
                {
                    "report": {"cases": 1, "mismatches": 0, "position_failures": 0},
                    "failed": False,
                    "returncode": 0,
                    "command": [],
                }
            ],
        }

    monkeypatch.setattr(run_full_validation, "_run_path", one_case)
    output = tmp_path / "report.json"
    rc = run_full_validation.main(
        ["--skip-build", "--cases-per-path", "20", "--only", "dense-tf32", "--output", str(output)]
    )
    report = json.loads(output.read_text())
    assert rc == 0
    assert report["total_cases"] == 1


def test_standalone_random_command_fails_on_mismatch(monkeypatch, capsys):
    sys.path.insert(0, str(HARDWARE_VALIDATION))
    try:
        spec = importlib.util.spec_from_file_location("hardware_validation_probe_cli", HARDWARE_VALIDATION / "mma_probe.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HARDWARE_VALIDATION))

    monkeypatch.setattr(
        module,
        "_run_tcgen05_random_validation",
        lambda args: {"cases": 1, "mismatches": 1, "position_failures": 0},
    )
    assert module.cmd_tcgen05_random(object()) == 1
    assert '"mismatches": 1' in capsys.readouterr().out

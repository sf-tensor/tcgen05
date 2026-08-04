import subprocess
import sys
from pathlib import Path


def test_hardware_validation_manifest_has_46_unique_paths():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "hardware_validation" / "run_full_validation.py"), "--list"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    paths = completed.stdout.splitlines()
    assert len(paths) == 46
    assert len(set(paths)) == 46
    assert "dense-tf32" in paths
    assert "sparse-mxf4nvf4-4x-ue4m3" in paths

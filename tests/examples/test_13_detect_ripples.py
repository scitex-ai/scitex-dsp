"""Smoke test for examples/13_detect_ripples.ipynb — runs jupyter nbconvert --execute."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parent.parent.parent
    / "examples" / "13_detect_ripples.ipynb"
)


def test_13_detect_ripples(tmp_path: Path) -> None:
    assert NOTEBOOK.exists(), f"missing example: {NOTEBOOK}"
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter not installed")
    pytest.importorskip("ripple_detection")

    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    r = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--output", target.name, str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, (
        f"13_detect_ripples.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )

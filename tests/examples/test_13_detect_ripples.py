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



pytestmark = pytest.mark.skipif(
    shutil.which("jupyter") is None,
    reason="jupyter not installed",
)
def test_13_detect_ripples_notebook_exists(tmp_path: Path) -> None:
    # Arrange
    # Act
    # Assert
    # Arrange
    # Act
    # Assert
    # Arrange
    # Act
    # Assert
    assert NOTEBOOK.exists(), f"missing example: {NOTEBOOK}"


def test_13_detect_ripples_r_returncode_equals_n_0(tmp_path: Path) -> None:
    # Arrange
    pytest.importorskip("ripple_detection")
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    # Act
    r = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--output", target.name, str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    # Act
    # Assert
    assert r.returncode == 0, (
        f"13_detect_ripples.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )



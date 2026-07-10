"""Smoke test for examples/04_filt.ipynb — runs jupyter nbconvert --execute."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parent.parent.parent
    / "examples" / "04_filt.ipynb"
)



pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nbconvert") is None,
    reason="nbconvert not installed",
)
def test_04_filt_notebook_exists(tmp_path: Path) -> None:
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


def test_04_filt_r_returncode_equals_n_0(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    # Act
    r = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--output", target.name, str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    # Act
    # Assert
    assert r.returncode == 0, (
        f"04_filt.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )



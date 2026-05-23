"""Smoke-test every _demo_*.py and __main__-bearing _test_*.py module.

Each demo is run as `python -m <module>` in an isolated tmp_path so the
caller-anchored I/O routing (scitex.io creates ``_out/`` siblings) lands
in a disposable dir, not the repo. Failure modes caught here include:

- NameError / AttributeError from drift against the library.
- ValueError from a kwarg the function never supported.
- Crash on import (e.g. pandas 2.2 removed Styler.applymap).

See ``scitex_dev/_skills/general/05_development_07_demo-smoke-tests.md``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Two-bucket list. Sibling demos go in the first bucket; in-source
# __main__ blocks in the second. Splitting them keeps the rationale
# greppable when someone adds a new demo and wonders which bucket fits.
SIBLING_DEMOS = [
    "scitex_dsp._synthesis._demo_sig",
]

EMBEDDED_DEMOS: list[str] = [
    # No __main__-bearing _test_*.py modules under src/ yet.
]

DEMOS = SIBLING_DEMOS + EMBEDDED_DEMOS


@pytest.mark.parametrize("module", DEMOS, ids=lambda m: m.rsplit(".", 1)[-1])
def test_demo_module_runs_to_zero_exit_code(module, tmp_path):
    """Execute the demo end-to-end in an isolated working directory and
    assert it exits cleanly. On non-zero exit, the full subprocess
    stdout + stderr are surfaced in the failure message via
    `subprocess.run(check=True)` so the real error is visible without
    a local re-run."""
    # Arrange
    pytest.importorskip("scitex")
    # Act
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=tmp_path,
        check=False,
        timeout=180,
        capture_output=True,
    )
    # Assert
    assert result.returncode == 0, (
        f"{module} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout.decode()}\n"
        f"--- stderr ---\n{result.stderr.decode()}"
    )

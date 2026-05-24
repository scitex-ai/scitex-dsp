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

# Per-demo optional runtime requirements beyond the core ``scitex`` umbrella.
# A demo's ``__main__`` block is a runnable demonstration; some of them pull
# in heavier, optional ecosystem tiers that ``scitex-dsp`` does NOT depend on.
# ``_demo_sig``'s ``__main__`` plots its synthesized traces: it calls
# ``scitex.session.start`` (which, via scitex-session's matplotlib setup,
# imports the standalone ``scitex_plt`` package = figrecipe) and
# ``scitex.plt.subplots``. ``scitex-dsp`` does NOT depend on ``scitex-plt``
# (it is not in base deps, ``[all]``, or ``[dev]``), so a CI install of
# ``scitex-dsp[all,dev]`` has the ``scitex`` umbrella (transitively, via
# scitex-gen / scitex-nn) but NOT ``scitex_plt`` — the demo subprocess then
# dies at ``setup_matplotlib`` with ``No module named 'scitex_plt'``.
#
# The library API the demo exercises (``demo_sig()``) needs none of that —
# real coverage lives in ``tests/scitex_dsp/_synthesis/test__demo_sig.py``
# (81 plt-free unit tests). The plotting is a demo-only nicety, so guard it
# with ``importorskip('scitex_plt')`` — the standard optional-dep-test
# pattern, NOT a blanket skip of real coverage. The skip target is the
# standalone ``scitex_plt`` (the actually-missing module), not the umbrella
# bridge ``scitex.plt`` (which IS importable from PyPI and would never skip).
# Every demo whose deps ARE present still runs end-to-end.
DEMO_OPTIONAL_IMPORTS: dict[str, tuple[str, ...]] = {
    "scitex_dsp._synthesis._demo_sig": ("scitex_plt",),
}


@pytest.mark.parametrize("module", DEMOS, ids=lambda m: m.rsplit(".", 1)[-1])
def test_demo_module_runs_to_zero_exit_code(module, tmp_path):
    """Execute the demo end-to-end in an isolated working directory and
    assert it exits cleanly. On non-zero exit, the full subprocess
    stdout + stderr are surfaced in the failure message via
    `subprocess.run(check=True)` so the real error is visible without
    a local re-run."""
    # Arrange
    pytest.importorskip("scitex")
    for optional_mod in DEMO_OPTIONAL_IMPORTS.get(module, ()):
        pytest.importorskip(optional_mod)
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

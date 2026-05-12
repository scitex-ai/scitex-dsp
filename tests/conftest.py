#!/usr/bin/env python3
"""Pytest conftest for scitex-dsp.

Two responsibilities:

1. Module-import-time coverage wiring (parallel + subprocess support).
   ``os.environ.setdefault`` would be a no-op here because pytest-cov has
   already set ``COVERAGE_FILE`` to a tmp dir by the time conftest is
   loaded.  See
   ``scitex_dev/_skills/general/05_development_06_subprocess-coverage.md``.

2. Alias ``scitex.dsp`` (umbrella) to the standalone ``scitex_dsp`` so
   tests like ``from scitex.dsp import wavelet`` resolve to the code
   under test rather than to whatever the umbrella distribution shipped.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. Subprocess coverage wiring -----------------------------------------

# Pin coverage's data file at the repo root and point process_startup at our
# pyproject so child interpreters configure themselves correctly.
os.environ["COVERAGE_PROCESS_START"] = str(_PROJECT_ROOT / "pyproject.toml")
os.environ["COVERAGE_FILE"] = str(_PROJECT_ROOT / ".coverage")


def _ensure_subprocess_coverage_shim() -> None:
    """Drop an idempotent ``.pth`` file in site-packages that auto-starts
    coverage in every child Python interpreter via
    ``coverage.process_startup()``.
    """
    purelib = Path(sysconfig.get_paths()["purelib"])
    pth = purelib / "_scitex_dsp_subprocess_coverage.pth"
    shim = (
        "import os, coverage\n"
        "if os.environ.get('COVERAGE_PROCESS_START'):\n"
        "    coverage.process_startup()\n"
    )
    try:
        if not pth.exists() or pth.read_text() != shim:
            pth.write_text(shim)
    except OSError:
        # site-packages may be read-only (e.g. system Python); silently
        # skip — local dev venvs are writable and that's where this matters.
        pass


_ensure_subprocess_coverage_shim()


# --- 2. Alias scitex.dsp -> scitex_dsp -------------------------------------


def _alias_scitex_dsp_to_standalone() -> None:
    try:
        import scitex_dsp  # noqa: F401  ensure standalone is importable
    except ImportError:
        return  # standalone not installed; let tests fail naturally

    # Alias scitex.dsp -> scitex_dsp.  We must:
    #   1. Make sure 'scitex' itself is importable (umbrella may or may not
    #      be installed; if not, create a stub so attribute access works).
    #   2. Replace any pre-loaded scitex.dsp with the standalone module.
    try:
        import scitex  # type: ignore
    except ImportError:
        import types

        scitex = types.ModuleType("scitex")
        sys.modules["scitex"] = scitex

    # Force-replace the dsp attribute and sys.modules entry.
    sys.modules["scitex.dsp"] = sys.modules["scitex_dsp"]
    setattr(scitex, "dsp", sys.modules["scitex_dsp"])

    # Also alias every already-imported scitex_dsp.<sub> as scitex.dsp.<sub>
    # so things like `from scitex.dsp._transform import TORCH_AVAILABLE` work.
    for mod_name in list(sys.modules):
        if mod_name.startswith("scitex_dsp."):
            alias = "scitex.dsp." + mod_name[len("scitex_dsp.") :]
            sys.modules[alias] = sys.modules[mod_name]


_alias_scitex_dsp_to_standalone()

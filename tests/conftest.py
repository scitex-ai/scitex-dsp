#!/usr/bin/env python3
"""Pytest conftest for scitex-dsp.

Ensures that ``scitex.dsp`` (the umbrella alias used in tests) resolves to
the standalone ``scitex_dsp`` package being developed in this repo, rather
than to whatever ``scitex/dsp`` happens to ship inside the umbrella
``scitex`` distribution on PyPI.

The standalone package is the source of truth here; the umbrella merely
re-exports it.  Pinning the alias at conftest load-time keeps test imports
like ``from scitex.dsp import wavelet`` pointing at the code under test.
"""

from __future__ import annotations

import sys


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

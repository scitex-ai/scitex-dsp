#!/usr/bin/env python3
"""Thin re-export of :mod:`scitex_dsp._synthesis.add_noise`.

A *physical* module living at the top level so that both
``import scitex_dsp.add_noise`` and (via the umbrella alias)
``import scitex.dsp.add_noise`` resolve through the normal import
machinery — no meta-path-finder gymnastics required.

This module is **not** imported at bare ``import scitex_dsp`` (submodules
are not auto-imported), so the PEP 562 lazy-import perf win is preserved:
``import scitex_dsp`` still does not pull in torch.
"""

from scitex_dsp._synthesis.add_noise import *  # noqa: F401,F403
from scitex_dsp._synthesis.add_noise import brown, gauss, pink, white  # noqa: F401

__all__ = ["brown", "gauss", "pink", "white"]

# EOF

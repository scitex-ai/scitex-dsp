#!/usr/bin/env python3
"""Scitex DSP utils — building blocks + cross-check helpers."""

from . import pac
from ._differential_bandpass_filters import (
    build_bandpass_filters,
    init_bandpass_filters,
)
from ._ensure_3d import ensure_3d
from ._ensure_even_len import ensure_even_len
from ._zero_pad import _zero_pad_1d, zero_pad
from .pac import _calc_pac_with_tensorpac

__all__ = [
    "_calc_pac_with_tensorpac",
    "_zero_pad_1d",
    "build_bandpass_filters",
    "ensure_3d",
    "ensure_even_len",
    "init_bandpass_filters",
    "pac",
    "zero_pad",
]

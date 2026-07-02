#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Device resolution for the spectral primitives.

``resolve_device`` turns the CPU-safe ``device="auto"`` default used by
``pac`` and ``wavelet`` into a concrete torch device string: ``"cuda"`` when a
GPU is available, else ``"cpu"``. Explicit ``"cuda"`` / ``"cpu"`` (or any other
value) pass through unchanged, so existing callers are unaffected.

Factored into one place so both ``_pac`` and ``_wavelet`` share identical
auto-detection semantics.
"""

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dep in practice
    TORCH_AVAILABLE = False
    torch = None


def resolve_device(device):
    """Resolve ``device="auto"`` to a concrete torch device string.

    Parameters
    ----------
    device : str
        ``"auto"`` → ``"cuda"`` if a CUDA device is available else ``"cpu"``.
        Any other value (``"cuda"``, ``"cpu"``, an explicit index, ...) is
        returned unchanged.

    Returns
    -------
    str
        The concrete device string to hand to ``.to(...)``.
    """
    if device == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device

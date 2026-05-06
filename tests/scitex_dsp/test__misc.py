"""Tests for scitex_dsp miscellaneous utilities."""

import scitex_dsp


def test_module_importable():
    """The package imports cleanly."""
    assert hasattr(scitex_dsp, "__version__")


def test_ensure_3d_top_level_exposed():
    """`ensure_3d` is reachable from the top-level namespace."""
    assert callable(scitex_dsp.ensure_3d)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

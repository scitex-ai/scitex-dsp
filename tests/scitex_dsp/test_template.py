#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2026-05-24 (mocks-removed)"
# File: ./tests/scitex_dsp/test_template.py

"""Tests for the `scitex.dsp.template` module — structural only.

The previous version mocked ``scitex.gen.start`` / ``scitex.gen.close`` and
asserted ``True``, which exercised nothing. The template module is a
copy-paste scaffold; the only useful contracts are (a) it imports cleanly
and (b) the on-disk file contains the canonical session-lifecycle imports
new callers should adopt. Both are checked against the real module — no
mocks, no patching.
"""

import os

import pytest


def test_scitex_dsp_template_module_imports_without_error():
    # Arrange
    # Act
    import scitex.dsp.template  # noqa: F401
    # Assert
    assert "scitex.dsp.template" in __import__("sys").modules


def test_scitex_dsp_template_module_has_dunder_file_attribute():
    # Arrange
    import scitex.dsp.template as template
    # Act
    has_file = hasattr(template, "__file__")
    # Assert
    assert has_file is True


def test_scitex_dsp_template_module_has_dunder_name_attribute():
    # Arrange
    import scitex.dsp.template as template
    # Act
    has_name = hasattr(template, "__name__")
    # Assert
    assert has_name is True


def test_scitex_dsp_template_file_contains_sys_import_pattern():
    # Arrange
    import scitex.dsp.template

    template_file = scitex.dsp.template.__file__
    # Act
    with open(template_file, "r") as fh:
        content = fh.read()
    # Assert
    assert "import sys" in content


def test_scitex_dsp_template_file_contains_matplotlib_pyplot_import():
    # Arrange
    import scitex.dsp.template

    template_file = scitex.dsp.template.__file__
    # Act
    with open(template_file, "r") as fh:
        content = fh.read()
    # Assert
    assert "import matplotlib.pyplot" in content


def test_scitex_dsp_template_file_contains_session_start_call():
    # Arrange
    import scitex.dsp.template

    template_file = scitex.dsp.template.__file__
    # Act
    with open(template_file, "r") as fh:
        content = fh.read()
    # Assert
    assert "scitex.session.start" in content


def test_scitex_dsp_template_file_contains_session_close_call():
    # Arrange
    import scitex.dsp.template

    template_file = scitex.dsp.template.__file__
    # Act
    with open(template_file, "r") as fh:
        content = fh.read()
    # Assert
    assert "scitex.session.close" in content


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

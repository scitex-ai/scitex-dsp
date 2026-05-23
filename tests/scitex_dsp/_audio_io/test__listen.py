#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2026-05-24 (mocks-removed)"
# File: ./tests/scitex_dsp/_audio_io/test__listen.py

"""Tests for `list_and_select_device`.

The production function now accepts ``query_devices``, ``input_fn``, and
``print_fn`` as keyword arguments (see ``src/scitex_dsp/_audio_io/_listen.py``).
Tests inject hand-rolled callables — no ``unittest.mock``, no patching of
``sounddevice`` / ``builtins``. The fakes are honest: rename or change the
signature in production and the tests fail loudly.
"""

import os
import sys

import pytest

pytest.importorskip("mne")
# CI runners lack the PortAudio C library; sounddevice imports fine but raises
# OSError at runtime. Skip the whole module rather than test-by-test.
try:
    import sounddevice  # noqa: F401

    sounddevice.query_devices()
except (OSError, ImportError) as _exc:  # pragma: no cover - env-specific
    pytest.skip(
        f"PortAudio/sounddevice unavailable: {_exc}",
        allow_module_level=True,
    )

from scitex.dsp import list_and_select_device


def _make_devices(n: int) -> list[dict]:
    return [{"name": f"Device {i}", "channels": 2} for i in range(n)]


class _PrintRecorder:
    """Captures print-fn calls so tests can assert on emitted lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *args, **kwargs) -> None:
        self.lines.append(" ".join(str(a) for a in args))


def test_list_and_select_device_returns_callable_symbol():
    # Arrange
    # Act
    target = list_and_select_device
    # Assert
    assert callable(target)


def test_list_and_select_device_returns_selected_id_for_valid_input():
    # Arrange
    devices = _make_devices(3)
    # Act
    device_id = list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "1",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 1


def test_list_and_select_device_returns_zero_for_out_of_range_id():
    # Arrange
    devices = _make_devices(2)
    # Act
    device_id = list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "5",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 0


def test_list_and_select_device_returns_zero_for_non_numeric_input():
    # Arrange
    devices = _make_devices(1)
    # Act
    device_id = list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "abc",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 0


def test_list_and_select_device_returns_zero_for_negative_id():
    # Arrange
    devices = _make_devices(2)
    # Act
    device_id = list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "-1",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 0


def test_list_and_select_device_prints_header_line_before_devices():
    # Arrange
    devices = _make_devices(2)
    recorder = _PrintRecorder()
    # Act
    list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "0",
        print_fn=recorder,
    )
    # Assert
    assert any("Available audio devices:" in line for line in recorder.lines)


def test_list_and_select_device_prints_device_list_during_selection():
    # Arrange
    devices = _make_devices(2)
    recorder = _PrintRecorder()
    # Act
    list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "0",
        print_fn=recorder,
    )
    # Assert
    assert any(str(devices) in line for line in recorder.lines)


def test_list_and_select_device_returns_zero_on_portaudio_error():
    # Arrange
    import sounddevice as sd

    def _raises() -> list[dict]:
        raise sd.PortAudioError("No devices found")

    # Act
    device_id = list_and_select_device(
        query_devices=_raises,
        input_fn=lambda _prompt: "0",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 0


def test_list_and_select_device_returns_zero_for_empty_input():
    # Arrange
    devices = _make_devices(1)
    # Act
    device_id = list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 0


def test_list_and_select_device_returns_max_id_at_upper_boundary():
    # Arrange
    devices = _make_devices(3)
    # Act
    device_id = list_and_select_device(
        query_devices=lambda: devices,
        input_fn=lambda _prompt: "2",
        print_fn=lambda *a, **k: None,
    )
    # Assert
    assert device_id == 2


def test_pulse_server_env_key_present_after_module_import():
    # Arrange
    # Act
    value_present = "PULSE_SERVER" in os.environ
    # Assert
    assert value_present is True


def test_pulse_server_env_value_matches_wslg_socket_path():
    # Arrange
    # Act
    value = os.environ["PULSE_SERVER"]
    # Assert
    assert value == "unix:/mnt/wslg/PulseServer"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

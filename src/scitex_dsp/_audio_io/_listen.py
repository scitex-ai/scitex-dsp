#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-08 09:15:59 (ywatanabe)"
# File: ./scitex_repo/src/scitex/dsp/_listen.py

import os

import scitex_logging as slogging

os.environ["PULSE_SERVER"] = "unix:/mnt/wslg/PulseServer"

# # WSL2 Sound Support
# export PULSE_SERVER=unix:/mnt/wslg/PulseServer


def _import_sounddevice():
    """Import the optional ``sounddevice`` engine (``scitex-dsp[all]``)."""
    try:
        import sounddevice
    except ImportError as exc:
        raise ImportError(
            "Audio device selection requires sounddevice. Install it via "
            "`pip install scitex-dsp[all]` (or `pip install sounddevice`)."
        ) from exc
    return sounddevice


def list_and_select_device(
    *,
    query_devices=None,
    input_fn=None,
    print_fn=None,
) -> int:
    """
    List available audio devices and prompt user to select one.

    Parameters
    ----------
    query_devices : callable, optional
        Callable returning the list of audio devices. Defaults to
        ``sounddevice.query_devices`` — pass a hand-rolled fake from
        tests to avoid mocking PortAudio.
    input_fn : callable, optional
        Callable taking a prompt string and returning user input.
        Defaults to the builtin ``input``.
    print_fn : callable, optional
        Callable taking one string, used for the interactive listing.
        Defaults to the prefix-free stdout console of scitex-logging.

    Example
    -------
    >>> device_id = list_and_select_device()
    Available audio devices:
    ...
    Enter the ID of the device you want to use:

    Returns
    -------
    int
        Selected device ID
    """
    sd = _import_sounddevice()
    if query_devices is None:
        query_devices = sd.query_devices
    if input_fn is None:
        input_fn = input
    if print_fn is None:
        print_fn = slogging.getPlainConsole(__name__).emit
    try:
        print_fn("Available audio devices:")
        devices = query_devices()
        print_fn(str(devices))
        device_id = int(input_fn("Enter the ID of the device you want to use: "))
        if device_id not in range(len(devices)):
            raise ValueError(f"Invalid device ID: {device_id}")
        return device_id
    except (ValueError, sd.PortAudioError) as err:
        print_fn(f"Error during device selection: {err}")
        return 0


# EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:24:54 (ywatanabe)"
# File: ./scitex_repo/src/scitex/dsp/_detect_ripples.py


import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scitex_math import to_z

from ._resample import resample
from ._spectral._hilbert import hilbert
from ._synthesis._demo_sig import demo_sig
from .filt import bandpass, gauss


def detect_ripples(
    xx,
    fs,
    low_hz,
    high_hz,
    sd=2.0,
    smoothing_sigma_ms=4,
    min_duration_ms=10,
    return_preprocessed_signal=False,
):
    """
    xx: 2-dimensional (n_chs, seq_len) or 3-dimensional (batch_size, n_chs, seq_len) wide-band signal.
    """

    try:
        xx_r, fs_r = _preprocess(xx, fs, low_hz, high_hz, smoothing_sigma_ms)
        df = _find_events(xx_r, fs_r, sd, min_duration_ms)
        df = _drop_ripples_at_edges(df, low_hz, xx_r, fs_r)
        df = _calc_relative_peak_position(df)
        # df = _calc_incidence(df, xx_r, fs_r)
        df = _sort_columns(df)

        if not return_preprocessed_signal:
            return df

        elif return_preprocessed_signal:
            return df, xx_r, fs_r

    except ValueError as e:
        print("Caught an error:", e)


def _preprocess(xx, fs, low_hz, high_hz, smoothing_sigma_ms=4):
    # Ensures three dimensional
    if xx.ndim == 2:
        xx = xx[np.newaxis]
    assert xx.ndim == 3

    # For readability
    RIPPLE_BANDS = np.vstack([[low_hz, high_hz]])

    # Downsampling: keep Nyquist safely above high_hz so the bandpass
    # upper edge is well within range. Without this, bandpass returns
    # ~0 and no ripples are detected.
    fs_tgt_default = low_hz * 3
    fs_tgt_safe = int(np.ceil(high_hz * 2.5))
    fs_tgt = max(fs_tgt_default, fs_tgt_safe)
    if fs_tgt > fs_tgt_default:
        warnings.warn(
            f"detect_ripples: target fs raised from {fs_tgt_default} to "
            f"{fs_tgt} Hz so Nyquist stays above high_hz={high_hz}.",
            stacklevel=2,
        )

    # Helper: scitex_decorators.signal_fn squeezes 1-size dims off the input
    # of every decorated signal call. Restore 3D before each per-channel op.
    def _ensure_3d(arr):
        arr = np.asarray(arr)
        while arr.ndim < 3:
            arr = arr[np.newaxis]
        return arr

    xx = _ensure_3d(resample(xx, float(fs), float(fs_tgt)))
    fs = fs_tgt

    # Subtracts the cross-channel mean to reduce EMG-driven false
    # positives (CAR-style). Skip for single-channel input — there
    # the mean equals the signal itself, which would zero everything.
    if xx.shape[1] > 1:
        xx -= np.nanmean(xx, axis=1, keepdims=True)
    else:
        warnings.warn(
            "detect_ripples: single-channel input — skipping the "
            "cross-channel mean subtraction (would zero the signal).",
            stacklevel=2,
        )

    # Bandpass Filtering
    xx = _ensure_3d(
        np.asarray(
            bandpass(
                np.array(xx),
                fs_tgt,
                RIPPLE_BANDS,
            )
        )
        .squeeze(-2)
        .astype(np.float64)
    )

    # Calculate RMS
    xx = xx**2
    _, xx_h = hilbert(xx)
    xx = _ensure_3d(np.asarray(xx_h))
    xx = _ensure_3d(
        np.asarray(gauss(xx, smoothing_sigma_ms * 1e-3 * fs_tgt)).squeeze(-2)
    )
    xx = np.sqrt(xx)

    # Scales across channels
    xx = xx.mean(axis=1)
    xx = to_z(xx, dim=-1)

    return xx, fs_tgt


def _find_events(xx_r, fs_r, sd, min_duration_ms):
    def _find_events_1d(xx_ri, fs_r, sd, min_duration_ms):
        # Finds peaks over the designated standard deviation
        peaks, properties = find_peaks(xx_ri, height=sd)

        # Determines the range around each peak (customize as needed)
        peaks_all = []
        peak_ranges = []
        peak_amplitudes_sd = []

        # Boundary threshold: half the detection threshold. Walking
        # out to envelope < 0 (the original behaviour) yields very
        # wide ranges because the z-scored envelope stays slightly
        # positive long after a burst.
        bound_thr = 0.5 * sd

        for peak in peaks:
            left_bound = np.where(xx_ri[:peak] < bound_thr)[0]
            right_bound = np.where(xx_ri[peak:] < bound_thr)[0]

            left_ips = left_bound.max() if left_bound.size > 0 else peak
            right_ips = peak + right_bound.min() if right_bound.size > 0 else peak

            # Avoid duplicates: Check if the current peak range is already listed
            if not any(
                (left_ips == start and right_ips == end) for start, end in peak_ranges
            ):
                peaks_all.append(peak)
                peak_ranges.append((left_ips, right_ips))
                peak_amplitudes_sd.append(xx_ri[peak])

        # Converts to DataFrame
        if peak_ranges:
            starts, ends = zip(*peak_ranges) if peak_ranges else ([], [])
            df = pd.DataFrame(
                {
                    "start_s": np.hstack(starts) / fs_r,
                    "peak_s": np.hstack(peaks_all) / fs_r,
                    "end_s": np.hstack(ends) / fs_r,
                    "peak_amp_sd": np.hstack(peak_amplitudes_sd),
                }
            ).round(3)
        else:
            df = pd.DataFrame(columns=["start_s", "peak_s", "end_s", "peak_amp_sd"])

        # Duration
        df["duration_s"] = df.end_s - df.start_s

        # Filters events with short duration
        df = df[df.duration_s > (min_duration_ms * 1e-3)]

        return df

    if xx_r.ndim == 1:
        xx_r = xx_r[np.newaxis, :]
    assert xx_r.ndim == 2

    dfs = []
    for i_ch in range(len(xx_r)):
        xx_ri = xx_r[i_ch]
        df_i = _find_events_1d(xx_ri, fs_r, sd, min_duration_ms)
        df_i.index = [i_ch for _ in range(len(df_i))]
        dfs.append(df_i)
    dfs = pd.concat(dfs)

    return dfs


def _drop_ripples_at_edges(df, low_hz, xx_r, fs_r):
    edge_s = 1 / low_hz * 3
    indi_drop = (df.start_s < edge_s) + (xx_r.shape[-1] / fs_r - edge_s < df.end_s)
    df = df[~indi_drop]
    return df


def _calc_relative_peak_position(df):
    delta_s = df.peak_s - df.start_s
    rel_peak = delta_s / df.duration_s
    df["rel_peak_pos"] = np.round(rel_peak, 3)
    return df


# def _calc_incidence(df, xx_r, fs_r):
#     n_ripples = len(df)
#     rec_s = xx_r.shape[-1] / fs_r
#     df["incidence_hz"] = n_ripples / rec_s
#     return df


def _sort_columns(df):
    sorted_columns = [
        "start_s",
        "end_s",
        "duration_s",
        "peak_s",
        "rel_peak_pos",
        "peak_amp_sd",
        # "incidence_hz",
    ]
    df = df[sorted_columns]
    return df


def main():
    xx, tt, fs = demo_sig(sig_type="ripple")
    df = detect_ripples(xx, fs, 80, 140)
    print(df)


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import scitex

    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
        sys, plt, verbose=False
    )
    main()
    scitex.session.close(CONFIG, verbose=False, notify=False)

# EOF

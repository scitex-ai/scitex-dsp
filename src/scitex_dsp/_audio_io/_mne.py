#!/usr/bin/env python3
# Time-stamp: "2024-11-04 02:07:36 (ywatanabe)"
# File: ./scitex_repo/src/scitex/dsp/_mne.py

import pandas as pd
from scitex_dev import try_import_optional

from ..params import EEG_MONTAGE_1020

mne = try_import_optional("mne", pkg="scitex-dsp")
MNE_AVAILABLE = mne is not None


def get_eeg_pos(channel_names=EEG_MONTAGE_1020):
    if not MNE_AVAILABLE:
        raise ImportError(
            "MNE-Python is not installed. Please install with: pip install mne"
        )
    # Load the standard 10-20 montage
    standard_montage = mne.channels.make_standard_montage("standard_1020")
    standard_montage.ch_names = [
        ch_name.upper() for ch_name in standard_montage.ch_names
    ]

    # Get the positions of the electrodes in the standard montage
    positions = standard_montage.get_positions()

    df = pd.DataFrame(positions["ch_pos"])[channel_names]

    # 3-row coordinate frame: rename rows to x/y/z. Skip when the
    # column selection produced an empty frame (e.g., empty
    # channel_names) so set_index doesn't blow up on length mismatch.
    if len(df) == 3:
        df = df.set_index(pd.Index(["x", "y", "z"]))

    return df


if __name__ == "__main__":
    print(get_eeg_pos())


# EOF

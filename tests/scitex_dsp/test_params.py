#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:00:00 (ywatanabe)"
# File: ./tests/scitex/dsp/test_params.py

import numpy as np
import pandas as pd
import pytest


def test_params_bands_exists_bands_is_not_none():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert BANDS is not None


def test_params_bands_exists_bands_is_pd_dataframe():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert isinstance(BANDS, pd.DataFrame)




def test_params_bands_structure_bands_shape_equals_n_2_6():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands


def test_params_bands_structure_list_bands_index_expected_index():
    # Arrange
    expected_index = ["low_hz", "high_hz"]
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert list(BANDS.index) == expected_index


def test_params_bands_structure_list_bands_columns_expected_columns():
    # Arrange
    expected_columns = ["delta", "theta", "lalpha", "halpha", "beta", "gamma"]
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert list(BANDS.columns) == expected_columns


@pytest.mark.parametrize(
    "band,expected_low",
    [
        ("delta", 0.5),
        ("theta", 4),
        ("lalpha", 8),
        ("halpha", 10),
        ("beta", 13),
        ("gamma", 32),
    ],
)
def test_params_bands_low_hz_value(band, expected_low):
    """Test BANDS DataFrame low_hz row contains correct values."""
    # Arrange
    from scitex.dsp.params import BANDS
    # Act
    actual = BANDS.loc["low_hz", band]
    # Assert
    assert actual == expected_low


@pytest.mark.parametrize(
    "band,expected_high",
    [
        ("delta", 4),
        ("theta", 8),
        ("lalpha", 10),
        ("halpha", 13),
        ("beta", 32),
        ("gamma", 75),
    ],
)
def test_params_bands_high_hz_value(band, expected_high):
    """Test BANDS DataFrame high_hz row contains correct values."""
    # Arrange
    from scitex.dsp.params import BANDS
    # Act
    actual = BANDS.loc["high_hz", band]
    # Assert
    assert actual == expected_high


def test_params_bands_data_types_all_numeric():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert BANDS.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


def test_params_bands_data_types_not_bands_isnull_any_any():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert not BANDS.isnull().any().any()




@pytest.mark.parametrize(
    "current,next_band",
    [
        ("delta", "theta"),
        ("theta", "lalpha"),
        ("lalpha", "halpha"),
        ("halpha", "beta"),
        ("beta", "gamma"),
    ],
)
def test_params_bands_frequency_ordering(current, next_band):
    """Test that frequency bands are properly ordered (non-overlapping)."""
    # Arrange
    from scitex.dsp.params import BANDS
    # Act
    current_high = BANDS.loc["high_hz", current]
    next_low = BANDS.loc["low_hz", next_band]
    # Assert
    assert current_high == next_low, f"{current} high should equal {next_band} low"


def test_params_eeg_montage_1020_exists_not_none():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert EEG_MONTAGE_1020 is not None


def test_params_eeg_montage_1020_exists_is_list():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert isinstance(EEG_MONTAGE_1020, list)




def test_params_eeg_montage_1020_structure_len_is_19():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert len(EEG_MONTAGE_1020) == 19


def test_params_eeg_montage_1020_structure_all_strings():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert all(isinstance(electrode, str) for electrode in EEG_MONTAGE_1020)


def test_params_eeg_montage_1020_equals_expected_electrodes():
    # Arrange
    expected_electrodes = [
        "FP1",
        "F3",
        "C3",
        "P3",
        "O1",
        "FP2",
        "F4",
        "C4",
        "P4",
        "O2",
        "F7",
        "T7",
        "P7",
        "F8",
        "T8",
        "P8",
        "FZ",
        "CZ",
        "PZ",
    ]
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert EEG_MONTAGE_1020 == expected_electrodes




def test_params_eeg_montage_1020_no_duplicates():
    """Test that EEG_MONTAGE_1020 has no duplicate electrodes."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert len(EEG_MONTAGE_1020) == len(set(EEG_MONTAGE_1020))


def test_params_eeg_montage_bipolar_tranverse_is_not_none():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert EEG_MONTAGE_BIPOLAR_TRANVERSE is not None


def test_params_eeg_montage_bipolar_tranverse_is_list():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert isinstance(EEG_MONTAGE_BIPOLAR_TRANVERSE, list)




def test_params_eeg_montage_bipolar_structure_length():
    """Test EEG_MONTAGE_BIPOLAR_TRANVERSE has 14 channels."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert len(EEG_MONTAGE_BIPOLAR_TRANVERSE) == 14


def test_params_eeg_montage_bipolar_structure_all_have_separator():
    """Test EEG_MONTAGE_BIPOLAR_TRANVERSE channels all contain bipolar separator."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert all("-" in ch for ch in EEG_MONTAGE_BIPOLAR_TRANVERSE)


def test_params_eeg_montage_bipolar_structure_two_electrodes_per_channel():
    """Test EEG_MONTAGE_BIPOLAR_TRANVERSE channels have exactly two electrodes."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert all(
        len(ch.split("-")) == 2 for ch in EEG_MONTAGE_BIPOLAR_TRANVERSE
    )


def test_params_eeg_montage_bipolar_content():
    """Test specific bipolar channel pairs."""
    # Arrange
    expected_channels = [
        "FP1-FP2",
        "F7-F3",
        "F3-FZ",
        "FZ-F4",
        "F4-F8",
        "T7-C3",
        "C3-CZ",
        "CZ-C4",
        "C4-T8",
        "P7-P3",
        "P3-PZ",
        "PZ-P4",
        "P4-P8",
        "O1-O2",
    ]
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert EEG_MONTAGE_BIPOLAR_TRANVERSE == expected_channels


def test_params_eeg_montage_bipolar_no_duplicates():
    """Test that EEG_MONTAGE_BIPOLAR_TRANVERSE has no duplicate channels."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert len(EEG_MONTAGE_BIPOLAR_TRANVERSE) == len(set(EEG_MONTAGE_BIPOLAR_TRANVERSE))


def test_params_all_imports_bands_is_not_none():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert BANDS is not None


def test_params_all_imports_eeg_montage_1020_is_not_none():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert EEG_MONTAGE_1020 is not None


def test_params_all_imports_eeg_montage_bipolar_tranverse_is_not_none():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert EEG_MONTAGE_BIPOLAR_TRANVERSE is not None


def test_params_all_imports_bands_is_pd_dataframe():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Assert
    assert isinstance(BANDS, pd.DataFrame)


def test_params_all_imports_eeg_montage_1020_is_list():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Assert
    assert isinstance(EEG_MONTAGE_1020, list)


def test_params_all_imports_eeg_montage_bipolar_tranverse_is_list():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Assert
    assert isinstance(EEG_MONTAGE_BIPOLAR_TRANVERSE, list)




if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/params.py
# --------------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
#
# BANDS = pd.DataFrame(
#     data=np.array([[0.5, 4], [4, 8], [8, 10], [10, 13], [13, 32], [32, 75]]).T,
#     index=["low_hz", "high_hz"],
#     columns=["delta", "theta", "lalpha", "halpha", "beta", "gamma"],
# )
#
# EEG_MONTAGE_1020 = [
#     "FP1", "F3", "C3", "P3", "O1", "FP2", "F4", "C4", "P4", "O2",
#     "F7", "T7", "P7", "F8", "T8", "P8", "FZ", "CZ", "PZ",
# ]
#
# EEG_MONTAGE_BIPOLAR_TRANVERSE = [
#     "FP1-FP2", "F7-F3", "F3-FZ", "FZ-F4", "F4-F8",
#     "T7-C3", "C3-CZ", "CZ-C4", "C4-T8",
#     "P7-P3", "P3-PZ", "PZ-P4", "P4-P8",
#     "O1-O2",
# ]
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/params.py
# --------------------------------------------------------------------------------

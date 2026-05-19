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
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert BANDS is not None


def test_params_bands_exists_bands_is_pd_dataframe():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert isinstance(BANDS, pd.DataFrame)




def test_params_bands_structure_bands_shape_equals_n_2_6():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands


def test_params_bands_structure_list_bands_index_expected_index_bands_shape_equals_n_2_6():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands


def test_params_bands_structure_list_bands_index_expected_index_list_bands_index_expected_index():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Test shape
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands
    # Test index
    expected_index = ["low_hz", "high_hz"]
    # Act
    # Assert
    assert list(BANDS.index) == expected_index




def test_params_bands_structure_list_bands_columns_expected_columns_bands_shape_equals_n_2_6():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands


def test_params_bands_structure_list_bands_columns_expected_columns_list_bands_index_expected_index():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Test shape
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands
    # Test index
    expected_index = ["low_hz", "high_hz"]
    # Act
    # Assert
    assert list(BANDS.index) == expected_index


def test_params_bands_structure_list_bands_columns_expected_columns_list_bands_columns_expected_columns():
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Test shape
    # Assert
    assert BANDS.shape == (2, 6)  # 2 rows (low_hz, high_hz), 6 bands
    # Test index
    expected_index = ["low_hz", "high_hz"]
    assert list(BANDS.index) == expected_index
    # Test columns (frequency bands)
    expected_columns = ["delta", "theta", "lalpha", "halpha", "beta", "gamma"]
    # Act
    # Assert
    assert list(BANDS.columns) == expected_columns






def test_params_bands_values():
    """Test BANDS DataFrame contains correct frequency values."""
    # Arrange
    # Act
    # Assert
    from scitex.dsp.params import BANDS

    # Test expected frequency ranges
    expected_values = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }

    for band, (low, high) in expected_values.items():
        assert BANDS.loc["low_hz", band] == low
        assert BANDS.loc["high_hz", band] == high


def test_params_bands_data_types_bands_dtypes_apply_lambda_x_np_issubdtype_x_np_number_all():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert BANDS.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


def test_params_bands_data_types_not_bands_isnull_any_any():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS
    # Act
    # Assert
    assert not BANDS.isnull().any().any()




def test_params_bands_frequency_ordering():
    """Test that frequency bands are properly ordered (non-overlapping)."""
    # Arrange
    # Act
    # Assert
    from scitex.dsp.params import BANDS

    # Test that each band's high frequency equals next band's low frequency
    bands = ["delta", "theta", "lalpha", "halpha", "beta", "gamma"]

    for i in range(len(bands) - 1):
        current_high = BANDS.loc["high_hz", bands[i]]
        next_low = BANDS.loc["low_hz", bands[i + 1]]
        assert (
            current_high == next_low
        ), f"{bands[i]} high should equal {bands[i+1]} low"


def test_params_eeg_montage_1020_exists_eeg_montage_1020_is_not_none():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Act
    # Assert
    assert EEG_MONTAGE_1020 is not None


def test_params_eeg_montage_1020_exists_eeg_montage_1020_is_list():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Act
    # Assert
    assert isinstance(EEG_MONTAGE_1020, list)




def test_params_eeg_montage_1020_structure_len_eeg_montage_1020_is_19():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Act
    # Assert
    assert len(EEG_MONTAGE_1020) == 19


def test_params_eeg_montage_1020_structure_all_isinstance_electrode_str_for_electrode_in_eeg_montage_10():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Act
    # Assert
    assert all(isinstance(electrode, str) for electrode in EEG_MONTAGE_1020)


def test_params_eeg_montage_1020_structure_eeg_montage_1020_equals_expected_electrodes_len_eeg_montage_1020_is_19():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Act
    # Assert
    assert len(EEG_MONTAGE_1020) == 19


def test_params_eeg_montage_1020_structure_eeg_montage_1020_equals_expected_electrodes_all_isinstance_electrode_str_for_electrode_in_eeg_montage_10():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Act
    # Assert
    assert all(isinstance(electrode, str) for electrode in EEG_MONTAGE_1020)


def test_params_eeg_montage_1020_structure_eeg_montage_1020_equals_expected_electrodes_eeg_montage_1020_equals_expected_electrodes():
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020
    # Test expected number of electrodes
    # Assert
    assert len(EEG_MONTAGE_1020) == 19
    # Test all elements are strings
    assert all(isinstance(electrode, str) for electrode in EEG_MONTAGE_1020)
    # Test specific expected electrodes
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
    # Assert
    assert EEG_MONTAGE_1020 == expected_electrodes






def test_params_eeg_montage_1020_no_duplicates():
    """Test that EEG_MONTAGE_1020 has no duplicate electrodes."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_1020

    # Assert
    assert len(EEG_MONTAGE_1020) == len(set(EEG_MONTAGE_1020))


def test_params_eeg_montage_bipolar_exists_eeg_montage_bipolar_tranverse_is_not_none():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert EEG_MONTAGE_BIPOLAR_TRANVERSE is not None


def test_params_eeg_montage_bipolar_exists_eeg_montage_bipolar_tranverse_is_list():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert isinstance(EEG_MONTAGE_BIPOLAR_TRANVERSE, list)




def test_params_eeg_montage_bipolar_structure():
    """Test EEG_MONTAGE_BIPOLAR_TRANVERSE structure and content."""
    # Arrange
    # Act
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE

    # Test expected number of bipolar channels
    # Assert
    assert len(EEG_MONTAGE_BIPOLAR_TRANVERSE) == 14

    # Test all elements are strings with proper format
    for channel in EEG_MONTAGE_BIPOLAR_TRANVERSE:
        assert isinstance(channel, str)
        assert "-" in channel  # Bipolar channels should have "-" separator
        parts = channel.split("-")
        assert len(parts) == 2  # Should have exactly 2 electrode names


def test_params_eeg_montage_bipolar_content():
    """Test specific bipolar channel pairs."""
    # Arrange
    from scitex.dsp.params import EEG_MONTAGE_BIPOLAR_TRANVERSE

    # Test some expected bipolar pairs
    # Act
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
    # Arrange
    # Act
    from scitex.dsp.params import BANDS, EEG_MONTAGE_1020, EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert BANDS is not None


def test_params_all_imports_eeg_montage_1020_is_not_none():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS, EEG_MONTAGE_1020, EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert EEG_MONTAGE_1020 is not None


def test_params_all_imports_eeg_montage_bipolar_tranverse_is_not_none():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS, EEG_MONTAGE_1020, EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert EEG_MONTAGE_BIPOLAR_TRANVERSE is not None


def test_params_all_imports_bands_is_pd_dataframe():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS, EEG_MONTAGE_1020, EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert isinstance(BANDS, pd.DataFrame)


def test_params_all_imports_eeg_montage_1020_is_list():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS, EEG_MONTAGE_1020, EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
    # Assert
    assert isinstance(EEG_MONTAGE_1020, list)


def test_params_all_imports_eeg_montage_bipolar_tranverse_is_list():
    # Arrange
    # Act
    # Arrange
    # Act
    from scitex.dsp.params import BANDS, EEG_MONTAGE_1020, EEG_MONTAGE_BIPOLAR_TRANVERSE
    # Act
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
#     "FP1",
#     "F3",
#     "C3",
#     "P3",
#     "O1",
#     "FP2",
#     "F4",
#     "C4",
#     "P4",
#     "O2",
#     "F7",
#     "T7",
#     "P7",
#     "F8",
#     "T8",
#     "P8",
#     "FZ",
#     "CZ",
#     "PZ",
# ]
#
# EEG_MONTAGE_BIPOLAR_TRANVERSE = [
#     # Frontal
#     "FP1-FP2",
#     "F7-F3",
#     "F3-FZ",
#     "FZ-F4",
#     "F4-F8",
#     # Central
#     "T7-C3",
#     "C3-CZ",
#     "CZ-C4",
#     "C4-T8",
#     # Parietal
#     "P7-P3",
#     "P3-PZ",
#     "PZ-P4",
#     "P4-P8",
#     # Occipital
#     "O1-O2",
# ]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/params.py
# --------------------------------------------------------------------------------

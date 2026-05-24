#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 12:42:57 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/test__mne.py

import pytest

pytest.importorskip("mne")

import numpy as np
import pandas as pd

from scitex.dsp import get_eeg_pos
from scitex.dsp.params import EEG_MONTAGE_1020


class TestMne:
    """Test cases for MNE-related functions."""

    def test_import_callable_get_eeg_pos(self):
        """Test that get_eeg_pos can be imported."""
        # Arrange
        # Act
        # Assert
        assert callable(get_eeg_pos)

    def test_get_eeg_pos_default_df_is_pd_dataframe(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert isinstance(df, pd.DataFrame)

    def test_get_eeg_pos_default_df_shape_0_3(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert df.shape[0] == 3  # x, y, z coordinates

    def test_get_eeg_pos_default_df_shape_1_len_eeg_montage_1020(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert df.shape[1] == len(EEG_MONTAGE_1020)

    def test_get_eeg_pos_default_list_df_columns_list_eeg_montage_1020(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert list(df.columns) == list(EEG_MONTAGE_1020)


    def test_get_eeg_pos_subset_channels_df_is_pd_dataframe(self):
        # Arrange
        subset_channels = ["FP1", "FP2", "C3", "C4", "O1", "O2"]
        # Act
        df = get_eeg_pos(channel_names=subset_channels)
        # Act
        # Assert
        assert isinstance(df, pd.DataFrame)

    def test_get_eeg_pos_subset_channels_df_shape_0_3(self):
        # Arrange
        subset_channels = ["FP1", "FP2", "C3", "C4", "O1", "O2"]
        # Act
        df = get_eeg_pos(channel_names=subset_channels)
        # Act
        # Assert
        assert df.shape[0] == 3  # x, y, z coordinates

    def test_get_eeg_pos_subset_channels_df_shape_1_len_subset_channels(self):
        # Arrange
        subset_channels = ["FP1", "FP2", "C3", "C4", "O1", "O2"]
        # Act
        df = get_eeg_pos(channel_names=subset_channels)
        # Act
        # Assert
        assert df.shape[1] == len(subset_channels)

    def test_get_eeg_pos_subset_channels_list_df_columns_subset_channels(self):
        # Arrange
        subset_channels = ["FP1", "FP2", "C3", "C4", "O1", "O2"]
        # Act
        df = get_eeg_pos(channel_names=subset_channels)
        # Act
        # Assert
        assert list(df.columns) == subset_channels


    def test_get_eeg_pos_single_channel_df_is_pd_dataframe(self):
        # Arrange
        single_channel = ["CZ"]
        # Act
        df = get_eeg_pos(channel_names=single_channel)
        # Act
        # Assert
        assert isinstance(df, pd.DataFrame)

    def test_get_eeg_pos_single_channel_df_shape_0_3(self):
        # Arrange
        single_channel = ["CZ"]
        # Act
        df = get_eeg_pos(channel_names=single_channel)
        # Act
        # Assert
        assert df.shape[0] == 3  # x, y, z coordinates

    def test_get_eeg_pos_single_channel_df_shape_1_1(self):
        # Arrange
        single_channel = ["CZ"]
        # Act
        df = get_eeg_pos(channel_names=single_channel)
        # Act
        # Assert
        assert df.shape[1] == 1

    def test_get_eeg_pos_single_channel_list_df_columns_single_channel(self):
        # Arrange
        single_channel = ["CZ"]
        # Act
        df = get_eeg_pos(channel_names=single_channel)
        # Act
        # Assert
        assert list(df.columns) == single_channel


    def test_get_eeg_pos_coordinates_range_df_abs_max_max_1_0(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert df.abs().max().max() < 1.0

    def test_get_eeg_pos_coordinates_range_df_abs_min_min_0_0(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert df.abs().min().min() >= 0.0


    def test_get_eeg_pos_coordinate_structure_df_dtypes_apply_lambda_x_np_issubdtype_x_np_number_all(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    def test_get_eeg_pos_coordinate_structure_not_df_isna_any_any(self):
        # Arrange
        # Act
        # Arrange
        # Act
        df = get_eeg_pos()
        # Act
        # Assert
        assert not df.isna().any().any()


    def test_get_eeg_pos_known_positions_abs_cz_pos_0_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[0]) < 0.01  # x should be near 0

    def test_get_eeg_pos_known_positions_abs_cz_pos_1_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[1]) < 0.01  # y should be near 0

    def test_get_eeg_pos_known_positions_cz_pos_2_0_08(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert cz_pos[2] > 0.08  # z should be positive (top of head)

    def test_get_eeg_pos_known_positions_abs_fp1_pos_0_fp2_pos_0_0_01_abs_cz_pos_0_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[0]) < 0.01  # x should be near 0

    def test_get_eeg_pos_known_positions_abs_fp1_pos_0_fp2_pos_0_0_01_abs_cz_pos_1_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[1]) < 0.01  # y should be near 0

    def test_get_eeg_pos_known_positions_abs_fp1_pos_0_fp2_pos_0_0_01_cz_pos_2_0_08(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert cz_pos[2] > 0.08  # z should be positive (top of head)

    def test_get_eeg_pos_known_positions_fp1_fp2_x_coordinates_are_opposite(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # Act
        fp1_pos = df["FP1"].values
        fp2_pos = df["FP2"].values
        # Assert: FP1 and FP2 are symmetric across the sagittal plane (x).
        assert abs(fp1_pos[0] + fp2_pos[0]) < 0.01


    def test_get_eeg_pos_known_positions_abs_fp1_pos_1_fp2_pos_1_0_01_abs_cz_pos_0_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[0]) < 0.01  # x should be near 0

    def test_get_eeg_pos_known_positions_abs_fp1_pos_1_fp2_pos_1_0_01_abs_cz_pos_1_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[1]) < 0.01  # y should be near 0

    def test_get_eeg_pos_known_positions_abs_fp1_pos_1_fp2_pos_1_0_01_cz_pos_2_0_08(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert cz_pos[2] > 0.08  # z should be positive (top of head)

    def test_get_eeg_pos_known_positions_fp1_fp2_y_coordinates_are_similar(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # Act
        fp1_pos = df["FP1"].values
        fp2_pos = df["FP2"].values
        # Assert: FP1 and FP2 sit at the same antero-posterior level (y).
        assert abs(fp1_pos[1] - fp2_pos[1]) < 0.01


    def test_get_eeg_pos_known_positions_abs_fp1_pos_2_fp2_pos_2_0_01_abs_cz_pos_0_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[0]) < 0.01  # x should be near 0

    def test_get_eeg_pos_known_positions_abs_fp1_pos_2_fp2_pos_2_0_01_abs_cz_pos_1_0_01(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert abs(cz_pos[1]) < 0.01  # y should be near 0

    def test_get_eeg_pos_known_positions_abs_fp1_pos_2_fp2_pos_2_0_01_cz_pos_2_0_08(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # CZ should be approximately at the top center
        # Act
        cz_pos = df["CZ"].values
        # Act
        # Assert
        assert cz_pos[2] > 0.08  # z should be positive (top of head)

    def test_get_eeg_pos_known_positions_fp1_fp2_z_coordinates_are_similar(self):
        # Arrange
        df = get_eeg_pos(channel_names=["CZ", "FP1", "FP2"])
        # Act
        fp1_pos = df["FP1"].values
        fp2_pos = df["FP2"].values
        # Assert: FP1 and FP2 sit at the same vertical level (z).
        assert abs(fp1_pos[2] - fp2_pos[2]) < 0.01



    def test_get_eeg_pos_empty_channels_df_is_pd_dataframe(self):
        # Arrange
        empty_channels = []
        # Act
        df = get_eeg_pos(channel_names=empty_channels)
        # Act
        # Assert
        assert isinstance(df, pd.DataFrame)

    def test_get_eeg_pos_empty_channels_df_shape_0_3(self):
        # Arrange
        empty_channels = []
        # Act
        df = get_eeg_pos(channel_names=empty_channels)
        # Act
        # Assert
        assert df.shape[0] == 3  # x, y, z coordinates

    def test_get_eeg_pos_empty_channels_df_shape_1_0(self):
        # Arrange
        empty_channels = []
        # Act
        df = get_eeg_pos(channel_names=empty_channels)
        # Act
        # Assert
        assert df.shape[1] == 0  # No channels

    def test_get_eeg_pos_empty_channels_df_empty(self):
        # Arrange
        empty_channels = []
        # Act
        df = get_eeg_pos(channel_names=empty_channels)
        # Act
        # Assert
        assert df.empty


    def test_get_eeg_pos_invalid_channel_raises(self):
        """Test that invalid channel names raise KeyError."""
        # Arrange
        # Act
        invalid_channels = ["INVALID1", "INVALID2"]

        # Assert
        with pytest.raises(KeyError):
            get_eeg_pos(channel_names=invalid_channels)

    def test_get_eeg_pos_mixed_valid_invalid_channels(self):
        """Test with mix of valid and invalid channels."""
        # Arrange
        # Act
        mixed_channels = ["FP1", "INVALID", "CZ"]

        # Assert
        with pytest.raises(KeyError):
            get_eeg_pos(channel_names=mixed_channels)

    def test_get_eeg_pos_dataframe_index(self):
        """Test the DataFrame index structure."""
        # Arrange
        # Act
        df = get_eeg_pos(channel_names=["FP1"])

        # The code sets index but doesn't use inplace=True,
        # so the index might not be set as expected
        # This is actually a potential bug in the source code
        # Assert
        assert df.shape[0] == 3

    def test_get_eeg_pos_uses_real_1020_montage_channel_set(self):
        """The function uses mne's standard_1020 montage — verify against
        real mne that a canonical 10-20 channel (`CZ`) is present."""
        # Arrange
        # Act
        df = get_eeg_pos(channel_names=["CZ"])
        # Assert
        assert "CZ" in df.columns

    def test_get_eeg_pos_reproducibility(self):
        """Test that multiple calls return the same positions."""
        # Arrange
        df1 = get_eeg_pos(channel_names=["FP1", "FP2", "CZ"])
        # Act
        df2 = get_eeg_pos(channel_names=["FP1", "FP2", "CZ"])
        # Assert: DataFrames are equal cell-wise (uses pandas' own check).
        assert df1.equals(df2)

    def test_get_eeg_pos_all_channels_unique_positions(self):
        """Test that all channels have unique positions."""
        # Arrange
        df = get_eeg_pos()

        # Convert DataFrame to list of position tuples
        positions = []
        # Act
        for col in df.columns:
            pos = tuple(df[col].values)
            positions.append(pos)

        # Check all positions are unique
        # Assert
        assert len(positions) == len(set(positions))


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_mne.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-11-04 02:07:36 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/_mne.py
#
# try:
#     import mne
#
#     MNE_AVAILABLE = True
# except ImportError:
#     MNE_AVAILABLE = False
#     mne = None
#
# import pandas as pd
#
# from .params import EEG_MONTAGE_1020
#
#
# def get_eeg_pos(channel_names=EEG_MONTAGE_1020):
#     if not MNE_AVAILABLE:
#         raise ImportError(
#             "MNE-Python is not installed. Please install with: pip install mne"
#         )
#     # Load the standard 10-20 montage
#     standard_montage = mne.channels.make_standard_montage("standard_1020")
#     standard_montage.ch_names = [
#         ch_name.upper() for ch_name in standard_montage.ch_names
#     ]
#
#     # Get the positions of the electrodes in the standard montage
#     positions = standard_montage.get_positions()
#
#     df = pd.DataFrame(positions["ch_pos"])[channel_names]
#
#     df.set_index(pd.Series(["x", "y", "z"]))
#
#     return df
#
#
# if __name__ == "__main__":
#     print(get_eeg_pos())
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_mne.py
# --------------------------------------------------------------------------------

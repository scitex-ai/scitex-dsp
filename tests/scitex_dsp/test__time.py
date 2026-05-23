#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 13:52:48 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/test__time.py

import pytest

pytest.importorskip("mne")
import numpy as np
from scitex.dsp import time


class TestTime:
    """Test cases for time array generation."""

    def test_import_callable_time(self):
        """Test that time can be imported."""
        # Arrange
        # Act
        # Assert
        assert callable(time)

    def test_time_basic_t_is_np_ndarray(self):
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert isinstance(t, np.ndarray)

    def test_time_basic_len_t_end_sec_start_sec_fs(self):
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == (end_sec - start_sec) * fs

    def test_time_basic_t_0_start_sec(self):
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert t[0] == start_sec

    def test_time_basic_abs_t_1_end_sec_1_0_fs_1e_09(self):
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert abs(t[-1] - (end_sec - 1.0 / fs)) < 1e-9


    def test_time_non_zero_start_len_t_end_sec_start_sec_fs(self):
        # Arrange
        start_sec = 5
        end_sec = 8
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == (end_sec - start_sec) * fs

    def test_time_non_zero_start_t_0_start_sec(self):
        # Arrange
        start_sec = 5
        end_sec = 8
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert t[0] == start_sec

    def test_time_non_zero_start_abs_t_1_end_sec_1_0_fs_1e_09(self):
        # Arrange
        start_sec = 5
        end_sec = 8
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert abs(t[-1] - (end_sec - 1.0 / fs)) < 1e-9


    def test_time_high_sampling_rate_len_t_int_end_sec_start_sec_fs(self):
        # Arrange
        start_sec = 0
        end_sec = 0.1  # 100ms
        fs = 10000  # 10 kHz
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == int((end_sec - start_sec) * fs)

    def test_time_high_sampling_rate_abs_t_0_start_sec_1e_09(self):
        # Arrange
        start_sec = 0
        end_sec = 0.1  # 100ms
        fs = 10000  # 10 kHz
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert abs(t[0] - start_sec) < 1e-9

    def test_time_high_sampling_rate_abs_t_1_end_sec_1_0_fs_1e_09(self):
        # Arrange
        start_sec = 0
        end_sec = 0.1  # 100ms
        fs = 10000  # 10 kHz
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert abs(t[-1] - (end_sec - 1.0 / fs)) < 1e-9


    def test_time_fractional_duration(self):
        """Test time array with fractional duration."""
        # Arrange
        start_sec = 0
        end_sec = 2.5
        fs = 100

        t = time(start_sec, end_sec, fs)

        # Act
        expected_len = int((end_sec - start_sec) * fs)
        # Assert
        assert len(t) == expected_len

    def test_time_uniform_spacing(self):
        """Test that time points are uniformly spaced."""
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 100

        t = time(start_sec, end_sec, fs)

        # Check uniform spacing
        dt = np.diff(t)
        # Act
        expected_dt = 1.0 / fs
        # Assert
        assert np.allclose(dt, expected_dt, rtol=1e-9)

    def test_time_negative_start_len_t_end_sec_start_sec_fs(self):
        # Arrange
        start_sec = -2
        end_sec = 3
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == (end_sec - start_sec) * fs

    def test_time_negative_start_t_0_start_sec(self):
        # Arrange
        start_sec = -2
        end_sec = 3
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert t[0] == start_sec

    def test_time_negative_start_abs_t_1_end_sec_1_0_fs_1e_09(self):
        # Arrange
        start_sec = -2
        end_sec = 3
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert abs(t[-1] - (end_sec - 1.0 / fs)) < 1e-9


    def test_time_low_sampling_rate_len_t_end_sec_start_sec_fs(self):
        # Arrange
        start_sec = 0
        end_sec = 10
        fs = 1  # 1 Hz
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == (end_sec - start_sec) * fs

    def test_time_low_sampling_rate_len_t_is_10(self):
        # Arrange
        start_sec = 0
        end_sec = 10
        fs = 1  # 1 Hz
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == 10


    def test_time_single_sample_len_t_is_1(self):
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 1
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == 1

    def test_time_single_sample_t_0_start_sec_and_t_0_end_sec(self):
        # Arrange
        start_sec = 0
        end_sec = 1
        fs = 1
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert t[0] >= start_sec and t[0] <= end_sec


    def test_time_precision_abs_t_1_t_0_0_001_1e_12(self):
        """Test precision of time array values."""
        # Arrange
        start_sec = 0.0
        end_sec = 1.0
        fs = 1000

        # Act
        t = time(start_sec, end_sec, fs)

        # Check that values are precise
        # Second sample should be exactly 1ms after first
        # Assert
        assert abs(t[1] - t[0] - 0.001) < 1e-12

    def test_time_zero_duration_returns_at_most_one_sample(self):
        # Arrange
        start_sec = 5
        end_sec = 5
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        # Assert
        assert len(t) <= 1

    def test_time_one_sample_zero_duration_value_is_start_or_empty(self):
        """When start == end, the array is either empty (allowed) or
        single-sample equal to start. Asserts the upper-bound shape
        property; companion test pins length <= 1."""
        # Arrange
        start_sec = 5
        end_sec = 5
        fs = 100
        # Act
        t = time(start_sec, end_sec, fs)
        is_valid = len(t) == 0 or t[0] == start_sec
        # Assert
        assert is_valid is True

    def test_time_very_long_duration_len_t_is_3600(self):
        # Arrange
        start_sec = 0
        end_sec = 3600  # 1 hour
        fs = 1  # 1 Hz to keep array manageable
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert len(t) == 3600

    def test_time_very_long_duration_t_0_start_sec(self):
        # Arrange
        start_sec = 0
        end_sec = 3600  # 1 hour
        fs = 1  # 1 Hz to keep array manageable
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert t[0] == start_sec

    def test_time_very_long_duration_abs_t_1_end_sec_1_0_fs_1e_09(self):
        # Arrange
        start_sec = 0
        end_sec = 3600  # 1 hour
        fs = 1  # 1 Hz to keep array manageable
        # Act
        t = time(start_sec, end_sec, fs)
        # Act
        # Assert
        assert abs(t[-1] - (end_sec - 1.0 / fs)) < 1e-9


    def test_time_floating_point_consistency_len_t_expected_samples(self):
        # Arrange
        start_sec = 0.0
        end_sec = 0.1
        fs = 44100  # Audio sampling rate
        t = time(start_sec, end_sec, fs)
        # Should have exactly the expected number of samples
        # Act
        expected_samples = int((end_sec - start_sec) * fs)
        # Act
        # Assert
        assert len(t) == expected_samples

    def test_time_floating_point_consistency_abs_t_0_start_sec_1e_09(self):
        # Arrange
        start_sec = 0.0
        end_sec = 0.1
        fs = 44100  # Audio sampling rate
        t = time(start_sec, end_sec, fs)
        # Should have exactly the expected number of samples
        # Act
        expected_samples = int((end_sec - start_sec) * fs)
        # Act
        # Assert
        assert abs(t[0] - start_sec) < 1e-9

    def test_time_floating_point_consistency_abs_t_1_end_sec_1_0_fs_1e_09(self):
        # Arrange
        start_sec = 0.0
        end_sec = 0.1
        fs = 44100  # Audio sampling rate
        t = time(start_sec, end_sec, fs)
        # Should have exactly the expected number of samples
        # Act
        expected_samples = int((end_sec - start_sec) * fs)
        # Act
        # Assert
        assert abs(t[-1] - (end_sec - 1.0 / fs)) < 1e-9



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_time.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-30 12:11:01 (ywatanabe)"
# # /mnt/ssd/ripple-wm-code/scripts/externals/scitex/src/scitex/dsp/_time.py
#
#
# import numpy as np
# import scitex
#
#
# def time(start_sec, end_sec, fs):
#     # return np.linspace(start_sec, end_sec, (end_sec - start_sec) * fs)
#     return scitex.gen.float_linspace(start_sec, end_sec, (end_sec - start_sec) * fs)
#
#
# def main():
#     out = time(10, 15, 256)
#     print(out)
#
#
# if __name__ == "__main__":
#     import sys
#
#     import matplotlib.pyplot as plt
#
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
#         sys, plt, verbose=False
#     )
#     main()
#     scitex.session.close(CONFIG, verbose=False, notify=False)
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_time.py
# --------------------------------------------------------------------------------

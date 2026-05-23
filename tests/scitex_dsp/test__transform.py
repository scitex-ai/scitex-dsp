#!/usr/bin/env python3
# Time-stamp: "2024-11-07 14:08:32 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/test__transform.py

import pytest

torch = pytest.importorskip("torch")
import numpy as np
import pandas as pd
from scitex.dsp import to_segments, to_sktime_df


class TestTransformAvailableFlags:
    """Test _AVAILABLE flags for optional dependencies."""

    def test_torch_available_flag_exists(self):
        """Test that TORCH_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex.dsp._transform import TORCH_AVAILABLE

        # Assert
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_check_torch_function_exists(self):
        """Test that _check_torch function is exported."""
        # Arrange
        # Act
        from scitex.dsp._transform import _check_torch

        # Assert
        assert callable(_check_torch)

    def test_torch_available_is_true_when_torch_installed(self):
        """Test that TORCH_AVAILABLE is True when torch is installed."""
        # Arrange
        # Act
        from scitex.dsp._transform import TORCH_AVAILABLE

        # Since we're running this test, torch must be available
        # Assert
        assert TORCH_AVAILABLE is True

    def test_check_torch_does_not_raise_when_available(self):
        """Test that _check_torch doesn't raise when torch is available."""
        # Arrange
        from scitex.dsp._transform import _check_torch

        # Act
        result = _check_torch()

        # Assert
        assert result is None


class TestTransform:
    """Test cases for signal transformation functions."""

    def test_import_callable_to_sktime_df(self):
        # Arrange
        # Act
        # Assert
        assert callable(to_sktime_df)

    def test_import_callable_to_segments(self):
        # Arrange
        # Act
        # Assert
        assert callable(to_segments)


    def test_to_sktime_df_basic_df_is_pd_dataframe(self):
        # Arrange
        n_samples, seq_len, n_channels = 5, 100, 3
        arr = np.random.randn(n_samples, seq_len, n_channels)
        # Act
        df = to_sktime_df(arr)
        # Act
        # Assert
        assert isinstance(df, pd.DataFrame)

    def test_to_sktime_df_basic_len_df_n_samples(self):
        # Arrange
        n_samples, seq_len, n_channels = 5, 100, 3
        arr = np.random.randn(n_samples, seq_len, n_channels)
        # Act
        df = to_sktime_df(arr)
        # Act
        # Assert
        assert len(df) == n_samples

    def test_to_sktime_df_basic_list_df_columns_dim_0(self):
        # Arrange
        n_samples, seq_len, n_channels = 5, 100, 3
        arr = np.random.randn(n_samples, seq_len, n_channels)
        # Act
        df = to_sktime_df(arr)
        # Act
        # Assert
        assert list(df.columns) == ["dim_0"]


    def test_to_sktime_df_single_channel_len_equals_n_samples(self):
        """Single channel: dataframe row count equals n_samples."""
        # Arrange
        n_samples, seq_len, n_channels = 3, 50, 1
        arr = np.random.randn(n_samples, seq_len, n_channels)

        # Act
        df = to_sktime_df(arr)

        # Assert
        assert len(df) == n_samples

    def test_to_sktime_df_single_channel_channel_0_present(self):
        """Single channel: every row exposes a 'channel_0' index."""
        # Arrange
        n_samples, seq_len, n_channels = 3, 50, 1
        arr = np.random.randn(n_samples, seq_len, n_channels)

        # Act
        df = to_sktime_df(arr)

        # Assert
        assert all("channel_0" in df.iloc[i, 0].index for i in range(n_samples))

    def test_to_sktime_df_single_channel_channel_length_equals_seq_len(self):
        """Single channel: each row's channel_0 series length equals seq_len."""
        # Arrange
        n_samples, seq_len, n_channels = 3, 50, 1
        arr = np.random.randn(n_samples, seq_len, n_channels)

        # Act
        df = to_sktime_df(arr)

        # Assert
        assert all(len(df.iloc[i, 0]["channel_0"]) == seq_len for i in range(n_samples))

    def test_to_sktime_df_2d_array_raises_value_error(self):
        # Arrange
        arr_2d = np.random.randn(10, 20)
        # Act
        ctx = pytest.raises(ValueError, match="Input data must be a 3D array")
        # Assert
        with ctx:
            to_sktime_df(arr_2d)

    def test_to_sktime_df_4d_array_raises_value_error(self):
        # Arrange
        arr_4d = np.random.randn(5, 10, 20, 3)
        # Act
        ctx = pytest.raises(ValueError, match="Input data must be a 3D array")
        # Assert
        with ctx:
            to_sktime_df(arr_4d)



    def test_to_sktime_df_data_preservation(self):
        """Test that data is preserved during conversion."""
        # Arrange
        n_samples, seq_len, n_channels = 2, 10, 2
        arr = np.arange(n_samples * seq_len * n_channels).reshape(
            n_samples, seq_len, n_channels
        )

        # Act
        df = to_sktime_df(arr)

        # Assert
        assert all(
            np.array_equal(df.iloc[i, 0][f"channel_{j}"].values, arr[i, :, j])
            for i in range(n_samples)
            for j in range(n_channels)
        )

    def test_to_segments_basic_numpy_segments_is_np_ndarray(self):
        # Arrange
        signal_len = 1000
        window_size = 100
        x = np.random.randn(1, 2, signal_len).astype(np.float32)
        # Act
        segments = to_segments(x, window_size)
        # Act
        # Assert
        assert isinstance(segments, np.ndarray)

    def test_to_segments_basic_numpy_segments_shape_equals_n_1_2_expected_n_segments_wind_segments_is_np_ndarray(self):
        # Arrange
        signal_len = 1000
        window_size = 100
        x = np.random.randn(1, 2, signal_len).astype(np.float32)
        # Act
        segments = to_segments(x, window_size)
        # Act
        # Assert
        assert isinstance(segments, np.ndarray)

    def test_to_segments_basic_numpy_segments_shape_matches_expected(self):
        # Arrange
        signal_len = 1000
        window_size = 100
        x = np.random.randn(1, 2, signal_len).astype(np.float32)
        expected_n_segments = signal_len // window_size
        # Act
        segments = to_segments(x, window_size)
        # Assert
        assert segments.shape == (1, 2, expected_n_segments, window_size)



    def test_to_segments_basic_torch_segments_is_torch_tensor(self):
        # Arrange
        signal_len = 500
        window_size = 50
        x = torch.randn(1, 3, signal_len)
        # Act
        segments = to_segments(x, window_size)
        # Act
        # Assert
        assert isinstance(segments, torch.Tensor)

    def test_to_segments_basic_torch_segments_shape_equals_n_1_3_expected_n_segments_wind_segments_is_torch_tensor(self):
        # Arrange
        signal_len = 500
        window_size = 50
        x = torch.randn(1, 3, signal_len)
        # Act
        segments = to_segments(x, window_size)
        # Act
        # Assert
        assert isinstance(segments, torch.Tensor)

    def test_to_segments_basic_torch_segments_shape_matches_expected(self):
        # Arrange
        signal_len = 500
        window_size = 50
        x = torch.randn(1, 3, signal_len)
        expected_n_segments = signal_len // window_size
        # Act
        segments = to_segments(x, window_size)
        # Assert
        assert segments.shape == (1, 3, expected_n_segments, window_size)



    def test_to_segments_overlap_segments_shape_2_expected_n_segments(self):
        # Arrange
        signal_len = 200
        window_size = 40
        overlap_factor = 2  # 50% overlap
        x = np.random.randn(1, 1, signal_len).astype(np.float32)
        segments = to_segments(x, window_size, overlap_factor=overlap_factor)
        stride = window_size // overlap_factor
        # Act
        expected_n_segments = (signal_len - window_size) // stride + 1
        # Act
        # Assert
        assert segments.shape[-2] == expected_n_segments

    def test_to_segments_overlap_segments_shape_1_window_size(self):
        # Arrange
        signal_len = 200
        window_size = 40
        overlap_factor = 2  # 50% overlap
        x = np.random.randn(1, 1, signal_len).astype(np.float32)
        segments = to_segments(x, window_size, overlap_factor=overlap_factor)
        stride = window_size // overlap_factor
        # Act
        expected_n_segments = (signal_len - window_size) // stride + 1
        # Act
        # Assert
        assert segments.shape[-1] == window_size


    def test_to_segments_no_overlap(self):
        """Test segmentation without overlap."""
        # Arrange
        signal_len = 300
        window_size = 50
        overlap_factor = 1  # stride == window_size
        x = np.random.randn(2, 4, signal_len).astype(np.float32)

        segments = to_segments(x, window_size, overlap_factor=overlap_factor)

        # Act
        expected_n_segments = signal_len // window_size
        # Assert
        assert segments.shape == (2, 4, expected_n_segments, window_size)

    def test_to_segments_different_dimensions(self):
        """Test segmentation along different dimensions."""
        # Test with dim=1, default overlap_factor=1 (stride == window_size).
        # Arrange
        x = np.random.randn(3, 100, 5).astype(np.float32)
        window_size = 20

        segments = to_segments(x, window_size, dim=1)

        # Act
        expected_n_segments = 100 // window_size
        # Assert
        assert segments.shape == (3, expected_n_segments, 5, window_size)

    def test_to_segments_edge_case_exact_fit(self):
        """Test when signal length is exact multiple of window size."""
        # Arrange
        signal_len = 100
        window_size = 100
        x = np.random.randn(1, 1, signal_len).astype(np.float32)

        # Act
        segments = to_segments(x, window_size)

        # Should have exactly 1 segment
        # Assert
        assert segments.shape == (1, 1, 1, window_size)

    def test_to_segments_window_larger_than_signal(self):
        """Window > signal → torch.unfold raises (no degenerate slice)."""
        # Arrange
        signal_len = 50
        window_size = 100
        # Act
        x = np.random.randn(1, 1, signal_len).astype(np.float32)

        # Assert
        with pytest.raises((RuntimeError, ValueError)):
            to_segments(x, window_size)

    def test_to_segments_dtype_preservation_segments_f32_dtype_equals_torch_float32(self):
        # Arrange
        signal_len = 200
        window_size = 50
        # Test float32
        x_f32 = torch.randn(1, 1, signal_len, dtype=torch.float32)
        # Act
        segments_f32 = to_segments(x_f32, window_size)
        # Act
        # Assert
        assert segments_f32.dtype == torch.float32

    def test_to_segments_dtype_preservation_segments_f64_dtype_equals_torch_float64_segments_f32_dtype_equals_torch_float32(self):
        # Arrange
        signal_len = 200
        window_size = 50
        # Test float32
        x_f32 = torch.randn(1, 1, signal_len, dtype=torch.float32)
        # Act
        segments_f32 = to_segments(x_f32, window_size)
        # Act
        # Assert
        assert segments_f32.dtype == torch.float32

    def test_to_segments_dtype_preservation_f64(self):
        # Arrange
        signal_len = 200
        window_size = 50
        x_f64 = torch.randn(1, 1, signal_len, dtype=torch.float64)
        # Act
        segments_f64 = to_segments(x_f64, window_size)
        # Assert
        assert segments_f64.dtype == torch.float64



    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_segments_device_preservation_segments_is_cuda(self):
        # Arrange
        signal_len = 200
        window_size = 50
        x = torch.randn(1, 2, signal_len).cuda()
        # Act
        segments = to_segments(x, window_size)
        # Assert
        assert segments.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_segments_device_preservation_segments_device_equals_x_device(self):
        # Arrange
        signal_len = 200
        window_size = 50
        x = torch.randn(1, 2, signal_len).cuda()
        # Act
        segments = to_segments(x, window_size)
        # Assert
        assert segments.device == x.device


    def test_to_segments_content_verification(self):
        """Test that segment content is correct."""
        # Arrange
        signal_len = 100
        window_size = 10
        x = np.arange(signal_len).reshape(1, 1, signal_len).astype(np.float32)
        stride = window_size  # default overlap_factor=1

        # Act
        segments = to_segments(x, window_size)

        # Assert
        assert all(
            np.array_equal(
                segments[0, 0, i, :],
                np.arange(i * stride, i * stride + window_size),
            )
            for i in range(min(5, segments.shape[2]))
        )

    def test_to_segments_high_overlap_segments_shape_2_expected_n_segments(self):
        # Arrange
        signal_len = 200
        window_size = 50
        overlap_factor = 10  # 90% overlap
        x = np.random.randn(1, 1, signal_len).astype(np.float32)
        segments = to_segments(x, window_size, overlap_factor=overlap_factor)
        stride = window_size // overlap_factor
        # Act
        expected_n_segments = (signal_len - window_size) // stride + 1
        # Act
        # Assert
        assert segments.shape[-2] == expected_n_segments

    def test_to_segments_high_overlap_segments_shape_2_signal_len_window_size(self):
        # Arrange
        signal_len = 200
        window_size = 50
        overlap_factor = 10  # 90% overlap
        x = np.random.randn(1, 1, signal_len).astype(np.float32)
        segments = to_segments(x, window_size, overlap_factor=overlap_factor)
        stride = window_size // overlap_factor
        # Act
        expected_n_segments = (signal_len - window_size) // stride + 1
        # Act
        # Assert
        assert segments.shape[-2] > signal_len // window_size



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_transform.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-04-08 12:41:59 (ywatanabe)"#!/usr/bin/env python3
#
#
#
# import numpy as np
# import pandas as pd
#
# try:
#     import torch
#
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     torch = None
#
# from scitex.decorators import torch_fn
#
#
# def _check_torch():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#
#
# def to_sktime_df(arr):
#     """
#     Convert a 3D numpy array into a DataFrame suitable for sktime.
#
#     Parameters:
#     arr (numpy.ndarray): A 3D numpy array with shape (n_samples, n_channels, seq_len)
#
#     Returns:
#     pandas.DataFrame: A DataFrame in sktime format
#     """
#     if len(arr.shape) != 3:
#         raise ValueError("Input data must be a 3D array")
#
#     n_samples, seq_len, n_channels = arr.shape
#
#     # Initialize an empty DataFrame for sktime format
#     sktime_df = pd.DataFrame(index=range(n_samples), columns=["dim_0"])
#
#     # Iterate over each sample
#     for i in range(n_samples):
#         # Combine all channels into a single cell
#         combined_series = pd.Series(
#             {f"channel_{j}": pd.Series(arr[i, :, j]) for j in range(n_channels)}
#         )
#         sktime_df.iloc[i, 0] = combined_series
#
#     return sktime_df
#
#
# @torch_fn
# def to_segments(x, window_size, overlap_factor=1, dim=-1):
#     stride = window_size // overlap_factor
#     num_windows = (x.size(dim) - window_size) // stride + 1
#     windows = x.unfold(dim, window_size, stride)
#     return windows
#
#
# if __name__ == "__main__":
#     import scitex
#
#     x, t, f = scitex.dsp.demo_sig()
#
#     y = to_segments(x, 256)
#
#     x = 100 * np.random.rand(16, 160, 1000)
#     print(_normalize_time(x))
#
#     x = torch.randn(16, 160, 1000)
#     print(_normalize_time(x))
#
#     x = torch.randn(16, 160, 1000).cuda()
#     print(_normalize_time(x))
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_transform.py
# --------------------------------------------------------------------------------

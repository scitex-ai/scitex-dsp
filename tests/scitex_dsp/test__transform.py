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
        # Act
        # Assert
        from scitex.dsp._transform import _check_torch

        # Should not raise
        _check_torch()


class TestTransform:
    """Test cases for signal transformation functions."""

    def test_import_callable_to_sktime_df(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert callable(to_sktime_df)

    def test_import_callable_to_segments(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
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


    def test_to_sktime_df_single_channel(self):
        """Test conversion with single channel."""
        # Arrange
        n_samples, seq_len, n_channels = 3, 50, 1
        arr = np.random.randn(n_samples, seq_len, n_channels)

        # Act
        df = to_sktime_df(arr)

        # Assert
        assert len(df) == n_samples
        for i in range(n_samples):
            cell_data = df.iloc[i, 0]
            assert "channel_0" in cell_data.index
            assert len(cell_data["channel_0"]) == seq_len

    def test_to_sktime_df_shape_validation_raises_valueerror(self):
        # Arrange
        # Act
        # Arrange
        # Act
        arr_2d = np.random.randn(10, 20)
        # Act
        # Assert
        with pytest.raises(ValueError, match="Input data must be a 3D array"):
            to_sktime_df(arr_2d)

    def test_to_sktime_df_shape_validation_raises_valueerror_raises_valueerror(self):
        # Arrange
        # Act
        arr_2d = np.random.randn(10, 20)
        # Act
        # Assert
        with pytest.raises(ValueError, match="Input data must be a 3D array"):
            to_sktime_df(arr_2d)

    def test_to_sktime_df_shape_validation_raises_valueerror_raises_valueerror_2(self):
        # Arrange
        # Act
        arr_2d = np.random.randn(10, 20)
        # Assert
        with pytest.raises(ValueError, match="Input data must be a 3D array"):
            to_sktime_df(arr_2d)
        # 4D array should raise error
        arr_4d = np.random.randn(5, 10, 20, 3)
        # Act
        # Assert
        with pytest.raises(ValueError, match="Input data must be a 3D array"):
            to_sktime_df(arr_4d)



    def test_to_sktime_df_data_preservation(self):
        """Test that data is preserved during conversion."""
        # Arrange
        # Act
        # Assert
        n_samples, seq_len, n_channels = 2, 10, 2
        arr = np.arange(n_samples * seq_len * n_channels).reshape(
            n_samples, seq_len, n_channels
        )

        df = to_sktime_df(arr)

        # Check that data is preserved
        for i in range(n_samples):
            cell_data = df.iloc[i, 0]
            for j in range(n_channels):
                channel_data = cell_data[f"channel_{j}"]
                expected_data = arr[i, :, j]
                np.testing.assert_array_equal(channel_data.values, expected_data)

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

    def test_to_segments_basic_numpy_segments_shape_equals_n_1_2_expected_n_segments_wind_segments_shape_equals_n_1_2_expected_n_segments_wind(self):
        # Arrange
        signal_len = 1000
        window_size = 100
        x = np.random.randn(1, 2, signal_len).astype(np.float32)
        # Act
        segments = to_segments(x, window_size)
        # Assert
        assert isinstance(segments, np.ndarray)
        expected_n_segments = signal_len // window_size
        # Act
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

    def test_to_segments_basic_torch_segments_shape_equals_n_1_3_expected_n_segments_wind_segments_shape_equals_n_1_3_expected_n_segments_wind(self):
        # Arrange
        signal_len = 500
        window_size = 50
        x = torch.randn(1, 3, signal_len)
        # Act
        segments = to_segments(x, window_size)
        # Assert
        assert isinstance(segments, torch.Tensor)
        expected_n_segments = signal_len // window_size
        # Act
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

    def test_to_segments_dtype_preservation_segments_f64_dtype_equals_torch_float64_segments_f64_dtype_equals_torch_float64(self):
        # Arrange
        signal_len = 200
        window_size = 50
        # Test float32
        x_f32 = torch.randn(1, 1, signal_len, dtype=torch.float32)
        # Act
        segments_f32 = to_segments(x_f32, window_size)
        # Assert
        assert segments_f32.dtype == torch.float32
        # Test float64
        x_f64 = torch.randn(1, 1, signal_len, dtype=torch.float64)
        segments_f64 = to_segments(x_f64, window_size)
        # Act
        # Assert
        assert segments_f64.dtype == torch.float64



    def test_to_segments_device_preservation_segments_is_cuda(self):
        # Arrange
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        signal_len = 200
        window_size = 50
        x = torch.randn(1, 2, signal_len).cuda()
        # Act
        segments = to_segments(x, window_size)
        # Act
        # Assert
        assert segments.is_cuda

    def test_to_segments_device_preservation_segments_device_equals_x_device(self):
        # Arrange
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        signal_len = 200
        window_size = 50
        x = torch.randn(1, 2, signal_len).cuda()
        # Act
        segments = to_segments(x, window_size)
        # Act
        # Assert
        assert segments.device == x.device


    def test_to_segments_content_verification(self):
        """Test that segment content is correct."""
        # Arrange
        # Act
        # Assert
        signal_len = 100
        window_size = 10
        # Create a simple pattern for easy verification
        x = np.arange(signal_len).reshape(1, 1, signal_len).astype(np.float32)

        segments = to_segments(x, window_size)

        # Default overlap_factor=1 → stride == window_size, so the i-th
        # segment starts at i*window_size.
        stride = window_size
        for i in range(min(5, segments.shape[2])):
            start = i * stride
            expected = np.arange(start, start + window_size)
            np.testing.assert_array_equal(segments[0, 0, i, :], expected)

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

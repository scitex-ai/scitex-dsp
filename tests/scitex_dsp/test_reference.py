import pytest

torch = pytest.importorskip("torch")
import numpy as np

import scitex


class TestCommonAverage:
    """Test common average referencing function."""

    def test_import_hasattr_scitex_dsp_reference_take_reference(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.reference, "common_average")

    def test_basic_2d_np_abs_np_mean_result_axis_0_max_1e_05(self):
        # Arrange
        signal = np.random.randn(10, 100)  # 10 channels, 100 samples
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=0)
        # Act
        # Assert
        assert np.abs(np.mean(result, axis=0)).max() < 1e-5

    def test_basic_2d_np_abs_np_std_result_axis_0_1_0_max_0_15(self):
        # Arrange
        signal = np.random.randn(10, 100)  # 10 channels, 100 samples
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=0)
        # Act
        # Assert
        assert np.abs(np.std(result, axis=0) - 1.0).max() < 0.15


    def test_basic_3d_result_shape_equals_signal_shape_3(self):
        """Test common average on 3D signal (trials, channels, time)."""
        # Arrange
        signal = np.random.randn(5, 10, 100)  # 5 trials, 10 channels, 100 samples
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=-2)

        # Check shape preserved
        # Assert
        assert result.shape == signal.shape

        # Check each trial is properly referenced
        for trial in range(5):
            trial_result = result[trial]
            assert np.abs(np.mean(trial_result, axis=0)).max() < 1e-5
            # Allow for Bessel correction
            assert np.abs(np.std(trial_result, axis=0) - 1.0).max() < 0.15

    def test_different_dimensions_calls_randn(self):
        """Test referencing along different dimensions."""
        # Arrange
        # Act
        # Assert
        signal = np.random.randn(4, 5, 6, 100)

        # Test along different dims
        for dim in [0, 1, 2, -2]:
            result = scitex.dsp.reference.common_average(signal, dim=dim)
            assert result.shape == signal.shape

            # Check normalization along specified dimension (float32 tolerance)
            mean_vals = np.mean(result, axis=dim)
            std_vals = np.std(result, axis=dim)
            assert np.abs(mean_vals).max() < 1e-5
            # Allow for Bessel correction
            assert np.abs(std_vals - 1.0).max() < 0.15

    def test_preserves_shape_smoke_case(self):
        """Test that function preserves input shape."""
        # Arrange
        # Act
        # Assert
        shapes = [(10, 100), (5, 10, 100), (2, 5, 10, 100)]
        for shape in shapes:
            signal = np.random.randn(*shape)
            result = scitex.dsp.reference.common_average(signal)
            assert result.shape == signal.shape

    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_result_shape_equals_signal_shape(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_torch_abs_mean_vals_max_1e_05_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_torch_abs_mean_vals_max_1e_05_result_shape_equals_signal_shape(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_torch_abs_mean_vals_max_1e_05_torch_abs_mean_vals_max_1e_05(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Assert
        assert isinstance(result, torch.Tensor)
        assert result.shape == signal.shape
        # Check normalization (float32 tolerance)
        mean_vals = torch.mean(result, dim=1)
        std_vals = torch.std(result, dim=1)
        # Act
        # Assert
        assert torch.abs(mean_vals).max() < 1e-5


    def test_torch_input_torch_abs_std_vals_1_0_max_0_02_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_torch_abs_std_vals_1_0_max_0_02_result_shape_equals_signal_shape(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_torch_abs_std_vals_1_0_max_0_02_torch_abs_std_vals_1_0_max_0_02(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Assert
        assert isinstance(result, torch.Tensor)
        assert result.shape == signal.shape
        # Check normalization (float32 tolerance)
        mean_vals = torch.mean(result, dim=1)
        std_vals = torch.std(result, dim=1)
        # Act
        # Assert
        assert torch.abs(std_vals - 1.0).max() < 0.02



    def test_constant_channels_np_all_np_isnan_result_or_np_all_np_isinf_result(self):
        """Test with constant values across channels."""
        # Arrange
        signal = np.ones((5, 100))
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=0)
        # Division by zero results in NaN values (std=0 for constant signal)
        # Assert
        assert np.all(np.isnan(result)) or np.all(np.isinf(result))


class TestRandom:
    """Test random channel referencing function."""

    def test_import_hasattr_scitex_dsp_reference_take_reference(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.reference, "random")

    def test_basic_2d_result_shape_equals_signal_shape(self):
        # Arrange
        np.random.seed(42)
        signal = np.random.randn(10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=0)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_basic_2d_not_np_allclose_result_signal(self):
        # Arrange
        np.random.seed(42)
        signal = np.random.randn(10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=0)
        # Act
        # Assert
        assert not np.allclose(result, signal)


    def test_basic_3d_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_not_np_allclose_result_signal(self):
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Act
        # Assert
        assert not np.allclose(result, signal)


    def test_different_dimensions_calls_randn(self):
        """Test referencing along different dimensions."""
        # Arrange
        # Act
        # Assert
        signal = np.random.randn(4, 5, 6)

        for dim in [0, 1, 2]:
            result = scitex.dsp.reference.random(signal, dim=dim)
            assert result.shape == signal.shape

            # Check that result is different from original
            assert not np.allclose(result, signal)

    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        torch.manual_seed(42)
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_result_shape_equals_signal_shape(self):
        # Arrange
        torch.manual_seed(42)
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Act
        # Assert
        assert result.shape == signal.shape


    def test_randomness_all_different(self):
        """Test that function produces different results on different calls."""
        # Arrange
        signal = np.random.randn(10, 100)

        # Get multiple results
        results = []
        for _ in range(5):
            results.append(scitex.dsp.reference.random(signal, dim=0))

        # Check that results are different
        all_different = True
        # Act
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if np.allclose(results[i], results[j]):
                    all_different = False
                    break

        # Assert
        assert all_different, "Random reference should produce different results"

    def test_preserves_shape_smoke_case(self):
        """Test that function preserves input shape."""
        # Arrange
        # Act
        # Assert
        shapes = [(10, 100), (5, 10, 100), (2, 5, 10, 100)]
        for shape in shapes:
            signal = np.random.randn(*shape)
            result = scitex.dsp.reference.random(signal)
            assert result.shape == signal.shape


class TestTakeReference:
    """Test specific channel referencing function."""

    def test_import_hasattr_scitex_dsp_reference_take_reference(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.reference, "take_reference")

    def test_basic_2d_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(10, 100)
        ref_channel = 3
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=0)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_basic_2d_np_allclose_result_ref_channel_0(self):
        # Arrange
        signal = np.random.randn(10, 100)
        ref_channel = 3
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=0)
        # Act
        # Assert
        assert np.allclose(result[ref_channel], 0)


    def test_basic_3d_result_shape_equals_signal_shape_3(self):
        """Test reference to specific channel on 3D signal."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        ref_channel = 7

        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)

        # Assert
        assert result.shape == signal.shape

        # Check each trial
        for trial in range(5):
            # Reference channel should be zero
            assert np.allclose(result[trial, ref_channel], 0)

            # Other channels correctly referenced (float32 tolerance)
            for ch in range(10):
                if ch != ref_channel:
                    expected = signal[trial, ch] - signal[trial, ref_channel]
                    assert np.allclose(
                        result[trial, ch], expected, rtol=1e-5, atol=1e-5
                    )

    def test_different_dimensions_calls_randn(self):
        """Test referencing along different dimensions."""
        # Arrange
        # Act
        # Assert
        signal = np.random.randn(4, 5, 6, 7)

        # Test along each dimension
        for dim in range(4):
            ref_idx = 2
            result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=dim)

            assert result.shape == signal.shape

            # Create index to check reference slice is zero
            idx = [slice(None)] * 4
            idx[dim] = ref_idx
            assert np.allclose(result[tuple(idx)], 0)

    def test_negative_dimension_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(3, 4, 5, 6)
        # Test dim=-2 (equivalent to dim=2)
        ref_idx = 1
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=-2)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_negative_dimension_np_allclose_result_ref_idx_0(self):
        # Arrange
        signal = np.random.randn(3, 4, 5, 6)
        # Test dim=-2 (equivalent to dim=2)
        ref_idx = 1
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=-2)
        # Act
        # Assert
        assert np.allclose(result[:, :, ref_idx, :], 0)


    def test_edge_indices_np_allclose_result_0_0(self):
        # Arrange
        signal = np.random.randn(10, 100)
        # First channel
        # Act
        result = scitex.dsp.reference.take_reference(signal, 0, dim=0)
        # Act
        # Assert
        assert np.allclose(result[0], 0)

    def test_edge_indices_np_allclose_result_9_0_np_allclose_result_0_0(self):
        # Arrange
        signal = np.random.randn(10, 100)
        # First channel
        # Act
        result = scitex.dsp.reference.take_reference(signal, 0, dim=0)
        # Act
        # Assert
        assert np.allclose(result[0], 0)

    def test_edge_indices_np_allclose_result_9_0_np_allclose_result_9_0(self):
        # Arrange
        signal = np.random.randn(10, 100)
        # First channel
        # Act
        result = scitex.dsp.reference.take_reference(signal, 0, dim=0)
        # Assert
        assert np.allclose(result[0], 0)
        # Last channel
        result = scitex.dsp.reference.take_reference(signal, 9, dim=0)
        # Act
        # Assert
        assert np.allclose(result[9], 0)



    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        ref_channel = 32
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_result_shape_equals_signal_shape(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        ref_channel = 32
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_torch_allclose_result_ref_channel_torch_zeros_like_result_re(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        ref_channel = 32
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Act
        # Assert
        assert torch.allclose(
            result[:, ref_channel, :], torch.zeros_like(result[:, ref_channel, :])
        )


    def test_preserves_shape_smoke_case(self):
        """Test that function preserves input shape."""
        # Arrange
        # Act
        # Assert
        shapes = [(10, 100), (5, 10, 100), (2, 5, 10, 100)]
        for shape in shapes:
            signal = np.random.randn(*shape)
            result = scitex.dsp.reference.take_reference(signal, 0)
            assert result.shape == signal.shape

    def test_invalid_index_raises_indexerror_runtimeerror_a(self):
        """Test with invalid reference index."""
        # Arrange
        # Act
        signal = np.random.randn(10, 100)

        # This should raise an error
        # Assert
        with pytest.raises((IndexError, RuntimeError, AssertionError)):
            scitex.dsp.reference.take_reference(signal, 10, dim=0)  # Out of bounds


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/reference.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "ywatanabe (2024-11-02 22:48:44)"
# # File: ./scitex_repo/src/scitex/dsp/reference.py
#
# try:
#     import torch as _torch
#
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     _torch = None
#
# from scitex.decorators import torch_fn as _torch_fn
#
#
# def _check_torch():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#
#
# @_torch_fn
# def common_average(x, dim=-2):
#     _check_torch()
#     re_referenced = (x - x.mean(dim=dim, keepdims=True)) / x.std(dim=dim, keepdims=True)
#     assert x.shape == re_referenced.shape
#     return re_referenced
#
#
# @_torch_fn
# def random(x, dim=-2):
#     _check_torch()
#     idx_all = [slice(None)] * x.ndim
#     idx_rand_dim = _torch.randperm(x.shape[dim])
#     idx_all[dim] = idx_rand_dim
#     y = x[idx_all]
#     re_referenced = x - y
#     assert x.shape == re_referenced.shape
#     return re_referenced
#
#
# @_torch_fn
# def take_reference(x, tgt_indi, dim=-2):
#     _check_torch()
#     idx_all = [slice(None)] * x.ndim
#     idx_all[dim] = tgt_indi
#     ref = x[tuple(idx_all)].unsqueeze(dim)
#     re_referenced = x - ref
#     assert x.shape == re_referenced.shape
#     return re_referenced
#
#
# if __name__ == "__main__":
#     import scitex
#
#     x, f, t = scitex.dsp.demo_sig()
#     y = common_average(x)
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/reference.py
# --------------------------------------------------------------------------------

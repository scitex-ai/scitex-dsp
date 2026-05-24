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
        # Assert
        assert np.abs(np.mean(result, axis=0)).max() < 1e-5

    def test_basic_2d_np_abs_np_std_result_axis_0_1_0_max_0_15(self):
        # Arrange
        signal = np.random.randn(10, 100)  # 10 channels, 100 samples
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=0)
        # Assert
        assert np.abs(np.std(result, axis=0) - 1.0).max() < 0.15


    def test_basic_3d_result_shape_equals_signal_shape_3(self):
        """Test common average on 3D signal preserves shape."""
        # Arrange
        signal = np.random.randn(5, 10, 100)  # 5 trials, 10 channels, 100 samples
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=-2)
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_each_trial_zero_mean(self):
        """Test common average on 3D signal zeros each trial mean."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=-2)
        # Assert
        max_abs_means = max(
            np.abs(np.mean(result[trial], axis=0)).max() for trial in range(5)
        )
        assert max_abs_means < 1e-5

    def test_basic_3d_each_trial_unit_std(self):
        """Test common average on 3D signal yields unit std per trial."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=-2)
        # Assert
        max_std_diff = max(
            np.abs(np.std(result[trial], axis=0) - 1.0).max() for trial in range(5)
        )
        assert max_std_diff < 0.15

    @pytest.mark.parametrize("dim", [0, 1, 2, -2])
    def test_different_dimensions_preserves_shape(self, dim):
        """Test referencing along different dimensions preserves shape."""
        # Arrange
        signal = np.random.randn(4, 5, 6, 100)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=dim)
        # Assert
        assert result.shape == signal.shape

    @pytest.mark.parametrize("dim", [0, 1, 2, -2])
    def test_different_dimensions_zero_mean(self, dim):
        """Test referencing along different dimensions yields zero mean."""
        # Arrange
        signal = np.random.randn(4, 5, 6, 100)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=dim)
        # Assert
        assert np.abs(np.mean(result, axis=dim)).max() < 1e-5

    @pytest.mark.parametrize("dim", [0, 1, 2, -2])
    def test_different_dimensions_unit_std(self, dim):
        """Test referencing along different dimensions yields unit std."""
        # Arrange
        signal = np.random.randn(4, 5, 6, 100)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=dim)
        # Assert
        assert np.abs(np.std(result, axis=dim) - 1.0).max() < 0.15

    @pytest.mark.parametrize("shape", [(10, 100), (5, 10, 100), (2, 5, 10, 100)])
    def test_preserves_shape_smoke_case(self, shape):
        """Test that function preserves input shape."""
        # Arrange
        signal = np.random.randn(*shape)
        # Act
        result = scitex.dsp.reference.common_average(signal)
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_result_shape_equals_signal_shape(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_torch_abs_mean_vals_max_1e_05(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        mean_vals = torch.mean(result, dim=1)
        # Assert
        assert torch.abs(mean_vals).max() < 1e-5

    def test_torch_input_torch_abs_std_vals_1_0_max_0_02(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=1)
        std_vals = torch.std(result, dim=1)
        # Assert
        assert torch.abs(std_vals - 1.0).max() < 0.02

    def test_constant_channels_np_all_np_isnan_result_or_np_all_np_isinf_result(self):
        """Test with constant values across channels."""
        # Arrange
        signal = np.ones((5, 100))
        # Act
        result = scitex.dsp.reference.common_average(signal, dim=0)
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
        # Assert
        assert result.shape == signal.shape

    def test_basic_2d_not_np_allclose_result_signal(self):
        # Arrange
        np.random.seed(42)
        signal = np.random.randn(10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=0)
        # Assert
        assert not np.allclose(result, signal)


    def test_basic_3d_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_not_np_allclose_result_signal(self):
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Assert
        assert not np.allclose(result, signal)


    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_different_dimensions_preserves_shape(self, dim):
        """Test referencing along different dimensions preserves shape."""
        # Arrange
        signal = np.random.randn(4, 5, 6)
        # Act
        result = scitex.dsp.reference.random(signal, dim=dim)
        # Assert
        assert result.shape == signal.shape

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_different_dimensions_differs_from_signal(self, dim):
        """Test referencing along different dimensions differs from input."""
        # Arrange
        signal = np.random.randn(4, 5, 6)
        # Act
        result = scitex.dsp.reference.random(signal, dim=dim)
        # Assert
        assert not np.allclose(result, signal)

    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        torch.manual_seed(42)
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_result_shape_equals_signal_shape(self):
        # Arrange
        torch.manual_seed(42)
        signal = torch.randn(8, 64, 1000)
        # Act
        result = scitex.dsp.reference.random(signal, dim=1)
        # Assert
        assert result.shape == signal.shape


    def test_randomness_all_different(self):
        """Test that function produces different results on different calls."""
        # Arrange
        signal = np.random.randn(10, 100)
        results = [
            scitex.dsp.reference.random(signal, dim=0) for _ in range(5)
        ]
        # Act
        any_equal = any(
            np.allclose(results[i], results[j])
            for i in range(len(results))
            for j in range(i + 1, len(results))
        )
        # Assert
        assert not any_equal, "Random reference should produce different results"

    @pytest.mark.parametrize("shape", [(10, 100), (5, 10, 100), (2, 5, 10, 100)])
    def test_preserves_shape_smoke_case(self, shape):
        """Test that function preserves input shape."""
        # Arrange
        signal = np.random.randn(*shape)
        # Act
        result = scitex.dsp.reference.random(signal)
        # Assert
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
        # Assert
        assert result.shape == signal.shape

    def test_basic_2d_np_allclose_result_ref_channel_0(self):
        # Arrange
        signal = np.random.randn(10, 100)
        ref_channel = 3
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=0)
        # Assert
        assert np.allclose(result[ref_channel], 0)


    def test_basic_3d_result_shape_equals_signal_shape_3(self):
        """Test reference to specific channel on 3D signal preserves shape."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        ref_channel = 7
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_reference_channel_is_zero(self):
        """Test reference channel becomes zero per trial."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        ref_channel = 7
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Assert
        assert all(
            np.allclose(result[trial, ref_channel], 0) for trial in range(5)
        )

    def test_basic_3d_other_channels_correctly_referenced(self):
        """Test non-reference channels equal signal minus reference."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        ref_channel = 7
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Assert
        assert all(
            np.allclose(
                result[trial, ch],
                signal[trial, ch] - signal[trial, ref_channel],
                rtol=1e-5,
                atol=1e-5,
            )
            for trial in range(5)
            for ch in range(10)
            if ch != ref_channel
        )

    @pytest.mark.parametrize("dim", [0, 1, 2, 3])
    def test_different_dimensions_preserves_shape(self, dim):
        """Test referencing along different dimensions preserves shape."""
        # Arrange
        signal = np.random.randn(4, 5, 6, 7)
        ref_idx = 2
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=dim)
        # Assert
        assert result.shape == signal.shape

    @pytest.mark.parametrize("dim", [0, 1, 2, 3])
    def test_different_dimensions_ref_slice_zero(self, dim):
        """Test referencing along different dimensions zeros reference slice."""
        # Arrange
        signal = np.random.randn(4, 5, 6, 7)
        ref_idx = 2
        idx = [slice(None)] * 4
        idx[dim] = ref_idx
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=dim)
        # Assert
        assert np.allclose(result[tuple(idx)], 0)

    def test_negative_dimension_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(3, 4, 5, 6)
        ref_idx = 1
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=-2)
        # Assert
        assert result.shape == signal.shape

    def test_negative_dimension_np_allclose_result_ref_idx_0(self):
        # Arrange
        signal = np.random.randn(3, 4, 5, 6)
        ref_idx = 1
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_idx, dim=-2)
        # Assert
        assert np.allclose(result[:, :, ref_idx, :], 0)


    def test_edge_indices_first_channel_is_zero(self):
        # Arrange
        signal = np.random.randn(10, 100)
        # Act
        result = scitex.dsp.reference.take_reference(signal, 0, dim=0)
        # Assert
        assert np.allclose(result[0], 0)

    def test_edge_indices_last_channel_is_zero(self):
        # Arrange
        signal = np.random.randn(10, 100)
        # Act
        result = scitex.dsp.reference.take_reference(signal, 9, dim=0)
        # Assert
        assert np.allclose(result[9], 0)


    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        ref_channel = 32
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_result_shape_equals_signal_shape(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        ref_channel = 32
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Assert
        assert result.shape == signal.shape

    def test_torch_input_torch_allclose_result_ref_channel_torch_zeros_like(self):
        # Arrange
        signal = torch.randn(8, 64, 1000)
        ref_channel = 32
        # Act
        result = scitex.dsp.reference.take_reference(signal, ref_channel, dim=1)
        # Assert
        assert torch.allclose(
            result[:, ref_channel, :], torch.zeros_like(result[:, ref_channel, :])
        )


    @pytest.mark.parametrize("shape", [(10, 100), (5, 10, 100), (2, 5, 10, 100)])
    def test_preserves_shape_smoke_case(self, shape):
        """Test that function preserves input shape."""
        # Arrange
        signal = np.random.randn(*shape)
        # Act
        result = scitex.dsp.reference.take_reference(signal, 0)
        # Assert
        assert result.shape == signal.shape

    def test_invalid_index_raises_indexerror_runtimeerror_a(self):
        """Test with invalid reference index."""
        # Arrange
        signal = np.random.randn(10, 100)
        # Act
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

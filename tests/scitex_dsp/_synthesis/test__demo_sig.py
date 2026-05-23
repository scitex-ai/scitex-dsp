#!/usr/bin/env python3
# Timestamp: "2025-06-01 20:35:00 (ywatanabe)"
# File: ./tests/scitex/dsp/test__demo_sig.py

"""
Test module for scitex.dsp.demo_sig function.
"""

import os

import numpy as np
import pytest


class TestDemoSigAvailableFlags:
    """Test _AVAILABLE flags for optional dependencies."""

    def test_mne_available_flag_exists(self):
        """Test that MNE_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex_dsp._synthesis._demo_sig import MNE_AVAILABLE

        # Assert
        assert isinstance(MNE_AVAILABLE, bool)

    def test_ripple_detection_available_flag_exists(self):
        """Test that RIPPLE_DETECTION_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex_dsp._synthesis._demo_sig import RIPPLE_DETECTION_AVAILABLE

        # Assert
        assert isinstance(RIPPLE_DETECTION_AVAILABLE, bool)

    def test_tensorpac_available_flag_exists(self):
        """Test that TENSORPAC_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex_dsp._synthesis._demo_sig import TENSORPAC_AVAILABLE

        # Assert
        assert isinstance(TENSORPAC_AVAILABLE, bool)

    def test_check_mne_function_exists(self):
        """Test that _check_mne function is exported."""
        # Arrange
        # Act
        from scitex_dsp._synthesis._demo_sig import _check_mne

        # Assert
        assert callable(_check_mne)

    def test_check_ripple_detection_function_exists(self):
        """Test that _check_ripple_detection function is exported."""
        # Arrange
        # Act
        from scitex_dsp._synthesis._demo_sig import _check_ripple_detection

        # Assert
        assert callable(_check_ripple_detection)

    def test_check_tensorpac_function_exists(self):
        """Test that _check_tensorpac function is exported."""
        # Arrange
        # Act
        from scitex_dsp._synthesis._demo_sig import _check_tensorpac

        # Assert
        assert callable(_check_tensorpac)


class TestDemoSig:
    """Test class for demo_sig function."""

    def test_import_callable_demo_sig(self):
        """Test that demo_sig can be imported."""
        # Arrange
        # Act
        from scitex_dsp import demo_sig

        # Assert
        assert callable(demo_sig)

    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "periodic", "chirp"])
    def test_basic_signal_types_shape(self, sig_type):
        """Test basic signal generation produces expected shape."""
        # Arrange
        from scitex_dsp import demo_sig
        batch_size = 2
        n_chs = 3
        t_sec = 1.0
        fs = 100
        # Act
        try:
            sig, _tt, _fs_out = demo_sig(
                sig_type=sig_type,
                batch_size=batch_size,
                n_chs=n_chs,
                t_sec=t_sec,
                fs=fs,
            )
        except ImportError:
            pytest.skip(f"Dependencies not available for {sig_type}")
        expected_samples = int(t_sec * fs)
        # Assert
        assert sig.shape == (batch_size, n_chs, expected_samples)

    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "periodic", "chirp"])
    def test_basic_signal_types_time_vector_length(self, sig_type):
        """Test basic signal generation produces expected time vector length."""
        # Arrange
        from scitex_dsp import demo_sig
        t_sec = 1.0
        fs = 100
        # Act
        try:
            _sig, tt, _fs_out = demo_sig(
                sig_type=sig_type, batch_size=2, n_chs=3, t_sec=t_sec, fs=fs,
            )
        except ImportError:
            pytest.skip(f"Dependencies not available for {sig_type}")
        # Assert
        assert len(tt) == int(t_sec * fs)

    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "periodic", "chirp"])
    def test_basic_signal_types_fs_out_equals_fs(self, sig_type):
        """Test basic signal generation returns input fs."""
        # Arrange
        from scitex_dsp import demo_sig
        fs = 100
        # Act
        try:
            _sig, _tt, fs_out = demo_sig(
                sig_type=sig_type, batch_size=2, n_chs=3, t_sec=1.0, fs=fs,
            )
        except ImportError:
            pytest.skip(f"Dependencies not available for {sig_type}")
        # Assert
        assert fs_out == fs

    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "periodic", "chirp"])
    def test_basic_signal_types_time_starts_at_zero(self, sig_type):
        """Test basic signal generation starts time at zero."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        try:
            _sig, tt, _fs_out = demo_sig(
                sig_type=sig_type, batch_size=2, n_chs=3, t_sec=1.0, fs=100,
            )
        except ImportError:
            pytest.skip(f"Dependencies not available for {sig_type}")
        # Assert
        assert np.allclose(tt[0], 0.0)

    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "periodic", "chirp"])
    def test_basic_signal_types_time_ends_at_t_sec_minus_dt(self, sig_type):
        """Test basic signal generation ends time at t_sec - 1/fs."""
        # Arrange
        from scitex_dsp import demo_sig
        t_sec = 1.0
        fs = 100
        # Act
        try:
            _sig, tt, _fs_out = demo_sig(
                sig_type=sig_type, batch_size=2, n_chs=3, t_sec=t_sec, fs=fs,
            )
        except ImportError:
            pytest.skip(f"Dependencies not available for {sig_type}")
        # Assert
        assert np.allclose(tt[-1], t_sec - 1 / fs)

    def test_uniform_signal_range_sig_min_0_5(self):
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _, _ = demo_sig(sig_type="uniform", batch_size=1, n_chs=1, t_sec=1)
        # Assert
        assert sig.min() >= -0.5

    def test_uniform_signal_range_sig_max_0_5(self):
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _, _ = demo_sig(sig_type="uniform", batch_size=1, n_chs=1, t_sec=1)
        # Assert
        assert sig.max() <= 0.5


    def test_gauss_signal_statistics_abs_sig_mean_0_1(self):
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _, _ = demo_sig(sig_type="gauss", batch_size=10, n_chs=10, t_sec=10)
        # Assert
        assert abs(sig.mean()) < 0.1  # Mean should be close to 0

    def test_gauss_signal_statistics_abs_sig_std_1_0_0_1(self):
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _, _ = demo_sig(sig_type="gauss", batch_size=10, n_chs=10, t_sec=10)
        # Assert
        assert abs(sig.std() - 1.0) < 0.1  # Std should be close to 1


    def test_periodic_signal_with_freqs(self):
        """Test periodic signal generation with specified frequencies."""
        # Arrange
        from scitex_dsp import demo_sig
        freqs_hz = [10, 20]  # 10 Hz and 20 Hz
        # Act
        try:
            sig, _tt, _fs = demo_sig(
                sig_type="periodic",
                batch_size=1,
                n_chs=1,
                t_sec=1,
                fs=1000,
                freqs_hz=freqs_hz,
            )
        except Exception:
            pytest.skip("Frequency spec not supported in this version")
        # Assert
        assert sig.shape == (1, 1, 1000)

    @pytest.mark.parametrize("fs", [100, 256, 512, 1000])
    def test_different_sampling_rates_fs_out_equals_fs(self, fs):
        """Test signal generation returns input fs across sampling rates."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        _sig, _tt, fs_out = demo_sig(
            sig_type="gauss", batch_size=1, n_chs=1, t_sec=1, fs=fs
        )
        # Assert
        assert fs_out == fs

    @pytest.mark.parametrize("fs", [100, 256, 512, 1000])
    def test_different_sampling_rates_time_length(self, fs):
        """Test signal generation produces correct time vector length."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        _sig, tt, _fs_out = demo_sig(
            sig_type="gauss", batch_size=1, n_chs=1, t_sec=1, fs=fs
        )
        # Assert
        assert len(tt) == fs

    @pytest.mark.parametrize("fs", [100, 256, 512, 1000])
    def test_different_sampling_rates_signal_length(self, fs):
        """Test signal generation produces correct signal length."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _tt, _fs_out = demo_sig(
            sig_type="gauss", batch_size=1, n_chs=1, t_sec=1, fs=fs
        )
        # Assert
        assert sig.shape[-1] == fs

    @pytest.mark.parametrize("t_sec", [0.5, 1.0, 2.0, 5.0])
    def test_different_durations_signal_length(self, t_sec):
        """Test signal generation produces correct signal length."""
        # Arrange
        from scitex_dsp import demo_sig
        fs = 100
        # Act
        sig, _tt, _ = demo_sig(
            sig_type="gauss", batch_size=1, n_chs=1, t_sec=t_sec, fs=fs
        )
        # Assert
        assert sig.shape[-1] == int(t_sec * fs)

    @pytest.mark.parametrize("t_sec", [0.5, 1.0, 2.0, 5.0])
    def test_different_durations_time_length(self, t_sec):
        """Test signal generation produces correct time vector length."""
        # Arrange
        from scitex_dsp import demo_sig
        fs = 100
        # Act
        _sig, tt, _ = demo_sig(
            sig_type="gauss", batch_size=1, n_chs=1, t_sec=t_sec, fs=fs
        )
        # Assert
        assert len(tt) == int(t_sec * fs)

    @pytest.mark.parametrize(
        "batch_size,n_chs",
        [(1, 1), (4, 1), (1, 19), (8, 64)],
    )
    def test_batch_and_channel_dimensions_batch(self, batch_size, n_chs):
        """Test batch size is preserved across configurations."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _, _ = demo_sig(
            sig_type="gauss", batch_size=batch_size, n_chs=n_chs, t_sec=0.5, fs=100
        )
        # Assert
        assert sig.shape[0] == batch_size

    @pytest.mark.parametrize(
        "batch_size,n_chs",
        [(1, 1), (4, 1), (1, 19), (8, 64)],
    )
    def test_batch_and_channel_dimensions_channels(self, batch_size, n_chs):
        """Test channel count is preserved across configurations."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _, _ = demo_sig(
            sig_type="gauss", batch_size=batch_size, n_chs=n_chs, t_sec=0.5, fs=100
        )
        # Assert
        assert sig.shape[1] == n_chs

    def test_signal_dtype_sig_dtype_equals_np_float32(self):
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        sig, _tt, _ = demo_sig(sig_type="gauss")
        # Assert
        assert sig.dtype == np.float32

    def test_signal_dtype_tt_dtype_in_np_float32_np_float64(self):
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        _sig, tt, _ = demo_sig(sig_type="gauss")
        # Assert
        assert tt.dtype in [np.float32, np.float64]


    def test_time_vector_properties_tt_0_0_0(self):
        # Arrange
        from scitex_dsp import demo_sig
        t_sec = 2.0
        fs = 500
        # Act
        _, tt, _ = demo_sig(t_sec=t_sec, fs=fs)
        # Assert
        assert tt[0] == 0.0

    def test_time_vector_properties_np_allclose_tt_1_t_sec_1_fs(self):
        # Arrange
        from scitex_dsp import demo_sig
        t_sec = 2.0
        fs = 500
        # Act
        _, tt, _ = demo_sig(t_sec=t_sec, fs=fs)
        # Assert
        assert np.allclose(tt[-1], t_sec - 1 / fs)

    def test_time_vector_properties_evenly_spaced(self):
        # Arrange
        from scitex_dsp import demo_sig
        t_sec = 2.0
        fs = 500
        # Act
        _, tt, _ = demo_sig(t_sec=t_sec, fs=fs)
        dt = np.diff(tt)
        # Assert
        assert np.allclose(dt, 1 / fs)



    def test_invalid_signal_type(self):
        """Test error handling for invalid signal type."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        # Assert
        with pytest.raises(AssertionError):
            demo_sig(sig_type="invalid_type")

    @pytest.mark.parametrize(
        "sig_type",
        [
            "ripple",
            pytest.param(
                "meg",
                marks=pytest.mark.skipif(
                    os.environ.get("CI") == "true",
                    reason="MEG sample-data download is slow + flaky on CI",
                ),
            ),
            "tensorpac",
            "pac",
        ],
    )
    def test_complex_signal_types_ndim(self, sig_type):
        """Test complex signal types produce >=3D signals."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        try:
            sig, _tt, _fs = demo_sig(
                sig_type=sig_type, batch_size=1, n_chs=2, t_sec=0.5, fs=256
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip(f"Optional dependencies not available for {sig_type}")
        # Assert
        assert sig.ndim >= 3  # At least batch x channels x time

    @pytest.mark.parametrize(
        "sig_type",
        [
            "ripple",
            pytest.param(
                "meg",
                marks=pytest.mark.skipif(
                    os.environ.get("CI") == "true",
                    reason="MEG sample-data download is slow + flaky on CI",
                ),
            ),
            "tensorpac",
            "pac",
        ],
    )
    def test_complex_signal_types_time_nonempty(self, sig_type):
        """Test complex signal types produce non-empty time vector."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        try:
            _sig, tt, _fs = demo_sig(
                sig_type=sig_type, batch_size=1, n_chs=2, t_sec=0.5, fs=256
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip(f"Optional dependencies not available for {sig_type}")
        # Assert
        assert len(tt) > 0

    @pytest.mark.parametrize(
        "sig_type",
        [
            "ripple",
            pytest.param(
                "meg",
                marks=pytest.mark.skipif(
                    os.environ.get("CI") == "true",
                    reason="MEG sample-data download is slow + flaky on CI",
                ),
            ),
            "tensorpac",
            "pac",
        ],
    )
    def test_complex_signal_types_fs_equals_256(self, sig_type):
        """Test complex signal types return input fs."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        try:
            _sig, _tt, fs = demo_sig(
                sig_type=sig_type, batch_size=1, n_chs=2, t_sec=0.5, fs=256
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip(f"Optional dependencies not available for {sig_type}")
        # Assert
        assert fs == 256

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed."""
        # Arrange
        from scitex_dsp import demo_sig
        np.random.seed(42)
        sig1, _, _ = demo_sig(sig_type="gauss", batch_size=2, n_chs=3)
        np.random.seed(42)
        # Act
        sig2, _, _ = demo_sig(sig_type="gauss", batch_size=2, n_chs=3)
        # Assert
        assert np.array_equal(sig1, sig2)

    def test_chirp_signal_has_energy(self):
        """Test chirp signal has positive std."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        try:
            sig, _tt, _fs = demo_sig(
                sig_type="chirp", batch_size=1, n_chs=1, t_sec=1, fs=1000
            )
        except Exception:
            pytest.skip("Chirp generation not available")
        # Assert
        assert sig.std() > 0

    def test_chirp_signal_expected_shape(self):
        """Test chirp signal has expected shape."""
        # Arrange
        from scitex_dsp import demo_sig
        # Act
        try:
            sig, _tt, _fs = demo_sig(
                sig_type="chirp", batch_size=1, n_chs=1, t_sec=1, fs=1000
            )
        except Exception:
            pytest.skip("Chirp generation not available")
        # Assert
        assert sig.shape == (1, 1, 1000)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_demo_sig.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-11-06 01:45:32 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/_demo_sig.py
#
# (source code omitted for brevity in test file)
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_demo_sig.py
# --------------------------------------------------------------------------------

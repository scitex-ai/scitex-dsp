import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scitex

# This file exercises the umbrella `scitex.plt.ax` convenience layer
# (and the dead `scitex.dsp.example` module) — neither is part of the
# scitex-dsp standalone surface, and both depend on the umbrella's
# optional `[plt]` extra. Skip the whole module unless those are
# available. See HANDOFF: "test_example.py is dead — rewrite or move
# to legacy".
pytest.importorskip("scitex.plt")
if not hasattr(getattr(scitex, "plt", None), "ax"):
    pytest.skip(
        "scitex.plt.ax not available; install scitex[plt] to run.",
        allow_module_level=True,
    )


class TestCalcNormResampleFiltHilbert:
    """Test calc_norm_resample_filt_hilbert function."""

    def test_import_hasattr_scitex_dsp_example_plot_psd(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.example, "calc_norm_resample_filt_hilbert")

    def test_basic_functionality_sigs_is_pd_dataframe(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=2, fs=1000, sig_type="chirp")
        # Apply function
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        # Assert
        assert isinstance(sigs, pd.DataFrame)

    def test_basic_functionality_sigs_index_name_index(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=2, fs=1000, sig_type="chirp")
        # Apply function
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        # Assert
        assert sigs.index.name == "index"

    def test_basic_functionality_len_sigs_columns_10(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=2, fs=1000, sig_type="chirp")
        # Apply function
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        # Assert
        assert len(sigs.columns) > 10  # Should have multiple processing steps


    def test_output_columns_all_any_col_in_c_for_c_in_sigs_columns_for_col_in_(self):
        """Test that all expected columns are present."""
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

        # Act
        expected_cols = [
            "orig",
            "z_normed",
            "minmax_normed",
            "resampled",
            "gaussian_noise_added",
            "white_noise_added",
            "pink_noise_added",
            "brown_noise_added",
            "hilbert_amp",
            "hilbert_pha",
        ]

        # Assert
        assert all(any((col in c for c in sigs.columns)) for col in expected_cols), f'Missing column: {col}'

    def test_signal_shapes_calls_demo_sig(self):
        """Test that signal shapes are preserved correctly."""
        # Arrange
        # Act
        # Assert
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

        # Original signal shape
        orig_shape = sigs["orig"][0].shape

        # Most signals should preserve shape
        for col in ["z_normed", "minmax_normed", "hilbert_amp", "hilbert_pha"]:
            sig, _, _ = sigs[col]
            assert sig.shape == orig_shape, f"{col} shape mismatch"

    def test_resampling_shape_resampled_sig_shape_1_expected_samples(self):
        # Arrange
        src_fs = 1024
        tgt_fs = 512  # Default target in example
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=src_fs)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Check resampled signal
        resampled_sig, resampled_tt, resampled_fs = sigs["resampled"]
        # Time dimension should be halved
        # Act
        expected_samples = xx.shape[-1] // (src_fs // tgt_fs)
        # Act
        # Assert
        assert resampled_sig.shape[-1] == expected_samples

    def test_resampling_shape_len_resampled_tt_expected_samples(self):
        # Arrange
        src_fs = 1024
        tgt_fs = 512  # Default target in example
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=src_fs)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Check resampled signal
        resampled_sig, resampled_tt, resampled_fs = sigs["resampled"]
        # Time dimension should be halved
        # Act
        expected_samples = xx.shape[-1] // (src_fs // tgt_fs)
        # Act
        # Assert
        assert len(resampled_tt) == expected_samples

    def test_resampling_shape_resampled_fs_equals_tgt_fs(self):
        # Arrange
        src_fs = 1024
        tgt_fs = 512  # Default target in example
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=src_fs)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Check resampled signal
        resampled_sig, resampled_tt, resampled_fs = sigs["resampled"]
        # Time dimension should be halved
        # Act
        expected_samples = xx.shape[-1] // (src_fs // tgt_fs)
        # Act
        # Assert
        assert resampled_fs == tgt_fs


    def test_tensorpac_signal_sigs_orig_0_shape_2_10_1000(self):
        """Test handling of tensorpac signal type."""
        # Create 3D signal to simulate tensorpac
        # Arrange
        xx_3d = np.random.randn(2, 10, 1000, 2)  # Extra dimension
        tt = np.linspace(0, 1, 1000)
        fs = 1000

        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx_3d, tt, fs, sig_type="tensorpac", verbose=False
        )

        # Should extract first component
        # Assert
        assert sigs["orig"][0].shape == (2, 10, 1000)

    def test_filtering_parameters_len_bandpass_cols_is_1(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1000)
        # These are hardcoded in the example
        LOW_HZ = 20
        HIGH_HZ = 50
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Check column names contain correct frequencies
        # Act
        bandpass_cols = [c for c in sigs.columns if "bandpass" in c]
        # Act
        # Assert
        assert len(bandpass_cols) == 1

    def test_filtering_parameters_f_low_hz_in_bandpass_cols_0(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1000)
        # These are hardcoded in the example
        LOW_HZ = 20
        HIGH_HZ = 50
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Check column names contain correct frequencies
        # Act
        bandpass_cols = [c for c in sigs.columns if "bandpass" in c]
        # Act
        # Assert
        assert f"{LOW_HZ}" in bandpass_cols[0]

    def test_filtering_parameters_f_high_hz_in_bandpass_cols_0(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1000)
        # These are hardcoded in the example
        LOW_HZ = 20
        HIGH_HZ = 50
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Check column names contain correct frequencies
        # Act
        bandpass_cols = [c for c in sigs.columns if "bandpass" in c]
        # Act
        # Assert
        assert f"{HIGH_HZ}" in bandpass_cols[0]


    def test_verbose_output_index_in_captured_out_or_index_in_captured_out(self, capsys):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=0.5, fs=512)
        # With verbose=True
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=True
        )
        # Act
        captured = capsys.readouterr()
        # Act
        # Assert
        assert "Index" in captured.out or "index" in captured.out

    def test_verbose_output_len_captured_out_0(self, capsys):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=0.5, fs=512)
        # With verbose=True
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=True
        )
        # Act
        captured = capsys.readouterr()
        # Act
        # Assert
        assert len(captured.out) > 0



class TestPlotSignals:
    """Test plot_signals function."""

    def test_import_hasattr_scitex_dsp_example_plot_psd(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.example, "plot_signals")

    @pytest.fixture
    def sample_sigs(self):
        """Create sample signals DataFrame for testing."""
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        return scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

    def test_basic_plotting_fig_is_plt_figure(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Act
        # Assert
        assert isinstance(fig, plt.Figure)

    def test_basic_plotting_len_fig_axes_len_sample_sigs_columns(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Act
        # Assert
        assert len(fig.axes) == len(sample_sigs.columns)

    def test_basic_plotting_fig_suptitle_is_not_none(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Act
        # Assert
        assert fig._suptitle is not None

    def test_basic_plotting_fig_suptitle_get_text_chirp(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Act
        # Assert
        assert fig._suptitle.get_text() == "chirp"


    def test_axes_properties_calls_plot_signals(self, sample_sigs):
        """Test axes properties are set correctly."""
        # Arrange
        # Act
        # Assert
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "test_signal")

        # Check all axes
        for ax in fig.axes:
            # Should have legend
            assert ax.get_legend() is not None

            # Should have data
            assert len(ax.lines) > 0

            # Should have xlim set
            xlim = ax.get_xlim()
            assert xlim[0] < xlim[1]

    def test_hilbert_amp_overlay_hilbert_ax_is_not_none(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Find hilbert_amp axis
        hilbert_ax = None
        # Act
        for ax, col in zip(fig.axes, sample_sigs.columns):
            if col == "hilbert_amp":
                hilbert_ax = ax
                break
        # Act
        # Assert
        assert hilbert_ax is not None

    def test_hilbert_amp_overlay_len_hilbert_ax_lines_is_2(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Find hilbert_amp axis
        hilbert_ax = None
        # Act
        for ax, col in zip(fig.axes, sample_sigs.columns):
            if col == "hilbert_amp":
                hilbert_ax = ax
                break
        # Act
        # Assert
        assert len(hilbert_ax.lines) == 2


    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "chirp"])
    def test_different_signal_types(self, sig_type):
        """Test plotting with different signal types."""
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=0.5, fs=256, sig_type=sig_type)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type=sig_type, verbose=False
        )

        # Act
        fig = scitex.dsp.example.plot_signals(plt, sigs, sig_type)

        # Assert
        assert fig._suptitle.get_text() == sig_type
        plt.close(fig)


class TestPlotWavelet:
    """Test plot_wavelet function."""

    def test_import_hasattr_scitex_dsp_example_plot_psd(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.example, "plot_wavelet")

    @pytest.fixture
    def sample_sigs(self):
        """Create sample signals DataFrame for testing."""
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        return scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

    def test_basic_wavelet_plot_fig_is_plt_figure(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "orig", "chirp")
        # Act
        # Assert
        assert isinstance(fig, plt.Figure)

    def test_basic_wavelet_plot_len_fig_axes_is_2(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "orig", "chirp")
        # Act
        # Assert
        assert len(fig.axes) == 2


    def test_wavelet_plot_structure_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_wavelet_plot_structure_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_wavelet_plot_structure_len_ax1_images_0_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_wavelet_plot_structure_len_ax1_images_0_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_wavelet_plot_structure_len_ax1_images_0_len_ax1_images_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: spectrogram
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert len(ax1.images) > 0  # Should have imshow


    def test_wavelet_plot_structure_ax1_get_ylabel_frequency_hz_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_wavelet_plot_structure_ax1_get_ylabel_frequency_hz_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_wavelet_plot_structure_ax1_get_ylabel_frequency_hz_ax1_get_ylabel_frequency_hz(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: spectrogram
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert ax1.get_ylabel() == "Frequency [Hz]"


    def test_wavelet_plot_structure_ax1_yaxis_inverted_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_wavelet_plot_structure_ax1_yaxis_inverted_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_wavelet_plot_structure_ax1_yaxis_inverted_ax1_yaxis_inverted(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: spectrogram
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert ax1.yaxis_inverted()  # Should be inverted



    @pytest.mark.parametrize(
        "sig_col", ["orig", "z_normed", "bandpass_filted (20 - 50 Hz)"]
    )
    def test_different_signal_columns(self, sample_sigs, sig_col):
        """Test wavelet plot with different signal columns."""
        # Arrange
        if sig_col not in sample_sigs.columns:
            # Find a bandpass column
            for col in sample_sigs.columns:
                if "bandpass" in col:
                    sig_col = col
                    break

        # Act
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, sig_col, "test")

        # Assert
        assert fig is not None
        plt.close(fig)


class TestPlotPSD:
    """Test plot_psd function."""

    def test_import_hasattr_scitex_dsp_example_plot_psd(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.example, "plot_psd")

    @pytest.fixture
    def sample_sigs(self):
        """Create sample signals DataFrame for testing."""
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        return scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

    def test_basic_psd_plot_fig_is_plt_figure(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "orig", "chirp")
        # Act
        # Assert
        assert isinstance(fig, plt.Figure)

    def test_basic_psd_plot_len_fig_axes_is_2(self, sample_sigs):
        # Arrange
        # Act
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "orig", "chirp")
        # Act
        # Assert
        assert len(fig.axes) == 2


    def test_psd_plot_structure_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_psd_plot_structure_ax0_get_xlabel_time_s(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_xlabel() == "Time [s]"

    def test_psd_plot_structure_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_psd_plot_structure_len_ax1_lines_0_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_psd_plot_structure_len_ax1_lines_0_ax0_get_xlabel_time_s(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_xlabel() == "Time [s]"

    def test_psd_plot_structure_len_ax1_lines_0_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_psd_plot_structure_len_ax1_lines_0_len_ax1_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_xlabel() == "Time [s]"
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: PSD
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert len(ax1.lines) > 0


    def test_psd_plot_structure_ax1_get_xlabel_frequency_hz_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_psd_plot_structure_ax1_get_xlabel_frequency_hz_ax0_get_xlabel_time_s(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_xlabel() == "Time [s]"

    def test_psd_plot_structure_ax1_get_xlabel_frequency_hz_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_psd_plot_structure_ax1_get_xlabel_frequency_hz_ax1_get_xlabel_frequency_hz(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_xlabel() == "Time [s]"
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: PSD
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert ax1.get_xlabel() == "Frequency [Hz]"


    def test_psd_plot_structure_ax1_get_ylabel_power_uv_2_hz_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_psd_plot_structure_ax1_get_ylabel_power_uv_2_hz_ax0_get_xlabel_time_s(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_xlabel() == "Time [s]"

    def test_psd_plot_structure_ax1_get_ylabel_power_uv_2_hz_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_psd_plot_structure_ax1_get_ylabel_power_uv_2_hz_ax1_get_ylabel_power_uv_2_hz(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_xlabel() == "Time [s]"
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: PSD
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert ax1.get_ylabel() == "Power [uV^2 / Hz]"


    def test_psd_plot_structure_ax1_get_yscale_log_len_ax0_lines_0(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert len(ax0.lines) > 0

    def test_psd_plot_structure_ax1_get_yscale_log_ax0_get_xlabel_time_s(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_xlabel() == "Time [s]"

    def test_psd_plot_structure_ax1_get_yscale_log_ax0_get_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Act
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_psd_plot_structure_ax1_get_yscale_log_ax1_get_yscale_log(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # First axis: signal
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0
        assert ax0.get_xlabel() == "Time [s]"
        assert ax0.get_ylabel() == "Voltage"
        # Second axis: PSD
        ax1 = fig.axes[1]
        # Act
        # Assert
        assert ax1.get_yscale() == "log"  # Should be log scale



    def test_psd_with_filtered_signal_filtered_col_is_not_none(self, sample_sigs):
        # Arrange
        filtered_col = None
        # Act
        for col in sample_sigs.columns:
            if "filted" in col:
                filtered_col = col
                break
        # Act
        # Assert
        assert filtered_col is not None

    def test_psd_with_filtered_signal_fig_suptitle_get_text_filtered_filtered_col_is_not_none(self, sample_sigs):
        # Arrange
        filtered_col = None
        # Act
        for col in sample_sigs.columns:
            if "filted" in col:
                filtered_col = col
                break
        # Act
        # Assert
        assert filtered_col is not None

    def test_psd_with_filtered_signal_fig_suptitle_get_text_filtered_fig_suptitle_get_text_filtered(self, sample_sigs):
        # Arrange
        filtered_col = None
        # Act
        for col in sample_sigs.columns:
            if "filted" in col:
                filtered_col = col
                break
        # Assert
        assert filtered_col is not None
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, filtered_col, "filtered")
        # Act
        # Assert
        assert fig._suptitle.get_text() == "filtered"




class TestExampleIntegration:
    """Test the full example workflow."""

    def test_full_workflow_calls_use(self, tmp_path):
        """Test the complete example workflow."""
        # Set up parameters
        # Arrange
        # Act
        # Assert
        T_SEC = 0.5  # Short for testing
        SIG_TYPES = ["chirp", "gauss"]
        SRC_FS = 512

        # Configure matplotlib
        plt.style.use("default")
        CC = {"blue": "blue", "red": "red"}  # Simple color config

        # Run workflow
        for sig_type in SIG_TYPES:
            # Generate signal
            xx, tt, fs = scitex.dsp.demo_sig(t_sec=T_SEC, fs=SRC_FS, sig_type=sig_type)

            # Process signal
            sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
                xx, tt, fs, sig_type, verbose=False
            )

            # Test plotting functions
            fig1 = scitex.dsp.example.plot_signals(plt, sigs, sig_type)
            assert fig1 is not None
            plt.close(fig1)

            # Test wavelet and PSD for one column
            fig2 = scitex.dsp.example.plot_wavelet(plt, sigs, "orig", sig_type)
            assert fig2 is not None
            plt.close(fig2)

            fig3 = scitex.dsp.example.plot_psd(plt, sigs, "orig", sig_type)
            assert fig3 is not None
            plt.close(fig3)

    def test_parameter_dependencies_sigs_resampled_2_tgt_fs(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        # Assert
        assert sigs["resampled"][2] == TGT_FS

    def test_parameter_dependencies_str_low_hz_in_bandpass_col_sigs_resampled_2_tgt_fs(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        # Assert
        assert sigs["resampled"][2] == TGT_FS

    def test_parameter_dependencies_str_low_hz_in_bandpass_col_str_low_hz_in_bandpass_col(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Check resampling target
        # Assert
        assert sigs["resampled"][2] == TGT_FS
        # Check filter parameters in column names
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        # Act
        # Assert
        assert str(LOW_HZ) in bandpass_col


    def test_parameter_dependencies_str_high_hz_in_bandpass_col_sigs_resampled_2_tgt_fs(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        # Assert
        assert sigs["resampled"][2] == TGT_FS

    def test_parameter_dependencies_str_high_hz_in_bandpass_col_str_high_hz_in_bandpass_col(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Check resampling target
        # Assert
        assert sigs["resampled"][2] == TGT_FS
        # Check filter parameters in column names
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        # Act
        # Assert
        assert str(HIGH_HZ) in bandpass_col


    def test_parameter_dependencies_str_sigma_in_gauss_col_sigs_resampled_2_tgt_fs(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        # Assert
        assert sigs["resampled"][2] == TGT_FS

    def test_parameter_dependencies_str_sigma_in_gauss_col_str_low_hz_in_bandpass_col(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Check resampling target
        # Assert
        assert sigs["resampled"][2] == TGT_FS
        # Check filter parameters in column names
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        # Act
        # Assert
        assert str(LOW_HZ) in bandpass_col

    def test_parameter_dependencies_str_sigma_in_gauss_col_str_high_hz_in_bandpass_col(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Check resampling target
        # Assert
        assert sigs["resampled"][2] == TGT_FS
        # Check filter parameters in column names
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        # Act
        # Assert
        assert str(HIGH_HZ) in bandpass_col

    def test_parameter_dependencies_str_sigma_in_gauss_col_str_sigma_in_gauss_col(self):
        # Arrange
        LOW_HZ = 20
        HIGH_HZ = 50
        SIGMA = 10
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Check resampling target
        # Assert
        assert sigs["resampled"][2] == TGT_FS
        # Check filter parameters in column names
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        assert str(LOW_HZ) in bandpass_col
        assert str(HIGH_HZ) in bandpass_col
        # Check gaussian filter sigma
        gauss_col = [c for c in sigs.columns if "gauss" in c and "sigma" in c][0]
        # Act
        # Assert
        assert str(SIGMA) in gauss_col




if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/example.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-04-06 01:36:18 (ywatanabe)"
#
# import matplotlib
#
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import pandas as pd
# import scitex
#
# import scitex
#
# # Module-level constants (defaults for example functions)
# TGT_FS = 512
# LOW_HZ = 20
# HIGH_HZ = 50
# SIGMA = 10
#
# # Default color cycle
# CC = {"blue": "#1f77b4", "red": "#d62728", "green": "#2ca02c"}
#
#
# # Functions
# def calc_norm_resample_filt_hilbert(xx, tt, fs, sig_type, verbose=True):
#     sigs = {"index": ("signal", "time", "fs")}  # Collector
#
#     if sig_type == "tensorpac":
#         xx = xx[:, :, 0]
#
#     sigs["orig"] = (xx, tt, fs)
#
#     # Normalization
#     sigs["z_normed"] = (scitex.dsp.norm.z(xx), tt, fs)
#     sigs["minmax_normed"] = (scitex.dsp.norm.minmax(xx), tt, fs)
#
#     # Resampling
#     resampled_xx = scitex.dsp.resample(xx, fs, TGT_FS)
#     # Create proper time vector for resampled signal
#     import numpy as np
#
#     resampled_tt = np.linspace(tt[0], tt[-1], resampled_xx.shape[-1])
#     sigs["resampled"] = (resampled_xx, resampled_tt, TGT_FS)
#
#     # Noise injection
#     sigs["gaussian_noise_added"] = (scitex.dsp.add_noise.gauss(xx), tt, fs)
#     sigs["white_noise_added"] = (scitex.dsp.add_noise.white(xx), tt, fs)
#     sigs["pink_noise_added"] = (scitex.dsp.add_noise.pink(xx), tt, fs)
#     sigs["brown_noise_added"] = (scitex.dsp.add_noise.brown(xx), tt, fs)
#
#     # Filtering (bands format is [[low_hz, high_hz]])
#     bands = [[LOW_HZ, HIGH_HZ]]
#     sigs[f"bandpass_filted ({LOW_HZ} - {HIGH_HZ} Hz)"] = (
#         scitex.dsp.filt.bandpass(xx, fs, bands),
#         tt,
#         fs,
#     )
#
#     sigs[f"bandstop_filted ({LOW_HZ} - {HIGH_HZ} Hz)"] = (
#         scitex.dsp.filt.bandstop(xx, fs, bands),
#         tt,
#         fs,
#     )
#     sigs[f"bandstop_gauss (sigma = {SIGMA})"] = (
#         scitex.dsp.filt.gauss(xx, sigma=SIGMA),
#         tt,
#         fs,
#     )
#
#     # Hilbert Transformation
#     pha, amp = scitex.dsp.hilbert(xx)
#     sigs["hilbert_amp"] = (amp, tt, fs)
#     sigs["hilbert_pha"] = (pha, tt, fs)
#
#     sigs = pd.DataFrame(sigs).set_index("index")
#
#     if verbose:
#         print(sigs.index)
#         print(sigs.columns)
#
#     return sigs
#
#
# def plot_signals(plt, sigs, sig_type):
#     fig, axes = plt.subplots(nrows=len(sigs.columns), sharex=True)
#
#     i_batch = 0
#     i_ch = 0
#     for ax, (i_col, col) in zip(axes, enumerate(sigs.columns)):
#         if col == "hilbert_amp":  # add the original signal to the ax
#             _col = "orig"
#             (
#                 _xx,
#                 _tt,
#                 _fs,
#             ) = sigs[_col]
#             ax.plot(_tt, _xx[i_batch, i_ch], label=_col, c=CC["blue"])
#
#         # Main
#         xx, tt, fs = sigs[col]
#         # if sig_type == "tensorpac":
#         #     xx = xx[:, :, 0]
#
#         # Handle potential shape mismatches from filter operations
#         signal = xx[i_batch, i_ch]
#         if hasattr(signal, "squeeze"):
#             signal = signal.squeeze()
#         if hasattr(signal, "numpy"):
#             signal = signal.numpy()
#
#         ax.plot(
#             tt,
#             signal,
#             label=col,
#             c=CC["red"] if col == "hilbert_amp" else CC["blue"],
#         )
#
#         # Adjustments
#         ax.legend(loc="upper left")
#         ax.set_xlim(tt[0], tt[-1])
#
#         ax = scitex.plt.ax.set_n_ticks(ax)
#
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Voltage")
#     fig.suptitle(sig_type)
#     return fig
#
#
# def plot_wavelet(plt, sigs, sig_col, sig_type):
#     xx, tt, fs = sigs[sig_col]
#     # if sig_type == "tensorpac":
#     #     xx = xx[:, :, 0]
#
#     # Wavelet Transformation
#     wavelet_coef, ff_ww = scitex.dsp.wavelet(xx, fs)
#
#     i_batch = 0
#     i_ch = 0
#
#     # Main
#     fig, axes = plt.subplots(nrows=2, sharex=True)
#     # Signal
#     axes[0].plot(
#         tt,
#         xx[i_batch, i_ch],
#         label=sig_col,
#         c=CC["blue"],
#     )
#     # Adjusts
#     axes[0].legend(loc="upper left")
#     axes[0].set_xlim(tt[0], tt[-1])
#     axes[0].set_ylabel("Voltage")
#     axes[0] = scitex.plt.ax.set_n_ticks(axes[0])
#
#     # Wavelet Spectrogram
#     axes[1].imshow(
#         wavelet_coef[i_batch, i_ch],
#         aspect="auto",
#         extent=[tt[0], tt[-1], 512, 1],
#         label="wavelet_coefficient",
#     )
#     # axes[1].set_xlabel("Time [s]")
#     axes[1].set_ylabel("Frequency [Hz]")
#     # axes[1].legend(loc="upper left")
#     axes[1].invert_yaxis()
#
#     fig.supxlabel("Time [s]")
#     fig.suptitle(sig_type)
#
#     return fig
#
#
# def plot_psd(plt, sigs, sig_col, sig_type):
#     xx, tt, fs = sigs[sig_col]
#
#     # if sig_type == "tensorpac":
#     #     xx = xx[:, :, 0]
#
#     # Power Spetrum Density
#     psd, ff_pp = scitex.dsp.psd(xx, fs)
#
#     # Main
#     i_batch = 0
#     i_ch = 0
#     fig, axes = plt.subplots(nrows=2, sharex=False)
#
#     # Signal
#     axes[0].plot(
#         tt,
#         xx[i_batch, i_ch],
#         label=sig_col,
#         c=CC["blue"],
#     )
#     # Adjustments
#     axes[0].legend(loc="upper left")
#     axes[0].set_xlim(tt[0], tt[-1])
#     axes[0].set_xlabel("Time [s]")
#     axes[0].set_ylabel("Voltage")
#     axes[0] = scitex.plt.ax.set_n_ticks(axes[0])
#
#     # PSD
#     axes[1].plot(ff_pp, psd[i_batch, i_ch], label="PSD")
#     axes[1].set_yscale("log")
#     axes[1].set_ylabel("Power [uV^2 / Hz]")
#     axes[1].set_xlabel("Frequency [Hz]")
#
#     fig.suptitle(sig_type)
#
#     return fig
#
#
# if __name__ == "__main__":
#     # Parameters
#     T_SEC = 4
#     SIG_TYPES = [
#         # "uniform",
#         # "gauss",
#         # "periodic",
#         # "chirp",
#         # "ripple",
#         # "meg",
#         "tensorpac",
#     ]
#     SRC_FS = 1024
#     TGT_FS = 512
#     FREQS_HZ = [10, 30, 100]
#     LOW_HZ = 20
#     HIGH_HZ = 50
#     SIGMA = 10
#
#     plt, CC = scitex.plt.configure_mpl(plt, fig_scale=10)
#     sdir = "/home/ywatanabe/proj/entrance/scitex/dsp/example/"
#
#     for sig_type in SIG_TYPES:
#         # Demo Signal
#         xx, tt, fs = scitex.dsp.demo_sig(
#             t_sec=T_SEC, fs=SRC_FS, freqs_hz=FREQS_HZ, sig_type=sig_type
#         )
#
#         # Apply calculations on the original signal
#         sigs = calc_norm_resample_filt_hilbert(xx, tt, fs, sig_type)
#
#         # Plots signals
#         fig = plot_signals(plt, sigs, sig_type)
#         scitex.io.save(fig, sdir + f"{sig_type}/1_signals.png")
#
#         # Plots wavelet coefficients and PSD
#         for sig_col in sigs.columns:
#             if "hilbert" in sig_col:
#                 continue
#
#             fig = plot_wavelet(plt, sigs, sig_col, sig_type)
#             scitex.io.save(fig, sdir + f"{sig_type}/2_wavelet_{sig_col}.png")
#
#             fig = plot_psd(plt, sigs, sig_col, sig_type)
#             scitex.io.save(fig, sdir + f"{sig_type}/3_psd_{sig_col}.png")
#
#     # plt.show()
#
#     """
#     python ./dsp/example.py
#     """

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/example.py
# --------------------------------------------------------------------------------

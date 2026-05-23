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

    def test_import_hasattr_scitex_dsp_example_calc_norm_resample_filt_hilbert(self):
        # Arrange
        # Act
        present = hasattr(scitex.dsp.example, "calc_norm_resample_filt_hilbert")
        # Assert
        assert present

    def test_basic_functionality_sigs_is_pd_dataframe(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=2, fs=1000, sig_type="chirp")
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Assert
        assert isinstance(sigs, pd.DataFrame)

    def test_basic_functionality_sigs_index_name_index(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=2, fs=1000, sig_type="chirp")
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Assert
        assert sigs.index.name == "index"

    def test_basic_functionality_len_sigs_columns_greater_than_10(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=2, fs=1000, sig_type="chirp")
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Assert
        assert len(sigs.columns) > 10  # Should have multiple processing steps

    def test_output_columns_all_expected_present(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
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
        # Act
        all_present = all(any(col in c for c in sigs.columns) for col in expected_cols)
        # Assert
        assert all_present

    @pytest.mark.parametrize(
        "col", ["z_normed", "minmax_normed", "hilbert_amp", "hilbert_pha"]
    )
    def test_signal_shapes_preserved(self, col):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        orig_shape = sigs["orig"][0].shape
        # Act
        sig, _, _ = sigs[col]
        # Assert
        assert sig.shape == orig_shape

    def test_resampling_shape_resampled_sig_shape_matches_expected(self):
        # Arrange
        src_fs = 1024
        tgt_fs = 512  # Default target in example
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=src_fs)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        resampled_sig, resampled_tt, resampled_fs = sigs["resampled"]
        expected_samples = xx.shape[-1] // (src_fs // tgt_fs)
        # Act
        actual = resampled_sig.shape[-1]
        # Assert
        assert actual == expected_samples

    def test_resampling_shape_len_resampled_tt_matches_expected(self):
        # Arrange
        src_fs = 1024
        tgt_fs = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=src_fs)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        resampled_sig, resampled_tt, resampled_fs = sigs["resampled"]
        expected_samples = xx.shape[-1] // (src_fs // tgt_fs)
        # Act
        actual = len(resampled_tt)
        # Assert
        assert actual == expected_samples

    def test_resampling_shape_resampled_fs_equals_tgt_fs(self):
        # Arrange
        src_fs = 1024
        tgt_fs = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=src_fs)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        resampled_sig, resampled_tt, resampled_fs = sigs["resampled"]
        # Assert
        assert resampled_fs == tgt_fs

    def test_tensorpac_signal_sigs_orig_0_shape_2_10_1000(self):
        # Arrange
        xx_3d = np.random.randn(2, 10, 1000, 2)  # Extra dimension
        tt = np.linspace(0, 1, 1000)
        fs = 1000
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx_3d, tt, fs, sig_type="tensorpac", verbose=False
        )
        # Assert
        assert sigs["orig"][0].shape == (2, 10, 1000)

    def test_filtering_parameters_len_bandpass_cols_is_1(self):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1000)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        bandpass_cols = [c for c in sigs.columns if "bandpass" in c]
        # Assert
        assert len(bandpass_cols) == 1

    def test_filtering_parameters_low_hz_in_bandpass_col(self):
        # Arrange
        LOW_HZ = 20
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1000)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        bandpass_cols = [c for c in sigs.columns if "bandpass" in c]
        # Assert
        assert f"{LOW_HZ}" in bandpass_cols[0]

    def test_filtering_parameters_high_hz_in_bandpass_col(self):
        # Arrange
        HIGH_HZ = 50
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1000)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )
        # Act
        bandpass_cols = [c for c in sigs.columns if "bandpass" in c]
        # Assert
        assert f"{HIGH_HZ}" in bandpass_cols[0]

    def test_verbose_output_index_in_captured_out(self, capsys):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=0.5, fs=512)
        scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=True
        )
        # Act
        captured = capsys.readouterr()
        # Assert
        assert "Index" in captured.out or "index" in captured.out

    def test_verbose_output_len_captured_out_greater_than_0(self, capsys):
        # Arrange
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=0.5, fs=512)
        scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=True
        )
        # Act
        captured = capsys.readouterr()
        # Assert
        assert len(captured.out) > 0


class TestPlotSignals:
    """Test plot_signals function."""

    @pytest.fixture
    def sample_sigs(self):
        """Create sample signals DataFrame for testing."""
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        return scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

    def test_import_hasattr_scitex_dsp_example_plot_signals(self):
        # Arrange
        # Act
        present = hasattr(scitex.dsp.example, "plot_signals")
        # Assert
        assert present

    def test_basic_plotting_fig_is_plt_figure(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Assert
        assert isinstance(fig, plt.Figure)

    def test_basic_plotting_len_fig_axes_len_sample_sigs_columns(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Assert
        assert len(fig.axes) == len(sample_sigs.columns)

    def test_basic_plotting_fig_suptitle_is_not_none(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Assert
        assert fig._suptitle is not None

    def test_basic_plotting_fig_suptitle_get_text_chirp(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Assert
        assert fig._suptitle.get_text() == "chirp"

    def test_axes_properties_all_have_legend(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "test_signal")
        # Act
        all_have_legend = all(ax.get_legend() is not None for ax in fig.axes)
        # Assert
        assert all_have_legend

    def test_axes_properties_all_have_lines(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "test_signal")
        # Act
        all_have_lines = all(len(ax.lines) > 0 for ax in fig.axes)
        # Assert
        assert all_have_lines

    def test_axes_properties_xlim_set(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "test_signal")
        # Act
        xlims = [ax.get_xlim() for ax in fig.axes]
        # Assert
        assert all(xlim[0] < xlim[1] for xlim in xlims)

    def test_hilbert_amp_overlay_hilbert_ax_is_not_none(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        # Act
        hilbert_ax = None
        for ax, col in zip(fig.axes, sample_sigs.columns):
            if col == "hilbert_amp":
                hilbert_ax = ax
                break
        # Assert
        assert hilbert_ax is not None

    def test_hilbert_amp_overlay_len_hilbert_ax_lines_is_2(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_signals(plt, sample_sigs, "chirp")
        hilbert_ax = None
        for ax, col in zip(fig.axes, sample_sigs.columns):
            if col == "hilbert_amp":
                hilbert_ax = ax
                break
        # Act
        n_lines = len(hilbert_ax.lines)
        # Assert
        assert n_lines == 2

    @pytest.mark.parametrize("sig_type", ["uniform", "gauss", "chirp"])
    def test_different_signal_types(self, sig_type):
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

    @pytest.fixture
    def sample_sigs(self):
        """Create sample signals DataFrame for testing."""
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        return scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

    def test_import_hasattr_scitex_dsp_example_plot_wavelet(self):
        # Arrange
        # Act
        present = hasattr(scitex.dsp.example, "plot_wavelet")
        # Assert
        assert present

    def test_basic_wavelet_plot_fig_is_plt_figure(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "orig", "chirp")
        # Assert
        assert isinstance(fig, plt.Figure)

    def test_basic_wavelet_plot_len_fig_axes_is_2(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "orig", "chirp")
        # Assert
        assert len(fig.axes) == 2

    def test_wavelet_plot_structure_ax0_has_lines(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0

    def test_wavelet_plot_structure_ax0_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_wavelet_plot_structure_ax1_has_images(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert len(ax1.images) > 0  # Should have imshow

    def test_wavelet_plot_structure_ax1_ylabel_frequency_hz(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert ax1.get_ylabel() == "Frequency [Hz]"

    def test_wavelet_plot_structure_ax1_yaxis_inverted(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_wavelet(plt, sample_sigs, "z_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert ax1.yaxis_inverted()  # Should be inverted

    @pytest.mark.parametrize(
        "sig_col", ["orig", "z_normed", "bandpass_filted (20 - 50 Hz)"]
    )
    def test_different_signal_columns_returns_fig(self, sample_sigs, sig_col):
        # Arrange
        if sig_col not in sample_sigs.columns:
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

    @pytest.fixture
    def sample_sigs(self):
        """Create sample signals DataFrame for testing."""
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=512)
        return scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type="chirp", verbose=False
        )

    def test_import_hasattr_scitex_dsp_example_plot_psd(self):
        # Arrange
        # Act
        present = hasattr(scitex.dsp.example, "plot_psd")
        # Assert
        assert present

    def test_basic_psd_plot_fig_is_plt_figure(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "orig", "chirp")
        # Assert
        assert isinstance(fig, plt.Figure)

    def test_basic_psd_plot_len_fig_axes_is_2(self, sample_sigs):
        # Arrange
        # Act
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "orig", "chirp")
        # Assert
        assert len(fig.axes) == 2

    def test_psd_plot_structure_ax0_has_lines(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert len(ax0.lines) > 0

    def test_psd_plot_structure_ax0_xlabel_time_s(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert ax0.get_xlabel() == "Time [s]"

    def test_psd_plot_structure_ax0_ylabel_voltage(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax0 = fig.axes[0]
        # Assert
        assert ax0.get_ylabel() == "Voltage"

    def test_psd_plot_structure_ax1_has_lines(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert len(ax1.lines) > 0

    def test_psd_plot_structure_ax1_xlabel_frequency_hz(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert ax1.get_xlabel() == "Frequency [Hz]"

    def test_psd_plot_structure_ax1_ylabel_power_uv2_hz(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert ax1.get_ylabel() == "Power [uV^2 / Hz]"

    def test_psd_plot_structure_ax1_yscale_log(self, sample_sigs):
        # Arrange
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, "minmax_normed", "test")
        # Act
        ax1 = fig.axes[1]
        # Assert
        assert ax1.get_yscale() == "log"

    def test_psd_with_filtered_signal_filtered_col_is_not_none(self, sample_sigs):
        # Arrange
        # Act
        filtered_col = None
        for col in sample_sigs.columns:
            if "filted" in col:
                filtered_col = col
                break
        # Assert
        assert filtered_col is not None

    def test_psd_with_filtered_signal_fig_suptitle_get_text_filtered(self, sample_sigs):
        # Arrange
        filtered_col = None
        for col in sample_sigs.columns:
            if "filted" in col:
                filtered_col = col
                break
        # Act
        fig = scitex.dsp.example.plot_psd(plt, sample_sigs, filtered_col, "filtered")
        # Assert
        assert fig._suptitle.get_text() == "filtered"


class TestExampleIntegration:
    """Test the full example workflow."""

    @pytest.mark.parametrize("sig_type", ["chirp", "gauss"])
    def test_full_workflow_signals_fig_is_not_none(self, sig_type):
        # Arrange
        T_SEC = 0.5
        SRC_FS = 512
        plt.style.use("default")
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=T_SEC, fs=SRC_FS, sig_type=sig_type)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type, verbose=False
        )
        # Act
        fig1 = scitex.dsp.example.plot_signals(plt, sigs, sig_type)
        # Assert
        assert fig1 is not None
        plt.close(fig1)

    @pytest.mark.parametrize("sig_type", ["chirp", "gauss"])
    def test_full_workflow_wavelet_fig_is_not_none(self, sig_type):
        # Arrange
        T_SEC = 0.5
        SRC_FS = 512
        plt.style.use("default")
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=T_SEC, fs=SRC_FS, sig_type=sig_type)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type, verbose=False
        )
        # Act
        fig2 = scitex.dsp.example.plot_wavelet(plt, sigs, "orig", sig_type)
        # Assert
        assert fig2 is not None
        plt.close(fig2)

    @pytest.mark.parametrize("sig_type", ["chirp", "gauss"])
    def test_full_workflow_psd_fig_is_not_none(self, sig_type):
        # Arrange
        T_SEC = 0.5
        SRC_FS = 512
        plt.style.use("default")
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=T_SEC, fs=SRC_FS, sig_type=sig_type)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, sig_type, verbose=False
        )
        # Act
        fig3 = scitex.dsp.example.plot_psd(plt, sigs, "orig", sig_type)
        # Assert
        assert fig3 is not None
        plt.close(fig3)

    def test_parameter_dependencies_sigs_resampled_2_tgt_fs(self):
        # Arrange
        TGT_FS = 512
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        # Act
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Assert
        assert sigs["resampled"][2] == TGT_FS

    def test_parameter_dependencies_low_hz_in_bandpass_col(self):
        # Arrange
        LOW_HZ = 20
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        # Assert
        assert str(LOW_HZ) in bandpass_col

    def test_parameter_dependencies_high_hz_in_bandpass_col(self):
        # Arrange
        HIGH_HZ = 50
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        bandpass_col = [c for c in sigs.columns if "bandpass" in c][0]
        # Assert
        assert str(HIGH_HZ) in bandpass_col

    def test_parameter_dependencies_sigma_in_gauss_col(self):
        # Arrange
        SIGMA = 10
        xx, tt, fs = scitex.dsp.demo_sig(t_sec=1, fs=1024)
        sigs = scitex.dsp.example.calc_norm_resample_filt_hilbert(
            xx, tt, fs, "chirp", verbose=False
        )
        # Act
        gauss_col = [c for c in sigs.columns if "gauss" in c and "sigma" in c][0]
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

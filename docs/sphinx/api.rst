API
===

.. automodule:: scitex_dsp
   :members:
   :undoc-members:
   :show-inheritance:

Spectral Primitives
-------------------

.. autofunction:: scitex_dsp.hilbert
   :no-index:

.. autofunction:: scitex_dsp.psd
   :no-index:

.. autofunction:: scitex_dsp.band_powers
   :no-index:

.. autofunction:: scitex_dsp.pac
   :no-index:

.. autofunction:: scitex_dsp.wavelet
   :no-index:

.. autofunction:: scitex_dsp.modulation_index
   :no-index:

Feature Extraction
------------------

.. autofunction:: scitex_dsp.pac_features
   :no-index:

.. autofunction:: scitex_dsp.feature_registry
   :no-index:

.. py:data:: scitex_dsp.PAC_FEATURE_REGISTRY

   Self-documenting registry (a ``dict``) mapping each PAC-summary feature
   name to its provenance record — family, computation engine,
   interpretation, and provisional flags. Backs ``feature_registry()``.

.. autofunction:: scitex_dsp.extract_all
   :no-index:

.. autofunction:: scitex_dsp.extract_all_registry
   :no-index:

.. py:data:: scitex_dsp.AVAILABLE_BACKENDS

   Tuple of the feature-extraction backends available to ``extract_all()``
   (``("pac", "catch22")``).

.. autofunction:: scitex_dsp.feature_correlation
   :no-index:

.. autofunction:: scitex_dsp.redundancy_summary
   :no-index:

.. autofunction:: scitex_dsp.pca_loadings
   :no-index:

.. autofunction:: scitex_dsp.correlation_by_group
   :no-index:

Pre-/Post-processing
--------------------

.. autofunction:: scitex_dsp.crop
   :no-index:

.. autofunction:: scitex_dsp.ensure_3d
   :no-index:

.. autofunction:: scitex_dsp.resample
   :no-index:

.. autofunction:: scitex_dsp.time
   :no-index:

.. autofunction:: scitex_dsp.to_segments
   :no-index:

.. autofunction:: scitex_dsp.to_sktime_df
   :no-index:

Ripple Detection
----------------

.. autofunction:: scitex_dsp.detect_ripples
   :no-index:

Signal Synthesis
----------------

.. autofunction:: scitex_dsp.demo_sig
   :no-index:

Noise synthesis (``scitex_dsp.add_noise``):

.. autofunction:: scitex_dsp.add_noise.gauss
   :no-index:

.. autofunction:: scitex_dsp.add_noise.white
   :no-index:

.. autofunction:: scitex_dsp.add_noise.pink
   :no-index:

.. autofunction:: scitex_dsp.add_noise.brown
   :no-index:

Audio / EEG I/O
---------------

.. autofunction:: scitex_dsp.list_and_select_device
   :no-index:

.. autofunction:: scitex_dsp.get_eeg_pos
   :no-index:

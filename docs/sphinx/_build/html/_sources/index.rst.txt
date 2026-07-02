scitex-dsp
=============

Digital signal-processing primitives for neuroscience — Hilbert analytic
signal, power spectral density / band powers, phase-amplitude coupling and
the modulation index, continuous wavelet transform, Buzsaki-style ripple
detection, deterministic demo signals, and pre-/post-processing utilities
(crop, resample, noise synthesis, segmentation). Extracted from the SciTeX
ecosystem as a standalone package.

.. code-block:: python

   import scitex_dsp as dsp

   xx, tt, fs = dsp.demo_sig(sig_type="chirp", fs=1024)
   ana = dsp.hilbert(xx)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

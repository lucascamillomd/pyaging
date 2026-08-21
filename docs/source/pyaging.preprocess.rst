Preprocess
==========

Please note that most functions are helper functions and are not meant to be used directly.

Cohort-relative transcriptomic clocks
-------------------------------------

The tAge clocks (``tage`` and ``tagemortality``) do not read a sample in isolation.
Every preprocessing stage — RLE normalisation, per-sample scaling, and centring on a
reference group — is computed across the whole cohort, so a prediction is an age or
hazard *difference* against the reference rather than an absolute value, and a single
sample cannot be prepared on its own. Because of this, these clocks refuse any input
that :func:`pyaging.preprocess.prepare_tage` did not produce; passing a raw matrix to
``predict_age`` raises rather than silently returning a meaningless number.

The flip side of cohort statistics is that they need a cohort. Two samples pass the
minimum ``prepare_tage`` enforces, but normalisation and centring estimated from fewer
than roughly ten samples — or from a reference group that small — are statistically
weak, and the predictions inherit that noise.

.. code-block:: python

   import pyaging as pya

   adata = pya.pp.prepare_tage(counts, species="mouse", reference_group=controls)
   pya.pred.predict_age(adata, ["tage", "tagemortality"])

``counts`` is a samples-by-genes AnnData of raw RNA-seq counts, ``species`` is one of
``pyaging.preprocess.TAGE_SPECIES``, and ``reference_group`` names the samples to
centre against — omit it to centre on every sample. ``tage`` is reported in months of
mouse age, so cohorts of other species need rescaling by their own maximum lifespan;
``tagemortality`` is a base-10 log hazard ratio and is never rescaled. Both clocks are
released for non-commercial academic research only.

pyaging.preprocess._preprocess
------------------------------

.. automodule:: pyaging.preprocess._preprocess
   :members:
   :undoc-members:
   :show-inheritance:

pyaging.preprocess._preprocess_utils
------------------------------------

.. automodule:: pyaging.preprocess._preprocess_utils
   :members:
   :undoc-members:
   :show-inheritance:

pyaging.preprocess._tage
------------------------

.. automodule:: pyaging.preprocess._tage
   :members:
   :undoc-members:
   :show-inheritance:

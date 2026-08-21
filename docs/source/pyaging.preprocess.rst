Preprocess
==========

Please note that most functions are helper functions and are not meant to be used directly.

.. _cohort-relative-transcriptomic-clocks:

Cohort-relative transcriptomic clocks
-------------------------------------

The tAge clocks (``tage`` and ``tagemortality``) do not read a sample in isolation.
Every preprocessing stage — gene filtering, mapping to mouse Entrez IDs, RLE
normalisation, per-sample scaling, and centring on a reference group — is computed
across the whole cohort, so a prediction is an age or hazard *difference* against the
reference rather than an absolute value, and a single sample cannot be scored on its
own.

There is no preprocessing call to make: ``predict_age`` runs the whole cohort
pipeline itself, once per call however many of these clocks are requested, and reads
its two settings off the input. Pass raw RNA-seq counts.

.. code-block:: python

   import pyaging as pya

   pya.pred.predict_age(counts, ["tage", "tagemortality"])

``counts`` is a samples-by-genes AnnData of raw counts whose ``var_names`` are gene
identifiers (symbol, Ensembl, or Entrez) of the cohort's species. The raw matrix is
never modified; the transformed values live only inside the prediction, and what the
run did is recorded in ``adata.uns["tage_preparation"]`` (species, genes filtered and
mapped, reference-group size).

**Species.** Add a column named ``mouse``, ``rat``, ``macaque``, or ``human`` (the
names in ``pyaging.preprocess.TAGE_SPECIES``) to the matrix, set to ``1`` for every
sample — the same idiom the mammalian clocks use for covariates such as ``female``.
The name is matched case-insensitively, so ``Human`` counts. Exactly one may be set.
The column is dropped before the gene pipeline, so it is never mistaken for a gene.
With no such column — or with the columns present but zero everywhere — the cohort is
taken to be mouse and a warning says so; setting two, or letting one vary between
samples, is an error. That warning is the only signal, and ``verbose=False`` suppresses
it, so name the species explicitly in scripted runs.

**Reference group.** Add ``adata.obs["tage_reference_group"]``, boolean or numeric
``0``/``1``; the truthy rows are the samples to centre against. Without the column the
cohort centres on every sample, which is the reference pipeline's own default. A column
that selects nothing is an error rather than a silent fallback.

The flip side of cohort statistics is that they need a cohort. Two samples pass the
hard minimum, but normalisation and centring estimated from fewer than roughly ten
samples — or from a reference group that small — are statistically weak, and the
predictions inherit that noise.

``tage`` is reported in months of mouse age, so cohorts of other species need rescaling
by their own maximum lifespan; ``tagemortality`` is a base-10 log hazard ratio and is
never rescaled. Both clocks are released for non-commercial academic research only.

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

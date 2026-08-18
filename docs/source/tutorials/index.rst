Tutorials
=========

One walkthrough per supported data type. Each downloads a small public
example dataset, builds an AnnData object, runs one or more clocks, and reads
out the predictions.

:doc:`tutorial_utils`
   Work the catalogue from Python: list every clock, pull a clock's metadata
   and reference, and find clocks by DOI with ``pya.utils``.

:doc:`tutorial_dnam_illumina_human_array`
   The standard workflow: Illumina 450K/EPIC beta values to age predictions
   with Horvath2013, AltumAge, PCGrimAge, GrimAge2, and DunedinPACE — start
   here.

:doc:`tutorial_cpgptgrimage3`
   Run CpGPTGrimAge3, the foundation-model GrimAge variant, including its
   heavier download and preprocessing requirements.

:doc:`tutorial_dnam_illumina_mammalian_array`
   Cross-species aging on the Illumina mammalian methylation array with
   Mammalian1, MammalianLifespan, and MammalianFemale.

:doc:`tutorial_dnam_rrbs`
   Reduced-representation bisulfite sequencing: aggregate RRBS methylation
   into clock features and run the mouse clocks Thompson, Meer, Petkovich,
   and Stubbs.

:doc:`tutorial_histonemarkchipseq`
   Histone mark ChIP-seq: turn bigWig signal into gene-level features with
   ``pya.pp.bigwig_to_df`` and run CamilloH3K4me3, CamilloH3K9me3, and
   CamilloPanHistone (requires the ``pyaging[histone]`` extra).

:doc:`tutorial_atacseq`
   Chromatin accessibility: ATAC-seq peak signal to age predictions with the
   Ocampo ATAC clocks.

:doc:`tutorial_rnaseq`
   Transcriptomic aging: normalize bulk RNA-seq counts and predict biological
   age with BiTAge.

:doc:`tutorial_bloodchemistry`
   No omics required: standard clinical blood panel values to PhenoAge.

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial_utils
   tutorial_dnam_illumina_human_array
   tutorial_cpgptgrimage3
   tutorial_dnam_illumina_mammalian_array
   tutorial_dnam_rrbs
   tutorial_histonemarkchipseq
   tutorial_atacseq
   tutorial_rnaseq
   tutorial_bloodchemistry

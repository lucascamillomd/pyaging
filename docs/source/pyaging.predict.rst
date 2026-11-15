Predict
=======

Please note that most functions are helper functions and are not meant to be used directly.

A clock may declare a cohort transform, in which case ``predict_age`` preprocesses the
whole input itself before scoring it rather than reading ``adata.X`` directly. The tAge
transcriptomic clocks are the ones that do; see
:ref:`cohort-relative-transcriptomic-clocks` for the input they expect.

Repeated datasets and memory
----------------------------

Hugging Face already caches downloaded files on disk. ``ClockCache`` also reuses
the prepared PyTorch models, so repeated calls avoid remote revision checks,
deserialization, conversion to float64, and device transfers.

.. code-block:: python

   import pyaging as pya

   clocks = ["Horvath2013", "AltumAge"]
   cache = pya.pred.ClockCache(maxsize=len(clocks))
   for adata in datasets:
       pya.pred.predict_age(adata, clocks, batch_size=512, clock_cache=cache)
   cache.clear()

``maxsize`` limits the number of models, not bytes. Choose it to fit RAM or GPU
memory. If all requested models fit, set it to the number of clocks to avoid
eviction between datasets. If only one fits, process that clock across your
datasets before moving to the next clock. Load and save each dataset in the
loop when keeping all datasets in RAM would be too expensive.

The cache keeps separate entries for each clock, device, and data revision.
Changing ``PYAGING_DATA_REVISION`` therefore causes a fresh load. Changes to a
moving revision such as ``main`` are visible after ``cache.clear()``. A cache is
for sequential calls; use a separate cache per worker when running processes.
PyTorch may keep released GPU memory in its allocator for reuse after clearing.

Input matrices and cohort transforms are recomputed for each dataset. Keep each
tAge reference cohort together; dividing it into separate datasets changes its
normalization and predictions. Passing both tAge clocks in one call reuses their
shared cohort preparation.

Keep ``clean=True`` to discard each clock's aligned input after prediction.
``batch_size`` bounds the sample count sent to the GPU at once; the complete
aligned matrix still occupies host memory. Mean or median imputation can be
cheaper than KNN for large matrices, but changes how missing values are estimated.
Select the method for your analysis and keep it consistent across comparisons.

.. autoclass:: pyaging.predict.ClockCache
   :members: clear, maxsize

pyaging.predict._pred
---------------------

.. automodule:: pyaging.predict._pred
   :members:
   :undoc-members:
   :show-inheritance:

pyaging.predict._pred_utils
---------------------------

.. automodule:: pyaging.predict._pred_utils
   :members:
   :undoc-members:
   :show-inheritance:

pyaging.predict._transforms
---------------------------

.. automodule:: pyaging.predict._transforms
   :members:
   :undoc-members:
   :show-inheritance:

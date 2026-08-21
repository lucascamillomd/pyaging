Predict
=======

Please note that most functions are helper functions and are not meant to be used directly.

A clock may declare a cohort transform, in which case ``predict_age`` preprocesses the
whole input itself before scoring it rather than reading ``adata.X`` directly. The tAge
transcriptomic clocks are the ones that do; see
:ref:`cohort-relative-transcriptomic-clocks` for the input they expect.

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

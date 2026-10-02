import gc
from functools import partial

import anndata
import torch

from ..logger._live import ClockRunDisplay, DisplayLogger, display_enabled, quiet_hf_bars
from ._cache import ClockCache
from ._pred_utils import (
    _validate_batch_size,
    add_pred_ages_and_clock_metadata_adata,
    apply_cohort_transform,
    build_cohort_feature_matrix,
    check_feature_ranges,
    check_features_in_adata,
    load_clock,
    predict_ages_with_model,
    set_torch_device,
)


def predict_age(
    adata: anndata.AnnData,
    clock_names: str = "horvath2013",
    dir: str = "pyaging_data",
    batch_size: int = 1024,
    clean: bool = True,
    verbose: bool = True,
    *,
    clock_cache: ClockCache | None = None,
) -> None:
    """
    Predicts biological age using specified aging clocks.

    This function takes an AnnData object and applies one or more specified aging
    clock models to predict the biological age of the samples. It handles the entire pipeline from data
    preprocessing, model loading, prediction, to postprocessing. It also enriches the input AnnData
    object with the predicted ages and relevant clock metadata.

    Parameters
    ----------
    adata: AnnData
        An AnnData object. The object should have .X attribute for the
        data matrix and .var_names for feature names.

    clock_names: str or list of str, optional
        Names of the aging clocks to be applied. It can be a single clock name as a string or a list
        of clock names, by default "horvath2013".

    dir: str
        Retained for backward compatibility. Hugging Face files use its standard cache.

    batch_size: int
        The batch size for age inferece. Defaults to 1024.

    clean: bool
        Whether to delete the matrix data create for each clock in adata.obsm[X_clock]. Defaults to True.

    verbose: bool
        Whether to show the progress display and warnings. Animated in
        notebooks and terminals, a plain summary when output is captured,
        and fully silent when False. Defaults to True.

    clock_cache: ClockCache or None
        Optional bounded cache of prepared models to reuse across datasets.
        Keep the same cache instance for repeated calls. Clear it to refresh
        weights at mutable revisions such as ``main``. Defaults to no reuse.

    Returns
    -------
    None
        The input AnnData object is modified in place: predicted ages are added to .obs and
        clock metadata to .uns. Do not assign the return value.

    Notes
    -----
    The function is designed to be flexible and can handle both single and multiple clock predictions.
    The predicted ages are appended to the .obs attribute of the AnnData object with the clock name as
    the key. The metadata of each clock used in the prediction is stored in the .uns attribute. Change
    batch size depending on memory constraints.

    It is important that the input AnnData object's .X attribute contains data suitable for age
    prediction.

    A few clocks are cohort-relative: they read a sample only in the context of the samples
    predicted alongside it, so their preprocessing is computed here, over the whole input, before
    any batching. The tAge clocks (`tage`, `tagemortality`) work this way. Pass them raw RNA-seq
    counts and at least two samples; .X itself is never modified, and what the preprocessing did is
    recorded in .uns["tage_preparation"]. Two optional columns steer it:

    - Species: a column named `mouse`, `rat`, `macaque`, or `human` among .var_names, set to 1 for
      every sample, the same idiom the mammalian clocks use for covariates such as `female`. Only
      one may be set, and it is dropped before the gene pipeline rather than read as a gene. With
      no such column — or with the columns present but zero everywhere — the cohort is taken to be
      mouse and a warning says so, on the display and as a UserWarning.
    - Reference group: .obs["tage_reference_group"], boolean or numeric 0/1, whose truthy rows are
      the samples to centre against. Without it the cohort centres on every sample. Predictions are
      differences against that reference, not absolute values.

    Pasta, Reg, and PastaMouse fill missing values with a median computed over the whole cohort
    before ranking each sample. Their inference batch size does not change that reference.

    The function automatically handles the transfer of data and models to the appropriate compute
    device (CPU or GPU) based on system configuration.

    Examples
    --------
    >>> adata = anndata.read_h5ad("sample_data.h5ad")
    >>> predict_age(adata, clock_names=["horvath2013", "hannum"])
    >>> adata.obs["horvath2013"]  # Access predicted ages by clock name

    """
    _validate_batch_size(batch_size)
    if adata.n_obs == 0:
        raise ValueError("Prediction requires at least one sample.")

    if isinstance(clock_names, str):
        clock_names = [clock_names]
    clock_names = [clock_name.lower() for clock_name in clock_names]

    device = set_torch_device()

    enabled = display_enabled(verbose)
    display = ClockRunDisplay(clock_names, str(device), enabled=enabled)
    # Whole-cohort preprocessing, keyed by transform name and computed at most
    # once per call: tage and tagemortality read the same transformed frame, and
    # recomputing it per clock would be both slow and pointless.
    cohort_frames = {}
    with quiet_hf_bars(verbose), display:
        for clock_name in clock_names:
            display.start_clock(clock_name, "loading weights")
            pipeline_logger = DisplayLogger(lambda m, name=clock_name: display.warn(name, m))

            if clock_cache is None:
                model = load_clock(clock_name, device, dir, pipeline_logger)
            else:
                model = clock_cache._get_or_load(
                    clock_name, device, partial(load_clock, clock_name, device, dir, pipeline_logger)
                )

            # Validate before generic alignment can fill absent required inputs.
            validate_inputs = getattr(model, "validate_inputs", None)
            if validate_inputs is not None:
                validate_inputs(adata)

            # Clocks saved before either attribute existed lack both.
            transform_name = getattr(model, "cohort_transform", None)
            required_flag = getattr(model, "required_uns_flag", None)
            # A clock whose input contract nothing can satisfy for it refuses to
            # run unmarked input. A declared transform supersedes the flag: it
            # produces exactly what the flag was there to demand, and weights
            # built before the transform existed still carry the flag.
            if transform_name is None and required_flag is not None and not adata.uns.get(required_flag, False):
                raise ValueError(
                    f"Clock '{clock_name}' needs preprocessed input (adata.uns['{required_flag}'] is missing)."
                )

            if model.metadata.get("research_only", False):
                display.warn(clock_name, "research use only")

            if transform_name is not None:
                # Cohort-relative clock: its features live in the space the
                # transform produces, not in adata.X.
                if transform_name not in cohort_frames:
                    display.stage(clock_name, "preprocessing the cohort")
                    cohort_frames[transform_name] = apply_cohort_transform(adata, transform_name, dir, pipeline_logger)
                display.stage(clock_name, "matching features")
                build_cohort_feature_matrix(adata, model, cohort_frames[transform_name], pipeline_logger)
            else:
                display.stage(clock_name, "matching features")
                check_features_in_adata(adata, model, pipeline_logger)

            display.stage(clock_name, "checking feature ranges")
            check_feature_ranges(adata, model, pipeline_logger)

            display.stage(clock_name, "predicting")

            def progress_callback(completed, total, name=clock_name):
                display.progress(name, completed, total)

            predicted_ages_tensor = predict_ages_with_model(
                adata, model, device, batch_size, pipeline_logger, progress_callback=progress_callback
            )

            display.stage(clock_name, "writing results")
            add_pred_ages_and_clock_metadata_adata(adata, model, predicted_ages_tensor, dir, pipeline_logger)

            if clean:
                del adata.obsm[f"X_{clock_name}"]

            # Release references before collecting memory and loading the next
            # clock, so two large models need not coexist on the GPU.
            del model, predicted_ages_tensor
            if clock_cache is None:
                gc.collect()
                torch.cuda.empty_cache()

            display.finish_clock(clock_name)
        display.finish(n_samples=adata.n_obs)

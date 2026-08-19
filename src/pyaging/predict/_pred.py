import gc

import anndata
import torch

from ..logger._live import ClockRunDisplay, DisplayLogger, display_enabled, quiet_hf_bars
from ._pred_utils import (
    add_pred_ages_and_clock_metadata_adata,
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

    The function automatically handles the transfer of data and models to the appropriate compute
    device (CPU or GPU) based on system configuration.

    Examples
    --------
    >>> adata = anndata.read_h5ad("sample_data.h5ad")
    >>> predict_age(adata, clock_names=["horvath2013", "hannum"])
    >>> adata.obs["horvath2013"]  # Access predicted ages by clock name

    """
    # Ensure clock_names is a list with lowercase names
    if isinstance(clock_names, str):
        clock_names = [clock_names]
    clock_names = [clock_name.lower() for clock_name in clock_names]

    # Set device for PyTorch operations
    device = set_torch_device()

    enabled = display_enabled(verbose)
    display = ClockRunDisplay(clock_names, str(device), enabled=enabled)
    with quiet_hf_bars(verbose), display:
        for clock_name in clock_names:
            display.start_clock(clock_name, "loading weights")
            # Pipeline warnings surface on the display
            pipeline_logger = DisplayLogger(lambda m, name=clock_name: display.warn(name, m))

            # Load and prepare the clock
            model = load_clock(clock_name, device, dir, pipeline_logger)

            # Disclaimer for commercial clocks
            if model.metadata.get("research_only", False):
                display.warn(clock_name, "research use only")

            # Check and update adata for missing features
            display.stage(clock_name, "matching features")
            check_features_in_adata(adata, model, pipeline_logger)

            # Perform age prediction applying preprocessing and postprocessing steps
            display.stage(clock_name, "predicting")

            def progress_callback(completed, total, name=clock_name):
                display.progress(name, completed, total)

            predicted_ages_tensor = predict_ages_with_model(
                adata, model, device, batch_size, pipeline_logger, progress_callback=progress_callback
            )

            # Add predicted ages and clock metadata to adata
            display.stage(clock_name, "writing results")
            add_pred_ages_and_clock_metadata_adata(adata, model, predicted_ages_tensor, dir, pipeline_logger)

            # Delete the clock matrix object
            if clean:
                del adata.obsm[f"X_{clock_name}"]

            # Flush memory
            gc.collect()
            torch.cuda.empty_cache()

            display.finish_clock(clock_name)
        display.finish(n_samples=adata.n_obs)

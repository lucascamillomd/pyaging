import gc

import anndata

from ._pred_utils import *


def predict_age(
    adata: anndata.AnnData,
    clock_names: str = "horvath2013",
    dir: str = "pyaging_data",
    batch_size: int = 1024,
    clean: bool = True,
    verbose: bool = True,
) -> anndata.AnnData:
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
        The directory to deposit the downloaded file. Defaults to "pyaging_data".

    batch_size: int
        The batch size for age inferece. Defaults to 1024.

    clean: bool
        Whether to delete the matrix data create for each clock in adata.obsm[X_clock]. Defaults to True.

    verbose: bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    AnnData
        The input AnnData object enriched with the predicted ages and clock metadata in the .obs and
        .uns attributes, respectively.

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
    >>> adata = predict_age(adata, clock_names=["horvath2013", "hannum"])
    >>> adata.obs["horvath2013"]  # Access predicted ages by clock name

    """
    logger = LoggerManager.gen_logger("predict_age")
    if not verbose:
        silence_logger("predict_age")
    logger.first_info("Starting predict_age function")

    # Ensure clock_names is a list with lowercase names
    if isinstance(clock_names, str):
        clock_names = [clock_names]
    clock_names = [clock_name.lower() for clock_name in clock_names]

    # Set device for PyTorch operations
    device = set_torch_device(logger)

    for clock_name in clock_names:
        logger.info(f"🕒 Processing clock: {clock_name}", indent_level=1)

        # Load and prepare the clock
        model = load_clock(clock_name, device, dir, logger, indent_level=2)

        # Disclaimer for commercial clocks
        if model.metadata.get("research_only", False):  # Defaults to False if 'research_only' key doesn't exist
            logger.warning(
                f"⚠️ Clock '{clock_name}' is for research purposes only. Please check the clock's documentation or notes for more information.",
                indent_level=2,
            )

        # Check and update adata for missing features
        check_features_in_adata(
            adata,
            model,
            logger,
            indent_level=2,
        )

        # Perform age prediction using the model applying preprocessing and postprocessing steps
        predicted_ages_tensor = predict_ages_with_model(adata, model, device, batch_size, logger, indent_level=2)

        # Add predicted ages and clock metadata to adata
        add_pred_ages_and_clock_metadata_adata(adata, model, predicted_ages_tensor, dir, logger, indent_level=2)

        # Delete the clock matrix object
        if clean:
            del adata.obsm[f"X_{clock_name}"]

        # Flush memory
        gc.collect()
        torch.cuda.empty_cache()

    logger.done()

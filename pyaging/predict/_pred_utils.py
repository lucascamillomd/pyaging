import marshal
import math
import ntpath
import os
import types
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import anndata
import numpy as np
import pandas as pd
import torch
from anndata.experimental.pytorch import AnnLoader
from torch.utils.data import DataLoader, TensorDataset

try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
except:
    CUPY_AVAILABLE = False

from ..logger import LoggerManager, main_tqdm, silence_logger
from ..models import *
from ..utils import download, load_clock_metadata, progress
from ._postprocessing import *
from ._preprocessing import *


@progress("Load clock")
def load_clock(clock_name: str, device: str, dir: str, logger, indent_level: int = 2) -> Tuple:
    """
    Loads the specified aging clock from a remote source and returns its components.

    This function downloads the weights and configuration of a specified aging clock from a
    remote server. This allows users to instantiate and use the clock in their analyses.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock to be loaded. This name is used to construct the URL
        for downloading the clock's weights and configuration.

    device : str
        Device to move clock to. Eithe 'cpu' or 'cuda'.

    dir : str
        The directory to deposit the downloaded file.

    logger : Logger
        A logger object used for logging information during the function execution.

    indent_level : int, optional
        The indentation level for the logger, by default 2. It controls the formatting
        of the log messages.

    Returns
    -------
    pyagingModel
        A clock model

    Notes
    -----
    The clock's weights and configuration are assumed to be stored in a .pt (PyTorch) file
    on a remote server. The URL for the clock is constructed based on the clock's name.
    The function uses the `download` utility to retrieve the file. If the clock or its
    components are not found, the function may fail or return incomplete information.

    The logger is used extensively for progress tracking and information logging, enhancing
    transparency and user experience.

    Examples
    --------
    >>> clock = load_clock("clock1", "pyaging_data", logger)

    """
    clock_name = clock_name.lower()
    url = f"https://pyaging.s3.amazonaws.com/clocks/weights0.1.0/{clock_name}.pt"
    try:
        download(url, dir, logger, indent_level=indent_level)
    except:
        logger.error(
            f"Clock {clock_name} is not available on pyaging. "
            f"Please refer to the clock names in the clock glossary table "
            f"in the package documentation page: pyaging.readthedocs.io",
            indent_level=indent_level + 1,
        )
        raise NameError

    # Define the path to the clock weights file
    weights_path = os.path.join(dir, f"{clock_name}.pt")

    # Load the clock from the file
    clock = torch.load(weights_path, weights_only=False)

    # Prepare clock for inference
    clock.to(torch.float64)
    clock.to(device)
    clock.eval()

    return clock


@progress("Check features in adata")
def check_features_in_adata(
    adata: anndata.AnnData,
    model: pyagingModel,
    logger,
    indent_level: int = 2,
) -> anndata.AnnData:
    """
    Verifies if all required features are present in an AnnData object and adds missing features.

    This function checks an AnnData object (commonly used in single-cell analysis) to ensure
    that it contains all the necessary features specified in the 'features' list inside the model.
    If any features are missing, they are added to the AnnData object with a default value of 0 or
    with a reference value if given. This is crucial for downstream analyses where the presence of
    all specified features is assumed.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked. It is a commonly used data structure in single-cell
        genomics containing high-dimensional data.

    model : pyagingModel
        The pyagingModel of the aging clock of interest. Must contain defined features.

    logger : Logger
        A logger object used for logging information about the process, such as the number
        of missing features.

    indent_level : int, optional
        The indentation level for the logger, by default 2. It controls the formatting
        of the log messages.

    Returns
    -------
    anndata.AnnData
        The updated AnnData object, which includes any missing features added with a default
        value of 0 (or reference value if provided).

    Notes
    -----
    This function is particularly useful in preprocessing steps where the consistency of
    data structure across different datasets is crucial. The function modifies the AnnData
    object if there are missing features and logs detailed information about these modifications.

    The added features are initialized with zeros. This approach, while providing completeness,
    may introduce biases if not accounted for in downstream analyses. If reference values are
    provided, then they are used instead of zeros.

    Examples
    --------
    >>> updated_adata = check_features_in_adata(adata, bitage, ["gene1", "gene2"], logger)
    >>> updated_adata.var_names
    Index(['gene1', 'gene2', ...], dtype='object')

    """

    # Preallocate the data matrix
    adata.obsm[f"X_{model.metadata['clock_name']}"] = (
        cp.empty((adata.n_obs, len(model.features)))
        if CUPY_AVAILABLE
        else np.empty((adata.n_obs, len(model.features)), order="F")
    )

    # Find indices of matching features in adata.var_names
    feature_indices = {feature: i for i, feature in enumerate(adata.var_names)}
    model_feature_indices = np.array([feature_indices.get(feature, -1) for feature in model.features])

    # Identify missing features
    missing_features_mask = model_feature_indices == -1
    missing_features = np.array(model.features)[missing_features_mask].tolist()

    # Assign values for existing features
    existing_features_mask = ~missing_features_mask
    existing_features_indices = model_feature_indices[existing_features_mask]
    adata.obsm[f"X_{model.metadata['clock_name']}"][:, existing_features_mask] = adata.X[:, existing_features_indices]

    # Handle missing features
    adata.obsm[f"X_{model.metadata['clock_name']}"][:, missing_features_mask] = (
        np.array(model.reference_values)[missing_features_mask] if model.reference_values is not None else 0
    )

    # Calculate missing features statistics
    num_missing_features = len(missing_features)
    percent_missing = 100 * num_missing_features / len(model.features)

    # Add missing features and percent missing values to the clock
    adata.uns[f"{model.metadata['clock_name']}_percent_na"] = percent_missing
    adata.uns[f"{model.metadata['clock_name']}_missing_features"] = missing_features

    # Raises error if there are no features in the data
    if percent_missing == 100:
        logger.error(
            f"Every single feature out of {len(model.features)} features "
            f"is missing. Please double check the features in the adata object"
            f" actually contain the clock features such as {missing_features[:np.min([3, num_missing_features])]}, etc.",
            indent_level=3,
        )
        raise NameError

    # Log and add missing features if any
    if len(missing_features) > 0:
        logger.warning(
            f"{num_missing_features} out of {len(model.features)} features "
            f"({percent_missing:.2f}%) are missing: {missing_features[:np.min([3, num_missing_features])]}, etc.",
            indent_level=indent_level + 1,
        )
        # If there are reference values provided
        if model.reference_values is not None:
            logger.info(
                f"Using reference feature values for {model.metadata['clock_name']}",
                indent_level=indent_level + 1,
            )
        else:
            logger.info(
                "Filling missing features entirely with 0",
                indent_level=indent_level + 1,
            )
    else:
        logger.info(
            "All features are present in adata.var_names.",
            indent_level=indent_level + 1,
        )


@progress("Predict ages with model")
def predict_ages_with_model(
    adata: anndata.AnnData,
    model: pyagingModel,
    device: str,
    batch_size: int,
    logger,
    indent_level: int = 2,
) -> torch.Tensor:
    """
    Predict biological ages using a trained model and input data.

    This function takes a machine learning model and input data, and returns predictions made by the model.
    It's primarily used for estimating biological ages based on various biological markers. The function
    assumes that the model is already trained. A dataloader is used because of possible memory constraints
    for large datasets.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the dataset. Its `.X` attribute is expected to be a matrix where rows
        correspond to samples and columns correspond to features.

    model : pyagingModel
        The pyagingModel of the aging clock of interest.

    device : str
        Device to move AnnData to during inference. Eithe 'cpu' or 'cuda'.

    batch_size : int
        Batch size for the AnnLoader object to predict age.

    logger : Logger
        A logger object for logging the progress or any relevant information during the prediction process.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    predictions : torch.Tensor
        An array of predicted ages or biological markers, as returned by the model.

    Notes
    -----
    Ensure that the data is preprocessed (e.g., scaled, normalized) as required by the model before
    passing it to this function. The model should be in evaluation mode if it's a type that has different
    behavior during training and inference (e.g., PyTorch models).

    The exact nature of the predictions (e.g., age, biological markers) depends on the model being used.

    Examples
    --------
    >>> model = load_pretrained_model()
    >>> predictions = predict_ages_with_model(model, "cpu", logger)
    >>> print(predictions[:5])
    [34.5, 29.3, 47.8, 50.1, 42.6]

    """

    # If there is a preprocessing step
    if model.preprocess_name is not None:
        logger.info(
            f"The preprocessing method is {model.preprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no preprocessing necessary", indent_level=indent_level + 1)

    # If there is a postprocessing step
    if model.postprocess_name is not None:
        logger.info(
            f"The postprocessing method is {model.postprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no postprocessing necessary", indent_level=indent_level + 1)

    # Create an AnnLoader
    use_cuda = torch.cuda.is_available()
    dataloader = AnnLoader(adata, batch_size=batch_size, use_cuda=use_cuda)

    # Use the AnnLoader for batched prediction
    predictions = []
    with torch.inference_mode():
        for batch in main_tqdm(dataloader, indent_level=indent_level + 1, logger=logger):
            batch_pred = model(batch.obsm[f"X_{model.metadata['clock_name']}"])
            predictions.append(batch_pred)
    # Concatenate all batch predictions
    predictions = torch.cat(predictions)

    return predictions


@progress("Add predicted ages and clock metadata to adata")
def add_pred_ages_and_clock_metadata_adata(
    adata: anndata.AnnData,
    model: pyagingModel,
    predicted_ages: torch.tensor,
    dir: str,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Add predicted ages to an AnnData object as a new column in the observation (obs) attribute. Also adds
    the specific clock metadata to the `uns` attribute of an AnnData object.

    This function appends the predicted ages, obtained from a biological aging clock or similar model, to
    the AnnData object's `obs` attribute. The predicted ages are added as a new column, named after the
    clock used to generate these predictions.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to which the predicted ages will be added. It's a data structure for handling
        large-scale biological data, like gene expression matrices, commonly used in bioinformatics.

    model : pyagingModel
        The aging clock from which to get the metadata.

    predicted_ages : torch.tensor
        A torch tensor of predicted ages corresponding to the samples in the AnnData object. The length
        of this array should match the number of samples in `adata`.

    dir: str
        The directory to deposit the downloaded file.

    logger : Logger
        A logger object for logging the progress or relevant information during the operation.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    None
        This function modifies the AnnData object in-place and does not return any value.

    Notes
    -----
    It is essential to ensure that the length of `predicted_ages` matches the number of samples in the
    `adata` object. Mismatch in lengths will lead to errors or misaligned data.

    This function is part of a pipeline that integrates aging clock predictions with the
    standard data structures used in bioinformatics, facilitating downstream analyses like visualization
    or statistical testing.

    Examples
    --------
    >>> adata = anndata.AnnData(np.random.rand(5, 10))
    >>> predicted_ages = [25, 30, 35, 40, 45]
    >>> add_pred_ages_adata(adata, predicted_ages_tensor, clock, "pyaging_data", logger)
    >>> adata.obs["horvath2013"]
    0    25
    1    30
    2    35
    3    40
    4    45
    Name: horvath2013, dtype: int64
    >>> adata.uns["horvath2013_metadata"]
    {'species': 'Homo sapiens', 'data_type': 'methylation', 'citation': 'Horvath, S. (2013)'}

    """
    # Convert from a torch tensor to a flat numpy array
    predicted_ages = predicted_ages.cpu().detach().numpy().flatten()

    # Add predicted ages to adata.obs
    adata.obs[model.metadata["clock_name"]] = predicted_ages

    # Add clock metadata to adata.uns
    adata.uns[f"{model.metadata['clock_name']}_metadata"] = model.metadata


@progress("Set PyTorch device")
def set_torch_device(logger, indent_level: int = 1) -> None:
    """
    Set and return the PyTorch device based on the availability of CUDA.

    This function checks if CUDA is available in the system and accordingly sets the PyTorch device to
    either 'cuda' or 'cpu'. If CUDA is available, it utilizes GPU acceleration for PyTorch operations,
    significantly enhancing computation speed for large datasets. The chosen device is logged for
    user reference.

    Parameters
    ----------
    logger : Logger
        A logger object for logging the selected device.

    indent_level : int, optional
        The indentation level for logging messages, by default 1.

    Returns
    -------
    torch.device
        The PyTorch device object set to 'cuda' if CUDA is available, or 'cpu' otherwise.

    Notes
    -----
    The function automatically detects the availability of CUDA and makes a decision without user input.
    This makes it convenient for deploying code on different machines without the need for manual
    configuration.

    It is important to use the returned device for all PyTorch operations to ensure that they are
    executed on the correct hardware (CPU or GPU).

    Examples
    --------
    >>> logger = pyaging.logger.LoggerManager.gen_logger("example")
    >>> device = set_torch_device(logger)
    >>> print(device)
    device(type='cuda')  # or device(type='cpu') if CUDA is not available

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}", indent_level=2)
    return device

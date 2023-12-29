from typing import Tuple, List
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import anndata
from anndata.experimental.pytorch import AnnLoader
import ntpath
from urllib.request import urlretrieve


from ..models import *
from ..utils import progress, load_clock_metadata, download
from ..logger import LoggerManager, main_tqdm, silence_logger
from ._preprocessing import *
from ._postprocessing import *


@progress("Load clock")
def load_clock(clock_name: str, dir: str, logger, indent_level: int = 2) -> Tuple:
    """
    Loads the specified aging clock from a remote source and returns its components.

    This function downloads the weights and configuration of a specified aging clock from a
    remote server. It then loads and returns various components of the clock such as its
    features, weight dictionary, and any preprocessing or postprocessing steps. This allows
    users to instantiate and use the clock in their analyses.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock to be loaded. This name is used to construct the URL
        for downloading the clock's weights and configuration.

    dir : str
        The directory to deposit the downloaded file.

    logger : Logger
        A logger object used for logging information during the function execution.

    indent_level : int, optional
        The indentation level for the logger, by default 2. It controls the formatting
        of the log messages.

    Returns
    -------
    tuple
        A tuple containing the following components of the clock:
        - features: The features used by the clock.
        - reference_feature_values: Reference values for features in case features are missing.
        - weight_dict: A dictionary of weights used in the clock's model.
        - preprocessing: Any preprocessing steps required for the clock's input data.
        - postprocessing: Any postprocessing steps applied to the clock's output.
        - preprocessing_helper: Any preprocessing helper file.
        - postprocessing_helper: Any postprocessing helper file.

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
    >>> features, reference_feature_values, _, _, _, _, _ = load_clock("clock1", "pyaging_data", logger)
    >>> print(features)
    ['feature1', 'feature2', ...]

    """
    url = f"https://pyaging.s3.amazonaws.com/clocks/weights/{clock_name}.pt"
    download(url, dir, logger, indent_level=indent_level)

    # Define the path to the clock weights file
    weights_path = os.path.join(dir, f"{clock_name}.pt")

    # Load the clock dictionary from the file
    clock_dict = torch.load(weights_path)

    # Extract relevant information from the clock dictionary
    features = clock_dict["features"]
    reference_feature_values = clock_dict.get("reference_feature_values", None)
    weight_dict = clock_dict["weight_dict"]
    preprocessing = clock_dict.get("preprocessing", None)
    postprocessing = clock_dict.get("postprocessing", None)
    preprocessing_helper = clock_dict.get("preprocessing_helper", None)
    postprocessing_helper = clock_dict.get("postprocessing_helper", None)

    return (
        features,
        reference_feature_values,
        weight_dict,
        preprocessing,
        postprocessing,
        preprocessing_helper,
        postprocessing_helper,
    )


@progress("Check features in adata")
def check_features_in_adata(
    adata: anndata.AnnData,
    clock_name: str,
    features: List[str],
    reference_feature_values: List[float],
    logger,
    indent_level: int = 2,
) -> anndata.AnnData:
    """
    Verifies if all required features are present in an AnnData object and adds missing features.

    This function checks an AnnData object (commonly used in single-cell analysis) to ensure
    that it contains all the necessary features specified in the 'features' list. If any features
    are missing, they are added to the AnnData object with a default value of 0 or with a reference
    value if given. This is crucial for downstream analyses where the presence of all specified
    features is assumed.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked. It is a commonly used data structure in single-cell
        genomics containing high-dimensional data.

    clock_name : str
        The name of the aging clock. The percent of missing features will be added to adata.uns
        for this specific clock.

    features : list
        A list of features (e.g., gene names or other identifiers) that are expected to be
        present in the 'adata'.

    reference_feature_values : list
        A list of the reference values for each feature. The order must match the order of features.
        If no

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
    >>> updated_adata = check_features_in_adata(adata, ['gene1', 'gene2'], logger)
    >>> updated_adata.var_names
    Index(['gene1', 'gene2', ...], dtype='object')

    """
    # Identify missing features
    missing_features = [
        feature for feature in features if feature not in adata.var_names
    ]

    # Calculate the percentage of missing features
    total_features = len(features)
    num_missing_features = len(missing_features)
    percent_missing = (
        (num_missing_features / total_features) * 100 if total_features > 0 else 0
    )

    # Add percent missing values to the clock
    adata.uns[f"{clock_name}_percent_na"] = percent_missing

    # Raises error if there are no features in the data
    if percent_missing == 100:
        logger.error(
            f"Every single feature out of {total_features} features "
            f"is missing. Please double check the features in the adata object"
            f" actually contain the clock features such as {missing_features[:np.min([3, num_missing_features])]}, etc.",
            indent_level=3,
        )
        raise NameError

    # Log and add missing features if any
    if missing_features:
        logger.warning(
            f"{num_missing_features} out of {total_features} features "
            f"({percent_missing:.2f}%) are missing: {missing_features[:np.min([3, num_missing_features])]}, etc.",
            indent_level=indent_level + 1,
        )

        # If there are reference values provided
        if reference_feature_values:
            logger.info(
                f"Using reference feature values for {clock_name}",
                indent_level=indent_level + 1,
            )

            # Map features to reference values
            feature_value_map = dict(zip(features, reference_feature_values))

            # Pre-allocate with reference values, if missing, use a default value (e.g., 0)
            missing_data = np.array(
                [feature_value_map.get(f, 0) for f in missing_features] * adata.n_obs
            ).reshape(adata.n_obs, num_missing_features)
        else:
            logger.info(
                f"Filling missing features entirely with 0",
                indent_level=indent_level + 1,
            )

            # Create an empty array
            missing_data = np.zeros((adata.n_obs, num_missing_features))

        adata_empty = anndata.AnnData(
            X=missing_data,
            obs=adata.obs,
            var=pd.DataFrame(
                np.ones((num_missing_features, 1)),
                index=missing_features,
                columns=["percent_na"],
            ),
            layers=dict(zip(adata.layers.keys(), [missing_data] * len(adata.layers))),
            uns=adata.uns,
        )

        # Concatenate original adata with the missing adata
        adata = anndata.concat(
            [adata, adata_empty], axis=1, merge="same", uns_merge="unique"
        )

        logger.info(
            f"Expanded adata with {num_missing_features} missing features",
            indent_level=indent_level + 1,
        )
    else:
        logger.info(
            "All features are present in adata.var_names.",
            indent_level=indent_level + 1,
        )

    return adata


@progress("Initialize model")
def initialize_model(
    clock_name: str,
    features: List[str],
    weight_dict: dict,
    device: str,
    logger,
    indent_level: int = 2,
) -> torch.nn.Module:
    """
    Initialize and configure a predictive model based on the specified aging clock.

    This function selects and initializes a machine learning model tailored to a particular aging clock,
    indicated by `clock_name`. It loads the model weights from `weight_dict` and prepares the model
    for inference (evaluation mode). Different types of clocks require different models, and this function
    handles the instantiation and configuration of these models based on the clock type.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock for which the model is to be initialized. This name determines
        the type of model to be used.

    features : list
        A list of feature names that the model will use for making predictions. The length of this
        list is used to configure the input dimension of the model.

    weight_dict : dict
        A dictionary containing the pre-trained weights of the model. These weights are loaded into
        the model for making predictions.

    device : str
        Device to move model to after initialization. Eithe 'cpu' or 'cuda'.

    logger : Logger
        A logger object for logging the progress and any information or warnings during the
        initialization process.

    indent_level : int, optional
        The indentation level for the logger, by default 2. It controls the formatting of the log
        messages.

    Returns
    -------
    model : torch.nn.Module
        The initialized and configured PyTorch model ready for making predictions.

    Raises
    ------
    ValueError
        If the provided `clock_name` is not supported or recognized.

    Notes
    -----
    The function currently supports a range of models including linear models, principal component
    (PC) based models, and specific models for complex clocks like `AltumAge` and `PCGrimAge`.
    It is crucial that the `weight_dict` matches the structure expected by the model corresponding
    to the `clock_name`.

    The function assumes the availability of a pre-defined set of model classes like `LinearModel`,
    `PCLinearModel`, `PCGrimAge`, `AltumAge`, etc., which should be defined elsewhere in the codebase.

    Examples
    --------
    >>> model = initialize_model('horvath2013', features, weight_dict, logger)
    >>> print(type(model))
    <class 'torch.nn.modules.linear.LinearModel'>

    """
    # Model selection based on clock name
    if clock_name in [
        "horvath2013",
        "skinandblood",
        "hannum",
        "phenoage",
        "dnamphenoage",
        "dnamtl",
        "dunedinpace",
        "replitali",
        "pedbe",
        "mammalian1",
        "mammalian2",
        "mammalian3",
        "mammalianlifespan",
        "ocampoatac1",
        "ocampoatac2",
        "bitage",
        "zhangmortality",
        "zhangen",
        "zhangblup",
        "leecontrol",
        "leerobust",
        "leerefinedrobust",
        "meermultitissue",
        "thompsonmultitissue",
        "petkovichblood",
        "stubbsmultitissue",
        "lin",
        "knight",
        "hrsinchphenoage",
    ]:
        model = LinearModel(len(features))
    elif clock_name in [
        "pchorvath2013",
        "pcskinandblood",
        "pchannum",
        "pcphenoage",
        "pcdnamtl",
    ]:
        model = PCLinearModel(len(features), pc_dim=weight_dict["rotation"].shape[1])
    elif clock_name == "pcgrimage":
        model = PCGrimAge(
            sum(["cg" in feature for feature in features]),
            pc_dim=weight_dict["rotation"].shape[1],
            comp_dims=[
                weight_dict[f"step1_layers.{i}.weight"].shape[1]
                for i in range(weight_dict["step2.weight"].shape[1] - 2)
            ],
        )
    elif clock_name == "altumage":
        model = AltumAge()
    elif clock_name in [
        "h3k4me3",
        "h3k4me1",
        "h3k9me3",
        "h3k9ac",
        "h3k27me3",
        "h3k27ac",
        "h3k36me3",
        "panhistone",
    ]:
        model = PCARDModel(len(features), pc_dim=weight_dict["rotation"].shape[1])
    else:
        raise ValueError(f"Clock '{clock_name}' is not supported.")

    model.load_state_dict(weight_dict)
    model.to(torch.float64)
    model.to(device)
    model.eval()
    return model


@progress("Preprocess data")
def preprocess_data(
    adata: anndata.AnnData,
    preprocessing: str,
    preprocessing_helper,
    features: List[str],
    logger,
    indent_level: int = 2,
) -> anndata.AnnData:
    """
    Preprocess the input data based on the specified method and the clock dictionary.

    This function applies a specified preprocessing method to the input data, which is necessary
    for aligning the data format with the requirements of the aging clock models. The function
    supports multiple preprocessing methods, including scaling, log transformation, binarization, etc.
    The specific preprocessing method to be used is indicated by the `preprocessing` parameter. A new
    layers with the format X_{preprocessing} is added to the adata object.

    Parameters
    ----------
    adata : anndata.AnnData
        The input data to be preprocessed in AnnData format.

    preprocessing : str
        The name of the preprocessing method to apply. Supported methods include 'scale', 'log1p',
        'binarize', and 'quantile_normalization_with_gold_standard'.

    preprocessing_helper:
        Any preprocessing helper file, from an scikit-learn scaler object to a list of standard values.

    features : List[str]
        The features used by the clock.

    logger : Logger
        A logger object for logging the progress and any information or warnings during preprocessing.

    indent_level : int, optional
        The indentation level for the logger, by default 2. It controls the formatting of the log
        messages.

    Returns
    -------
    adata : anndata.AnnData
        The preprocessed data, ready to be input into the aging clock models.

    Notes
    -----
    The function assumes that the input `data` is in a format compatible with the preprocessing methods.
    For instance, 'scale' may expect a numeric tensor, while 'log1p' requires non-negative values.
    Users must ensure that the input data is appropriate for the selected preprocessing method.

    Examples
    --------
    >>> processed_adata = preprocess_data(adata, "scale", preprocessing_helper, features, logger)
    >>> print(processed_adata.shape)
    np.array([...])

    """

    # Move to adata.X for preprocessing
    adata.X = (
        adata.layers["X_imputed"].copy()
        if "X_imputed" in adata.layers
        else adata.layers["X_original"].copy()
    )

    # Skip if it there is no preprocessing to be done
    if preprocessing is None:
        logger.info(
            "There is no preprocessing to be done",
            indent_level=3,
        )
        return adata

    logger.info(f"Preprocessing data with function {preprocessing}", indent_level=3)
    # Apply specified preprocessing method
    if preprocessing == "tpm_norm_log1p":
        adata.X = tpm_norm_log1p(adata.X, preprocessing_helper)
    elif preprocessing == "binarize":
        adata.X = binarize(adata.X)
    elif preprocessing == "scale_row":
        adata.X = scale_row(adata.X, adata[:, features].X)
    elif preprocessing == "scale":
        X = adata[:, features].X
        X = scale(X, preprocessing_helper)
        adata[:, features].X = X
    elif preprocessing == "quantile_normalization_with_gold_standard":
        gold_standard_df = pd.DataFrame(
            dict(
                zip(
                    preprocessing_helper["gold_standard_probes"],
                    preprocessing_helper["gold_standard_means"],
                )
            ),
            index=["means"],
        ).T
        common_features = np.intersect1d(
            adata.var_names, gold_standard_df.index.tolist()
        )
        X = adata[:, common_features].X
        X = quantile_normalize_with_gold_standard(
            X, gold_standard_df.loc[common_features, "means"].tolist()
        )
        adata[:, common_features].X = X
    else:
        logger.error(f"Preprocessing function {preprocessing} not found.")
        raise ValueError()

    return adata


@progress("Postprocess data")
def postprocess_data(
    data: np.ndarray,
    postprocessing: str,
    clock_dict: dict,
    logger,
    indent_level: int = 2,
) -> np.ndarray:
    """
    Apply postprocessing to the data based on the specified method.

    This function is designed to apply a specific postprocessing method to data, typically after
    it has been processed by a machine learning model. The type of postprocessing is determined
    by the `postprocessing` parameter. The function supports several methods, including various
    forms of anti-log transformations, which are applied using a vectorized approach for efficiency.

    Parameters
    ----------

    data : array-like
        The data to be postprocessed. It should be compatible with the specified postprocessing method.

    postprocessing : str
        The name of the postprocessing method to be applied. Supported methods include
        'anti_log_linear', 'anti_logp2', 'anti_log', etc.

    postprocessing_helper:
        Any postprocessing helper file, from an scikit-learn scaler object to a list of standard values.

    logger : Logger
        A logger object for logging the progress, information, or warnings during postprocessing.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    postprocessed_data : array-like
        The data after applying the specified postprocessing method.

    Notes
    -----
    The function relies on numpy's vectorization capabilities for efficient computation. The
    postprocessing functions, like `anti_log_linear`, need to be defined elsewhere in the codebase.

    The choice of postprocessing method should be consistent with the preprocessing method and the
    nature of the data and model predictions.

    Examples
    --------
    >>> postprocessed_data = postprocess_data(data, "anti_log_linear", postprocessing_helper, logger)
    >>> print(postprocessed_data.shape)
    (num_samples, num_features)

    """
    # Skip if it there is no postprocessing to be done
    if postprocessing is None:
        logger.info(
            "There is no postprocessing to be done",
            indent_level=3,
        )
        return data

    logger.info(f"Postprocessing data with function {postprocessing}", indent_level=3)
    # Apply specified postprocessing method using vectorization
    if postprocessing == "anti_log_linear":
        vectorized_function = np.vectorize(anti_log_linear)
        data = vectorized_function(data)
    elif postprocessing == "anti_logp2":
        vectorized_function = np.vectorize(anti_logp2)
        data = vectorized_function(data)
    elif postprocessing == "anti_log":
        vectorized_function = np.vectorize(anti_log)
        data = vectorized_function(data)
    elif postprocessing == "anti_log_log":
        vectorized_function = np.vectorize(anti_log_log)
        data = vectorized_function(data)
    elif postprocessing == "mortality_to_phenoage":
        vectorized_function = np.vectorize(mortality_to_phenoage)
        data = vectorized_function(data)
    elif postprocessing == "petkovichblood":
        vectorized_function = np.vectorize(petkovichblood)
        data = vectorized_function(data)
    elif postprocessing == "stubbsmultitissue":
        vectorized_function = np.vectorize(stubbsmultitissue)
        data = vectorized_function(data)
    else:
        logger.error(f"Postprocessing function {postprocessing} not found.")
        raise ValueError()
    return data


@progress("Predict ages with model")
def predict_ages_with_model(
    model: torch.nn.Module,
    adata: torch.Tensor,
    features: List[str],
    device: str,
    logger,
    indent_level: int = 2,
) -> torch.Tensor:
    """
    Predict biological ages using a trained model and input data.

    This function takes a machine learning model and input data, and returns predictions made by the model.
    It's primarily used for estimating biological ages based on various biological markers. The function
    assumes that the model is already trained and the data is preprocessed according to the model's requirements.
    A dataloader is used because of possible memory constraints.

    Parameters
    ----------
    model : torch.nn.Module
        A pre-trained machine learning model that can make predictions.

    adata : anndata.AnnData
        The AnnData object containing the dataset. Its `.X` attribute is expected to be a matrix where rows
        correspond to samples and columns correspond to features.

    features : list of str
        A list of feature names to be included in the output array. Only these features from the AnnData object will
        be extracted for age prediction.

    device : str
        Device to move AnnData to during inference. Eithe 'cpu' or 'cuda'.

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
    >>> predictions = predict_ages_with_model(model, adata, features, 'cpu', logger)
    >>> print(predictions[:5])
    [34.5, 29.3, 47.8, 50.1, 42.6]

    """
    # Create an AnnLoader
    use_cuda = device == "cuda"
    dataloader = AnnLoader(adata, batch_size=1024, use_cuda=use_cuda)

    # Use the AnnLoader for batched prediction
    predictions = []
    with torch.no_grad():
        for batch in main_tqdm(
            dataloader, indent_level=indent_level + 1, logger=logger
        ):
            batch_pred = model(batch[:, features].X)
            predictions.append(batch_pred)

    # Concatenate all batch predictions
    predictions = torch.cat(predictions)
    return predictions


@progress("Convert tensor to numpy array")
def convert_tensor_to_numpy_array(
    tensor: torch.Tensor, logger, indent_level: int = 2
) -> np.ndarray:
    """
    Convert a PyTorch tensor to a flattened NumPy array.

    This utility function takes a PyTorch tensor and converts it into a 1D NumPy array. It is useful in scenarios
    where PyTorch tensors need to be processed or analyzed using NumPy-based tools or functions. The function ensures
    that the tensor is detached from the current computation graph (if applicable) and then flattens it to a 1D array.

    Parameters
    ----------
    tensor : torch.Tensor
        The PyTorch tensor to be converted. This can be a tensor of any shape.

    logger : Logger
        A logger object for logging the progress or any relevant information during the conversion process.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    numpy_array : ndarray
        A 1D NumPy array equivalent to the input tensor.

    Notes
    -----
    This function detaches the tensor from the computation graph, so it should be used with caution if the tensor
    is part of a PyTorch computational graph, such as when dealing with gradients in training neural networks.

    The flattening to a 1D array may lose the original shape information of the tensor, which should be considered
    if the shape is important for further processing.

    Examples
    --------
    >>> tensor = torch.tensor([[1, 2], [3, 4]])
    >>> numpy_array = convert_tensor_to_numpy_array(tensor, logger)
    >>> numpy_array
    array([1, 2, 3, 4])

    """
    return tensor.cpu().detach().numpy().flatten()


@progress("Add predicted ages to adata")
def add_pred_ages_adata(
    adata: anndata.AnnData,
    predicted_ages: np.ndarray,
    clock_name: str,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Add predicted ages to an AnnData object as a new column in the observation (obs) attribute.

    This function appends the predicted ages, obtained from a biological aging clock or similar model, to
    the AnnData object's `obs` attribute. The predicted ages are added as a new column, named after the
    clock used to generate these predictions. This allows for easy integration and comparison of age
    predictions within the bioinformatics data analysis workflows.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to which the predicted ages will be added. It's a data structure for handling
        large-scale biological data, like gene expression matrices, commonly used in bioinformatics.

    predicted_ages : numpy.ndarray or list
        An array or list of predicted ages corresponding to the samples in the AnnData object. The length
        of this array should match the number of samples in `adata`.

    clock_name : str
        The name of the aging clock used to generate the predicted ages. This name will be used
        as the column name in `adata.obs`.

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
    >>> add_pred_ages_adata(adata, predicted_ages, 'horvath2013', logger)
    >>> adata.obs['horvath2013']
    0    25
    1    30
    2    35
    3    40
    4    45
    Name: horvath2013, dtype: int64

    """
    adata.obs[clock_name] = predicted_ages


@progress("Return adata to original size")
def filter_missing_features(
    adata: anndata.AnnData,
    logger,
    indent_level: int = 2,
) -> anndata.AnnData:
    """
    Returns adata with original features.

    This function checks for variables that have 100% of samples with missing features, and removes them
    from the adata object. It is useful for returning the adata in the original state.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object that will be filtered.

    logger : Logger
        A logger object for logging the progress or relevant information during the operation.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    anndata.AnnData
        The filtered adata in which all features appear in at least one sample.

    Notes
    -----
    During filtering, the adata object is filtered based on a column called "percent_na" which should be less
    than 1. If the column is not present, an error will appear.

    Examples
    --------
    >>> adata = anndata.AnnData(np.random.rand(5, 10))
    >>> adata = filter_missing_features(adata, logger)

    """
    n_missing_features = sum(adata.var["percent_na"] == 1)
    if n_missing_features > 0:
        logger.info(
            f"Removing {n_missing_features} added features",
            indent_level=indent_level + 1,
        )
        adata = adata[:, adata.var["percent_na"] < 1].copy()
    else:
        logger.info(
            "No missing features, so adata size did not change",
            indent_level=indent_level + 1,
        )

    return adata


@progress("Add clock metadata to adata.uns")
def add_clock_metadata_adata(
    adata: anndata.AnnData,
    clock_name: str,
    dir: str,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Add specific clock metadata to the `uns` attribute of an AnnData object.

    This function enriches an AnnData object with metadata related to a specific aging clock.
    The metadata is stored under a key named after the clock in the AnnData's `uns` attribute, which is
    a dictionary-like structure used for storing unstructured data. This addition aids in keeping track
    of the clock-specific information, such as model parameters or references, which can be crucial for
    subsequent analyses and reproducibility.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to which the clock metadata will be added. AnnData is widely used in
        bioinformatics for storing large-scale biological data like gene expression matrices.

    clock_name : str
        The name of the aging clock. The metadata associated with this clock will be added to
        the AnnData object.

    dir: str
        The directory to deposit the downloaded file. Defaults to "pyaging_data".

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
    The `uns` attribute in AnnData is a flexible slot for storing miscellaneous data. Adding clock metadata
    here ensures that all relevant information about the aging clocks used in the analysis is
    preserved alongside the dataset.

    Examples
    --------
    >>> adata = anndata.AnnData(np.random.rand(5, 10))
    >>> add_clock_metadata_adata(adata, 'horvath2013', 'pyaging_data', logger)
    >>> adata.uns['horvath2013_metadata']
    {'description': 'Horvath’s 2013 epigenetic clock', 'reference': 'Horvath, S. (2013)'}

    """

    # Load all clocks metadata
    url = f"https://pyaging.s3.amazonaws.com/clocks/metadata/all_clock_metadata.pt"
    download(url, dir, logger, indent_level=indent_level)
    all_clock_metadata = torch.load(f"{dir}/all_clock_metadata.pt")

    # Add clock metadata to adata.uns
    adata.uns[f"{clock_name}_metadata"] = all_clock_metadata[clock_name]


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

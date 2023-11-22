import os
import torch
import numpy as np
import pandas as pd
import anndata
import ntpath
from urllib.request import urlretrieve
from ..models import *
from ..utils import progress
from ..logger import LoggerManager, main_tqdm
from ._preprocessing import *
from ._postprocessing import *
from ..data import *

@progress("Load clock")
def load_clock(clock_name, logger, indent_level: int = 2):
    """
    Load the clock dictionary from the specified file.

    Args:
    - clock_name: Name of the clock to be loaded.
    - logger: Logger for logging progress.

    Returns:
    - Tuple containing features, weight dictionary, preprocessing, postprocessing, and the entire clock dictionary.
    """

    url = f"https://pyaging.s3.amazonaws.com/clocks/weights/{clock_name}.pt"
    download(url, logger, indent_level=2)
        
    # Define the path to the clock weights file
    weights_path = os.path.join("./pyaging_data", f"{clock_name}.pt")

    # Load the clock dictionary from the file
    clock_dict = torch.load(weights_path)

    # Extract relevant information from the clock dictionary
    features = clock_dict["features"]
    weight_dict = clock_dict["weight_dict"]
    preprocessing = clock_dict.get("preprocessing", None)
    postprocessing = clock_dict.get("postprocessing", None)

    return features, weight_dict, preprocessing, postprocessing, clock_dict

@progress("Load all clock metadata")
def load_clock_metadata(logger, indent_level: int = 2) -> dict:
    """
    Loads the metadata of all available clocks.

    Args:
    - logger: Logger object for logging messages.

    Returns:
    - pandas DataFrame with genome metadata.
    """
    file_id = '1w4aR_Z6fY4HAWFk1seYf6ELbZYb3GSmZ'
    url = f"https://drive.google.com/uc?id={file_id}"
    dir="./pyaging_data"
    file_path =  'all_clock_metadata.pt'
    file_path = os.path.join(dir, file_path)
    
    if os.path.exists(file_path):
        logger.info(f'Data found in {file_path}', indent_level=3)
    else:
        if not os.path.exists(dir):
            os.mkdir("pyaging_data")
        logger.info(f"Downloading data to {file_path}", indent_level=3)
        logger.indent_level = 3
        urlretrieve(url, file_path, reporthook=logger.request_report_hook)

    # Read data
    all_clock_metadata = torch.load("./pyaging_data/all_clock_metadata.pt")
    return all_clock_metadata


@progress("Check features in adata")
def check_features_in_adata(adata, features, logger, indent_level: int = 2):
    """
    Check for missing features in adata.var_names and add them with default value 0 if not present.

    Args:
    - adata: AnnData object to be checked.
    - features: List of features to check for in the AnnData object.
    - logger: Logger for logging progress.

    Returns:
    - Updated AnnData object with missing features added.
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

    # Log and add missing features if any
    if missing_features:
        logger.warning(
            f"{num_missing_features} out of {total_features} features "
            f"({percent_missing:.2f}%) are missing and will be "
            f"added with default value 0: {missing_features[:np.min([3, num_missing_features])]}, etc.",
            indent_level=3,
        )

        # Create an empty AnnData object for missing features
        adata_empty = anndata.AnnData(
            np.zeros((adata.n_obs, num_missing_features)),
            obs=adata.obs,
            var=pd.DataFrame(index=missing_features),
        )

        # Concatenate original adata with the empty adata
        adata = anndata.concat([adata, adata_empty], axis=1)
        logger.info(
            f"Expanded adata with {num_missing_features} missing features.",
            indent_level=3,
        )
    else:
        logger.info("All features are present in adata.var_names.", indent_level=2)

    return adata


@progress("Initialize model")
def initialize_model(clock_name, features, weight_dict, logger, indent_level: int = 2):
    """
    Initialize the clock model based on model type and features.

    Args:
    - clock_name: Name of the clock model to be initialized.
    - features: List of features to be used in the model.
    - weight_dict: Dictionary containing the weights for the model.
    - logger: Logger object for logging information.

    Returns:
    - Initialized model.
    """
    # Model selection based on clock name
    if clock_name in [
        "horvath2013",
        "skinandblood",
        "hannum",
        "phenoage",
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
    ]:
        model = LinearModel(len(features))
        model.load_state_dict(weight_dict)
        model.eval()
        return model
    elif clock_name in [
        "pchorvath2013",
        "pcskinandblood",
        "pchannum",
        "pcphenoage",
        "pcdnamtl",
    ]:
        model = PCLinearModel(len(features), pc_dim=weight_dict["rotation"].shape[1])
        model.load_state_dict(weight_dict)
        model.eval()
        return model
    elif clock_name == "pcgrimage":
        model = PCGrimAge(
            sum(["cg" in feature for feature in features]),
            pc_dim=weight_dict["rotation"].shape[1],
            comp_dims=[
                weight_dict[f"step1_layers.{i}.weight"].shape[1]
                for i in range(weight_dict["step2.weight"].shape[1] - 2)
            ],
        )
        model.load_state_dict(weight_dict)
        model.eval()
        return model
    elif clock_name == "altumage":
        model = AltumAge()
        model.load_state_dict(weight_dict)
        model.eval()
        return model
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
        model.load_state_dict(weight_dict)
        model.eval()
        return model
    else:
        raise ValueError(f"Clock '{clock_name}' is not supported.")


@progress("Preprocess data")
def preprocess_data(preprocessing, data, clock_dict, logger, indent_level: int = 2):
    """
    Apply preprocessing steps to the data.

    Args:
    - preprocessing: String specifying the preprocessing method.
    - data: Data to be preprocessed.
    - clock_dict: Dictionary containing clock-related information.
    - logger: Logger object for logging information.

    Returns:
    - Preprocessed data.
    """
    logger.info(f"Preprocessing data with function {preprocessing}", indent_level=3)
    # Apply specified preprocessing method
    if preprocessing == "scale":
        data = scale(data, clock_dict["preprocessing_helper"])
    elif preprocessing == "log1p":
        data = torch.log1p(data)
    elif preprocessing == "binarize":
        data = binarize(data)
    return data


@progress("Postprocess data")
def postprocess_data(postprocessing, data, clock_dict, logger, indent_level: int = 2):
    """
    Apply postprocessing steps to the data.

    Args:
    - postprocessing: String specifying the postprocessing method.
    - data: Data to be postprocessed.
    - clock_dict: Dictionary containing clock-related information.
    - logger: Logger object for logging information.

    Returns:
    - Postprocessed data.
    """
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
    return data


@progress("Predict ages with model")
def predict_ages_with_model(model, data, logger, indent_level: int = 2):
    """
    Use the model to predict ages from the data.

    Args:
    - model: The model to be used for prediction.
    - data: Data on which age prediction is to be performed.
    - logger: Logger object for logging information.

    Returns:
    - Predicted ages.
    """
    return model(data)


@progress("Convert tensor to numpy array")
def convert_tensor_to_numpy_array(tensor, logger, indent_level: int = 2):
    """
    Convert a PyTorch tensor to a Numpy array.

    Args:
    - tensor: PyTorch tensor to be converted.
    - logger: Logger object for logging information.

    Returns:
    - Numpy array equivalent of the input tensor.
    """
    return tensor.detach().numpy().flatten()


@progress("Convert adata.X to torch.tensor and filter features")
def convert_adata_to_tensor_and_filter_features(adata, features, logger, indent_level: int = 2):
    """
    Convert an AnnData object to a PyTorch tensor.

    Args:
    - adata: AnnData object to be converted.
    - features: List of features to filter in the AnnData object.
    - logger: Logger object for logging information.

    Returns:
    - PyTorch tensor of filtered data.
    """
    return torch.tensor(adata[:, features].X, dtype=torch.float32)


@progress("Add predicted ages to adata")
def add_pred_ages_adata(adata, predicted_ages, clock_name, logger, indent_level: int = 2):
    """
    Add predicted ages to adata.obs.

    Args:
    - adata: AnnData object to be updated.
    - predicted_ages: Predicted ages to be added.
    - clock_name: Name of the clock used for prediction.
    - logger: Logger object for logging information.
    """
    adata.obs[clock_name] = predicted_ages


@progress("Add clock metadata to adata.uns")
def add_clock_metadata_adata(adata, clock_name, all_clock_metadata, logger, indent_level: int = 2):
    """
    Add clock metadata to adata.

    Args:
    - adata: AnnData object to be updated.
    - clock_name: Name of the clock used for prediction.
    - all_clock_metadata: Dictionary containing clock-related information.
    - logger: Logger object for logging information.
    """
    adata.uns[f"{clock_name}_metadata"] = all_clock_metadata[clock_name]


@progress("Set PyTorch device")
def set_torch_device(logger, indent_level: int = 1):
    """
    Set the PyTorch device for computation.

    Args:
    - logger: Logger object for logging information.

    Returns:
    - Device object (either CPU or CUDA).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}", indent_level=2)
    return device


def predict_age(adata, clock_names="horvath1"):
    """
    Predicts the biological age of samples in an AnnData object based on specified DNA methylation clocks.

    Args:
    - adata: AnnData object containing methylation data.
    - clock_names: List or single name of clocks for prediction.

    Returns:
    - Updated AnnData object with predicted ages.
    """
    logger = LoggerManager.gen_logger("predict_age")
    logger.first_info("Starting predict_age function")

    # Ensure clock_names is a list
    if isinstance(clock_names, str):
        clock_names = [clock_names]

    # Set device for PyTorch operations
    device = set_torch_device(logger)

    for clock_name in clock_names:
        logger.info(f"Processing clock: {clock_name}", indent_level=1)

        # Load and prepare the clock
        clock_name = clock_name.lower()
        features, weight_dict, preprocessing, postprocessing, clock_dict = load_clock(
            clock_name, logger, indent_level=2
        )
        
        # Check and update adata for missing features
        adata_expanded = check_features_in_adata(adata, features, logger, indent_level=2)

        # Convert adata to tensor and filter features
        x_tensor = convert_adata_to_tensor_and_filter_features(
            adata_expanded, features, logger, indent_level=2
        )

        # Apply preprocessing if specified
        if preprocessing:
            x_tensor = preprocess_data(preprocessing, x_tensor, clock_dict, logger, indent_level=2)

        # Move tensor to the appropriate device
        x_tensor = x_tensor.to(device)

        # Initialize and configure the model
        clock_model = initialize_model(clock_name, features, weight_dict, logger, indent_level=2)
        clock_model = clock_model.to(device)  # Move model to the appropriate device

        # Perform age prediction using the model
        predicted_ages_tensor = predict_ages_with_model(clock_model, x_tensor, logger, indent_level=2)

        # Convert adata tensor to numpy array
        predicted_ages = convert_tensor_to_numpy_array(
            predicted_ages_tensor.cpu(), logger, indent_level=2
        )

        # Apply postprocessing if specified
        if postprocessing:
            predicted_ages = postprocess_data(
                postprocessing, predicted_ages, clock_dict, logger, indent_level=2
            )

        # Add predicted ages to adata
        add_pred_ages_adata(adata, predicted_ages, clock_name, logger, indent_level=2)
        
        # Load all clocks metadata
        all_clock_metadata = load_clock_metadata(logger, indent_level=2)

        # Add clock metadata to adata object
        add_clock_metadata_adata(adata, clock_name, all_clock_metadata, logger, indent_level=2)

    logger.done()
    return adata

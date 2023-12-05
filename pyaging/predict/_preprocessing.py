import numpy as np


def scale(data, scaler):
    """
    Scales the input data using the provided scaler.
    """
    # Apply the scaling transformation to the NumPy array
    data_scaled = scaler.transform(data)

    return data_scaled


def binarize(data):
    """
    Binarizes an array based on the median of each row, excluding zeros.
    """

    # Create a mask for non-zero elements
    non_zero_mask = data != 0

    # Apply mask, calculate median for each row, and change data
    medians = np.zeros(data.shape[0])
    for i, row in enumerate(data):
        non_zero_elements = row[non_zero_mask[i]]
        data[i] = data[i] > np.median(non_zero_elements)

    return data

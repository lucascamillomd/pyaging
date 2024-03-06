import numpy as np


def scale(x, scaler):
    """
    Scales the input data using the provided scaler.
    """
    # Apply the scaling transformation to the NumPy array
    x_scaled = scaler.transform(x)

    return x_scaled


def scale_with_gold_standard(x, column_means, column_stds):
    """
    Scales the input data per column given means and standard deviations.
    """
    # Ensure column_stds is a numpy array
    column_stds = np.array(column_stds)

    # Avoid division by zero in case of a column with constant value
    column_stds[np.abs(column_stds) < 10e-10] = 1

    x_scaled = (x - column_means) / column_stds
    return x_scaled


def scale_row(x, x_overlap):
    """
    Scales the input data per row with mean 0 and std 1.
    """
    row_means = np.mean(x_overlap, axis=1, keepdims=True)
    row_stds = np.std(x_overlap, axis=1, keepdims=True)

    # Avoid division by zero in case of a row with constant value
    row_stds[row_stds == 0] = 1

    x_scaled = (x - row_means) / row_stds
    return x_scaled


def binarize(x):
    """
    Binarizes an array based on the median of each row, excluding zeros.
    """

    # Create a mask for non-zero elements
    non_zero_mask = x != 0

    # Apply mask, calculate median for each row, and change data
    for i, row in enumerate(x):
        non_zero_elements = row[non_zero_mask[i]]
        x[i] = x[i] > np.median(non_zero_elements)

    return x


def tpm_norm_log1p(x, lengths):
    """
    Normalize an array of counts to TPM (Transcripts Per Million) then
    transforms with log1p.
    """
    # Normalize by length
    tpm = 1000 * (x / lengths)

    # Scale to TPM (Transcripts Per Million)
    tpm = 1e6 * (tpm / tpm.sum(axis=1, keepdims=True))

    tpm_log1p = np.log1p(tpm)

    return tpm_log1p


def quantile_normalize_with_gold_standard(x, gold_standard_means):
    """
    Apply quantile normalization on x using gold standard means.
    """
    # Create a copy of x to avoid modifying a view
    x_normalized = x.copy()

    # Sort the gold standard means
    sorted_gold_standard = np.sort(gold_standard_means)

    # Iterate through each row in x_normalized
    for i in range(x_normalized.shape[0]):
        # Sort the row data and store the original indices
        sorted_indices = np.argsort(x_normalized[i, :])
        sorted_data = x_normalized[i, sorted_indices]

        # Map the sorted data to their quantile values in the gold standard
        quantile_indices = np.round(np.linspace(0, len(sorted_gold_standard) - 1, len(sorted_data))).astype(int)
        normalized_data = sorted_gold_standard[quantile_indices]

        # Re-order the normalized data to the original order
        original_order_indices = np.argsort(sorted_indices)
        x_normalized[i, :] = normalized_data[original_order_indices]

    return x_normalized

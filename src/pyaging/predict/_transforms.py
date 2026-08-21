import numpy as np
import torch


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


# Lowest C-reactive protein ``PhenoAge`` will take the natural log of, in mg/dL. Its
# clamp is the only CRP transform that needs a floor, and the value matches the registry
# floor in data/feature_ranges.json so that anything the clamp touches has already been
# reported by check_feature_ranges.
PHENOAGE_CRP_FLOOR_MG_DL = 0.01


def crp_index(features):
    """Locate the C-reactive protein column, or explain why there is not one.

    pyaging 0.5.0 renamed this feature from ``log_crp`` and moved the log into the
    clock, so 0.5.0 code cannot run a weight file built before it.

    Raises
    ------
    ValueError
        If ``features`` has no ``c_reactive_protein`` entry.
    """
    try:
        return features.index("c_reactive_protein")
    except ValueError:
        pass
    legacy = " Its features call it 'log_crp', the name pyaging used before 0.5.0." if "log_crp" in features else ""
    raise ValueError(
        "This clock has no 'c_reactive_protein' feature, so its C-reactive protein transform "
        f"cannot be applied.{legacy} Weights built before pyaging 0.5.0 do not work with "
        "pyaging 0.5.0 or later. Unset PYAGING_DATA_REVISION, or pin it to v0.5.0 or a later "
        "tag, to download weights that match the installed version; pin pyaging itself to "
        "<0.5.0 to keep using the older weights."
    )


def log1p_crp(features, x):
    """Apply BioAge's ``lncrp`` transform to the C-reactive protein column alone.

    BioAge fits ``lncrp`` against ``log1p(CRP in mg/dL)``, not the natural log
    ``PhenoAge`` uses, so every clock ported from that package shares this
    transform; users supply the raw measurement so one column can feed clocks
    that log it differently.

    Notes
    -----
    CRP is not floored. ``log1p`` is finite everywhere it is defined, and
    ``log1p(0) == 0`` is a point BioAge's own fitted preimage includes, so a
    below-detection reading coded as ``0``, a constant imputer, or an absent
    column all pass through as the reference scores them. Only a value at or
    below ``-1`` mg/dL is outside the domain, and that is not a measurement;
    those samples become ``NaN`` so they surface in the output instead of
    carrying a value the clock made up for them.
    """
    index = crp_index(features)
    crp = x[:, index : index + 1]
    crp = torch.where(crp > -1, crp, torch.nan)
    return torch.cat([x[:, :index], torch.log1p(crp), x[:, index + 1 :]], dim=1)

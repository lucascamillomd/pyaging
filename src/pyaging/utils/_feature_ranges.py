"""Package-wide ground truth for feature units and plausibility ranges."""

import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np

_UNBOUNDED = {"unit": None, "low": None, "high": None}


@lru_cache(maxsize=1)
def load_feature_range_registry() -> dict:
    """
    Load and cache the feature range registry shipped with the package.

    Returns
    -------
    dict
        The parsed registry, with keys ``schema_version``, ``modality_defaults``,
        and ``features``. The same object is returned on every call, so callers
        must not mutate it.
    """
    source = resources.files("pyaging.data").joinpath("feature_ranges.json")
    return json.loads(source.read_text(encoding="utf-8"))


def resolve_feature_bounds(features, data_type, feature_units=None):
    """Resolve each feature's unit and bounds as parallel sequences.

    Same resolution rules as :func:`resolve_feature_ranges`, which is written in
    terms of this. The largest clocks carry 453,152 features, and a dict per
    feature is a measurable share of a prediction's run time, so callers that
    only need to compare numbers take the arrays instead.

    Parameters
    ----------
    features : list of str
        Feature names, in the order the caller uses them.

    data_type : str or None
        The clock's modality.

    feature_units : list of str or str, optional
        Units that override the registry unit, as in :func:`resolve_feature_ranges`.

    Returns
    -------
    tuple of (list of str or None, numpy.ndarray, numpy.ndarray)
        The resolved units, and the lower and upper bounds as float arrays whose
        unbounded sides are ``-inf`` and ``inf``.

    Raises
    ------
    ValueError
        If ``feature_units`` is a list whose length differs from ``features``.
    """
    features = list(features)
    if feature_units is None or isinstance(feature_units, str):
        overrides = [feature_units] * len(features)
    elif len(feature_units) != len(features):
        raise ValueError(f"feature_units has {len(feature_units)} entries but there are {len(features)} features")
    else:
        overrides = list(feature_units)

    registry = load_feature_range_registry()
    entries = registry["features"]
    default = registry["modality_defaults"].get(data_type, _UNBOUNDED)
    default_low = -math.inf if default["low"] is None else float(default["low"])
    default_high = math.inf if default["high"] is None else float(default["high"])

    units = [override or default["unit"] for override in overrides]
    low = np.full(len(features), default_low)
    high = np.full(len(features), default_high)
    # Most features on a methylation clock are CpGs with no entry of their own, so
    # only pay for the per-feature lookup when the registry has something to say.
    if entries.keys() & set(features):
        for index, feature in enumerate(features):
            entry = entries.get(feature)
            if entry is None:
                continue
            units[index] = overrides[index] or entry["unit"]
            low[index] = -math.inf if entry["low"] is None else float(entry["low"])
            high[index] = math.inf if entry["high"] is None else float(entry["high"])
    return units, low, high


def resolve_feature_ranges(features, data_type, feature_units=None) -> list[dict]:
    """
    Resolve each feature to its unit and plausibility bounds.

    A per-feature registry entry wins over the modality default for the clock's
    ``data_type``; features in neither are unbounded.

    Parameters
    ----------
    features : list of str
        Feature names, in the order the caller uses them.

    data_type : str or None
        The clock's modality, e.g. ``"DNA methylation"`` or ``"clinical biomarkers"``.
        A modality with no default leaves unlisted features unbounded.

    feature_units : list of str or str, optional
        Units that override the registry unit. A single string applies to every
        feature; a list must have one entry per feature, and a ``None`` entry
        falls back to the registry unit for that feature.

    Returns
    -------
    list of dict
        One record per feature, in the order of ``features``, each with keys
        ``feature``, ``unit``, ``low``, and ``high``. Unbounded sides are
        ``-inf`` and ``inf``.

    Raises
    ------
    ValueError
        If ``feature_units`` is a list whose length differs from ``features``.
    """
    units, low, high = resolve_feature_bounds(features, data_type, feature_units)
    return [
        {"feature": feature, "unit": unit, "low": float(lower), "high": float(upper)}
        for feature, unit, lower, upper in zip(features, units, low, high, strict=True)
    ]


def get_feature_ranges(clock_name: str):
    """
    Return a clock's feature units and normal ranges as a DataFrame.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock. Case-insensitive.

    Returns
    -------
    pandas.DataFrame
        One row per clock feature, with columns ``feature``, ``unit``, ``low``,
        and ``high``.

    Raises
    ------
    ValueError
        If the loaded clock carries no feature list.

    Examples
    --------
    >>> get_feature_ranges("phenoage")  # doctest: +SKIP
    """
    import pandas as pd

    from ..predict import load_clock

    model = load_clock(clock_name, verbose=False)
    if model.features is None:
        raise ValueError(f"clock {clock_name!r} has no feature list, so its feature ranges cannot be resolved")
    records = resolve_feature_ranges(
        model.features,
        model.metadata.get("data_type"),
        getattr(model, "feature_units", None),
    )
    return pd.DataFrame.from_records(records)

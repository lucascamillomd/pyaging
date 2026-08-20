"""Package-wide ground truth for feature units and plausibility ranges."""

import json
import math
from functools import lru_cache
from importlib import resources

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


def _as_records(features, data_type, units):
    registry = load_feature_range_registry()
    per_feature = registry["features"]
    default = registry["modality_defaults"].get(data_type, _UNBOUNDED)

    for index, feature in enumerate(features):
        entry = per_feature.get(feature, default)
        unit = units[index] if units is not None else entry["unit"]
        yield {
            "feature": feature,
            "unit": unit,
            "low": -math.inf if entry["low"] is None else float(entry["low"]),
            "high": math.inf if entry["high"] is None else float(entry["high"]),
        }


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
        feature; a list must have one entry per feature.

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
    features = list(features)
    if isinstance(feature_units, str):
        feature_units = [feature_units] * len(features)
    if feature_units is not None and len(feature_units) != len(features):
        raise ValueError(f"feature_units has {len(feature_units)} entries but there are {len(features)} features")
    return list(_as_records(features, data_type, feature_units))


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

    Examples
    --------
    >>> get_feature_ranges("phenoage")  # doctest: +SKIP
    """
    import pandas as pd

    from ..predict import load_clock

    model = load_clock(clock_name, verbose=False)
    records = resolve_feature_ranges(
        model.features,
        model.metadata.get("data_type"),
        getattr(model, "feature_units", None),
    )
    return pd.DataFrame.from_records(records)

"""metadata_cols accepts a single name or any list-like of names."""

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess import df_to_adata

FRAME = pd.DataFrame(
    {"cg1": [0.1, 0.2], "sex": ["F", "M"], "tissue": ["blood", "saliva"]},
    index=["s1", "s2"],
)


@pytest.mark.parametrize(
    ("metadata_cols", "expected"),
    [
        ("sex", ["sex"]),
        (("sex", "tissue"), ["sex", "tissue"]),
        (pd.Index(["tissue", "sex"]), ["tissue", "sex"]),
        (np.array(["sex", "tissue"]), ["sex", "tissue"]),
        (["sex", "tissue"], ["sex", "tissue"]),
    ],
)
def test_metadata_cols_accepts_a_name_or_list_like(metadata_cols, expected):
    adata = df_to_adata(FRAME[["cg1", *expected]], metadata_cols=metadata_cols, verbose=False)
    assert adata.obs.columns.tolist() == expected
    assert adata.var_names.tolist() == ["cg1"]
    assert adata.obs["sex"].tolist() == ["F", "M"]


@pytest.mark.parametrize("metadata_cols", ["age", ("sex", "age", "batch")])
def test_unknown_metadata_cols_are_named(metadata_cols):
    with pytest.raises(KeyError, match="Metadata columns not found in df: \\['age'"):
        df_to_adata(FRAME, metadata_cols=metadata_cols, verbose=False)

"""OrganAge parity against the authors' R scoring expressions and fold-one CSVs."""

import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "organage"


def test_organage_scores_author_adipose_fold_one():
    """Catch wrong fold, protein ordering, intercept, scaling, or missingness."""
    assert importlib.util.find_spec("pyaging.models._organage") is not None, "OrganAge implementation is missing"
    module = importlib.import_module("pyaging.models._organage")
    model = module.OrganAge()
    model.base_model = torch.nn.Linear(5, 1, dtype=torch.float64)
    with torch.no_grad():
        model.base_model.weight.copy_(
            torch.tensor(
                [[1.27774045748466, 0.359697350178859, 2.81732857203834, -0.801675705736661, 0.963173154759831]],
                dtype=torch.float64,
            )
        )
        model.base_model.bias.fill_(57.1044228722264)
    values = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [-1, 0, 1, 2, -2]], dtype=torch.float64)
    np.testing.assert_allclose(
        model(values).detach().numpy().ravel(),
        [57.1044228722264, 61.72068670095143, 55.114313265787],
        rtol=0,
        atol=1e-10,
    )
    assert torch.isnan(model(torch.full((1, 5), float("nan"), dtype=torch.float64))).all()


def test_builder_preserves_fold_one_protein_symbols_and_license(tmp_path, monkeypatch):
    """Catch a fold mix-up, lost protein symbols, or dropped author license."""
    source = tmp_path
    builder_path = Path(__file__).resolve().parents[2] / "clocks" / "build_organage.py"
    assert builder_path.is_file(), "Reproducible OrganAge builder is missing"
    spec = importlib.util.spec_from_file_location("build_organage", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    relative = "data/output_Python/instance_0/chronological_models/Adipose_coefs_GTEx_4x_FC.csv"
    coefficient_file = source / relative
    coefficient_file.parent.mkdir(parents=True)
    coefficient_file.write_text(
        '"Intercept","ADIPOQ","HLA-DRA","NTproBNP","LEP","PLIN1"\n'
        "57.1044228722264,1.27774045748466,0.359697350178859,2.81732857203834,"
        "-0.801675705736661,0.963173154759831\n" + "0,0,0,0,0,0\n" * 4
    )
    (source / "LICENSE").write_bytes((FIXTURES / "LICENSE.author").read_bytes())
    monkeypatch.setitem(builder.SOURCE_SHA256, relative, hashlib.sha256(coefficient_file.read_bytes()).hexdigest())
    model = builder.build_clock("organagechronologicalolink3000adipose", source_dir=source)
    assert model.features == ["ADIPOQ", "HLA-DRA", "NTproBNP", "LEP", "PLIN1"]
    x = torch.tensor([[0, 0, 0, 0, 0], [-1, 0, 1, 2, -2]], dtype=torch.float64)
    np.testing.assert_allclose(
        model(x).detach().numpy().ravel(), [57.1044228722264, 55.114313265787], rtol=0, atol=1e-10
    )
    assert model.metadata["research_only"] is True
    assert "Non-Commercial" in model.license_text
    coefficient_file.write_text(coefficient_file.read_text().replace("57.1044228722264", "1"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_clock("organagechronologicalolink3000adipose", source_dir=source)


def _author_fixture_model(name):
    """Configure inference from original author coefficients, without a network."""
    from pyaging.models._organage import OrganAge

    with gzip.open(FIXTURES / "author_fold_one.json.gz", "rt") as stream:
        parameters = json.load(stream)[name]
    model = OrganAge()
    model.features = parameters["features"]
    model.base_model_features = list(model.features)
    model.feature_units = ["NPX"] * len(model.features)
    model.reference_values = [0.0] * len(model.features)
    model.base_model = torch.nn.Linear(len(model.features), 1, dtype=torch.float64)
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([parameters["coefficients"]], dtype=torch.float64))
        model.base_model.bias.fill_(parameters["intercept"])
    model.metadata.update(clock_name=name, data_type="proteomics", research_only=True)
    return model.eval()


@pytest.fixture(scope="module")
def oracle():
    import pandas as pd

    return pd.read_csv(FIXTURES / "expected_predictions.csv")


@pytest.fixture(scope="module")
def npx():
    import pandas as pd

    return pd.read_csv(FIXTURES / "input_npx.csv.gz", index_col=0)


@pytest.mark.parametrize("case", ["complete", "omitted_proteins", "present_na"])
@pytest.mark.parametrize("source", ["author_fixture", "local_artifact"])
def test_all_90_models_match_original_author_r(oracle, npx, case, source):
    """Catch wrong folds, protein alignments, altered input scale, or NA handling."""
    from tests.helpers import load_local_clock

    expected = oracle.loc[oracle.case.eq(case)]
    assert expected.clock.nunique() == 90
    for name, rows in expected.groupby("clock", sort=False):
        model = _author_fixture_model(name) if source == "author_fixture" else load_local_clock(name)
        frame = npx.copy()
        if case == "omitted_proteins":
            frame = frame.iloc[:, (np.arange(frame.shape[1]) + 1) % 3 != 0]
        if case == "present_na":
            frame.iloc[0, frame.columns.get_loc(model.features[0])] = np.nan
        frame = frame.reindex(columns=model.features, fill_value=0.0)
        with torch.inference_mode():
            actual = model(torch.tensor(frame.to_numpy(), dtype=torch.float64)).numpy().ravel()
        np.testing.assert_allclose(actual, rows.prediction, rtol=0, atol=1e-10, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("target", ["chronological", "mortality"])
@pytest.mark.parametrize("panel", ["1500", "3000"])
def test_predict_age_omission_and_batching_match_author_r(tmp_path, monkeypatch, oracle, npx, target, panel):
    """Exercise real alignment and batch scoring on absent and reordered proteins."""
    import anndata

    import pyaging as pya

    name = f"organage{target}olink{panel}conventional"
    model = _author_fixture_model(name)
    artifact = tmp_path / f"{name}.pt"
    torch.save(model, artifact)
    frame = npx.iloc[:, (np.arange(npx.shape[1]) + 1) % 3 != 0].iloc[:, ::-1]
    expected = oracle.loc[oracle.clock.eq(name) & oracle.case.eq("omitted_proteins"), "prediction"]
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    for batch_size in (1, 3, 1000):
        adata = anndata.AnnData(frame.copy())
        pya.pred.predict_age(adata, name, batch_size=batch_size, verbose=False)
        np.testing.assert_allclose(adata.obs[name], expected, rtol=0, atol=1e-10)
        assert set(adata.uns[f"{name}_missing_features"]) == set(model.features).difference(frame.columns)

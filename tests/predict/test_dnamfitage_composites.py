import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch

import pyaging as pya

pytestmark = pytest.mark.full_catalog
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "weights"


def _beta(feature):
    digest = int(hashlib.sha256(feature.encode()).hexdigest()[:8], 16)
    return 0.2 + (digest % 6000) / 10000


def _load_local(name):
    path = WEIGHTS_DIR / f"{name}.pt"
    assert path.is_file(), f"execute clocks/notebooks/{name}.ipynb"
    model = torch.load(path, weights_only=False, map_location="cpu")
    model.to(torch.float64).eval()
    return model


@pytest.mark.parametrize(
    ("clock_name", "expected_count", "expected"),
    [
        ("dnamfitagegait", 111, [2.5945144207221453, 1.9238271172398522]),
        ("dnamfitagegrip", 183, [42.75495020679189, 30.54383665460349]),
    ],
)
def test_merged_fitness_clocks_match_retired_sex_specific_oracles(clock_name, expected_count, expected):
    model = _load_local(clock_name)
    assert len(model.features) == expected_count
    frame = pd.DataFrame(
        [{feature: _beta(feature) for feature in model.features} for _ in range(2)],
        index=["male", "female"],
    )
    frame["female"] = [0.0, 1.0]
    values = torch.as_tensor(frame[model.features].to_numpy(), dtype=torch.float64)
    with torch.no_grad():
        predictions = model(values).ravel().tolist()
    assert predictions == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("clock_name", "expected"),
    [
        ("dnamfitagegait", 1.9238271172398522),
        ("dnamfitagegrip", 30.54383665460349),
    ],
)
def test_missing_female_uses_grimage_reference(monkeypatch, clock_name, expected):
    model = _load_local(clock_name)
    frame = pd.DataFrame([{feature: _beta(feature) for feature in model.features if feature != "female"}])
    monkeypatch.setattr(
        "pyaging.predict._pred_utils.download_clock_weights",
        lambda *args, **kwargs: str(WEIGHTS_DIR / f"{clock_name}.pt"),
    )
    adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
    pya.pred.predict_age(adata, clock_name, verbose=False)
    assert float(adata.obs[clock_name].iloc[0]) == pytest.approx(expected, abs=1e-10)
    assert adata.uns[f"{clock_name}_missing_features"] == ["female"]
    assert adata.uns[f"{clock_name}_percent_na"] == pytest.approx(100 / len(model.features))


def test_dnamfitage_embeds_grimage_and_matches_retired_two_stage_oracle():
    model = _load_local("dnamfitage")
    assay_features = list(
        dict.fromkeys(
            feature
            for component in (model.GrimAge, model.Gait, model.Grip, model.VO2Max)
            for feature in component.features
            if feature not in {"female", "age"}
        )
    )
    assert len(model.features) == 1343
    assert model.features == assay_features + ["female", "age"]
    assert model.features[-2:] == ["female", "age"]
    assert "grimage" not in model.features
    assert model.GrimAge.metadata["clock_name"] == "grimage"

    frame = pd.DataFrame(
        [{feature: _beta(feature) for feature in model.features} for _ in range(2)],
        index=["male", "female"],
    )
    frame["female"] = [0.0, 1.0]
    frame["age"] = 57.0
    values = torch.as_tensor(frame[model.features].to_numpy(), dtype=torch.float64)
    with torch.no_grad():
        predictions = model(values).ravel().tolist()
    assert predictions == pytest.approx([131.19968704579992, 133.55199017157764], abs=1e-10)


def test_dnamfitage_missing_age_and_female_use_grimage_references(monkeypatch):
    model = _load_local("dnamfitage")
    frame = pd.DataFrame([{feature: _beta(feature) for feature in model.features if feature not in {"female", "age"}}])
    monkeypatch.setattr(
        "pyaging.predict._pred_utils.download_clock_weights",
        lambda *args, **kwargs: str(WEIGHTS_DIR / "dnamfitage.pt"),
    )
    adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
    pya.pred.predict_age(adata, "dnamfitage", verbose=False)
    assert float(adata.obs["dnamfitage"].iloc[0]) == pytest.approx(137.34972213448359, abs=1e-10)
    assert adata.uns["dnamfitage_missing_features"] == ["female", "age"]

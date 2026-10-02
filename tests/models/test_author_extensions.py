"""Independent author-source checks for the 0.5.4 linear-clock additions."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyaging.models._author_extensions import DNAmPhenoAgeNonPRC, DNAmPhenoAgePRC, IntrinClock370
from pyaging.models._base_models import LinearModel

DATA = Path(__file__).resolve().parents[1] / "data/author_extensions"


def _configured(cls, features, weights, intercept=0.0):
    model = cls()
    model.features = list(features)
    model.base_model = LinearModel(len(features)).double()
    with torch.no_grad():
        model.base_model.linear.weight.copy_(torch.tensor(np.asarray(weights)[None], dtype=torch.float64))
        model.base_model.linear.bias.fill_(intercept)
    return model


def test_phenoage_partitions_match_original_r_functions():
    coefficients = pd.read_csv(DATA / "pheno_coefficients.csv")
    frame = pd.read_csv(DATA / "pheno_input.csv")
    expected = pd.read_csv(DATA / "pheno_expected.csv")
    missing_expected = pd.read_csv(DATA / "pheno_missing_expected.csv")
    for cls, mask, key, count in (
        (DNAmPhenoAgePRC, coefficients.PRC.eq(1), "PRC_PhenoAge", 55),
        (DNAmPhenoAgeNonPRC, ~coefficients.PRC.eq(1), "nonPRC_PhenoAge", 458),
    ):
        selected = coefficients.loc[mask]
        assert len(selected) == count
        model = _configured(cls, selected.CpG, selected.Weight)
        result = model(torch.tensor(frame[model.features].to_numpy(), dtype=torch.float64))
        np.testing.assert_allclose(result.detach().numpy().ravel(), expected[key], atol=1e-10, rtol=1e-12)
        missing = frame.drop(columns=coefficients.CpG.iloc[[0, 15, 17]])
        aligned = missing.reindex(columns=model.features, fill_value=0)
        result = model(torch.tensor(aligned.to_numpy(), dtype=torch.float64))
        np.testing.assert_allclose(result.detach().numpy().ravel(), missing_expected[key], atol=1e-10, rtol=1e-12)


def test_intrinclock370_matches_original_glmnet_predict_default():
    coefficients = pd.read_csv(DATA / "intrin_coefficients.csv").set_index("feature").coefficient
    selected = coefficients.drop("(Intercept)")
    selected = selected[selected != 0]
    assert len(selected) == 370
    model = _configured(IntrinClock370, selected.index, selected, coefficients["(Intercept)"])
    frame = pd.read_csv(DATA / "intrin_input.csv")
    expected = pd.read_csv(DATA / "intrin_expected.csv").predicted
    actual = model(torch.tensor(frame[model.features].to_numpy(), dtype=torch.float64))
    np.testing.assert_allclose(actual.detach().numpy().ravel(), expected, atol=1e-10, rtol=1e-12)


def test_intrinclock370_inverse_transform_covers_child_and_adult_branches():
    model = IntrinClock370()
    score = torch.tensor([[-2.0], [0.0], [1.5]], dtype=torch.float64)
    expected = torch.tensor([[21 * np.exp(-2) - 1], [20], [51.5]], dtype=torch.float64)
    torch.testing.assert_close(model.postprocess(score), expected)


def test_phenoage_partitions_skip_missing_betas_without_intercept():
    for cls in (DNAmPhenoAgePRC, DNAmPhenoAgeNonPRC):
        model = cls()
        model.base_model = LinearModel(3).double()
        with torch.no_grad():
            model.base_model.linear.weight.copy_(torch.tensor([[2.0, -3.0, 4.0]]))
            model.base_model.linear.bias.zero_()
        values = torch.tensor([[0.5, float("nan"), 0.25], [0.0, 0.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(model(values), torch.tensor([[2.0], [0.0]], dtype=torch.float64))
        assert torch.isnan(values[0, 1])

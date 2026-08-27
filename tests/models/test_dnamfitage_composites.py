import pytest
import torch

from pyaging.models import DNAmFitAgeGait, DNAmFitAgeGrip, LinearModel


def _linear(weight, bias):
    model = LinearModel(len(weight)).to(torch.float64)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([weight], dtype=torch.float64))
        model.linear.bias.copy_(torch.tensor([bias], dtype=torch.float64))
    return model


@pytest.fixture(params=[DNAmFitAgeGait, DNAmFitAgeGrip])
def gated_model(request):
    model = request.param().to(torch.float64)
    model.features = ["female_probe", "male_probe", "female"]
    model.female_model = _linear([2.0], 1.0)
    model.male_model = _linear([3.0], -1.0)
    model.female_feature_indices = torch.tensor([0])
    model.male_feature_indices = torch.tensor([1])
    model.female_reference_values = [2.0]
    model.male_reference_values = [4.0]
    model.female_index = 2
    model.reference_values = [float("nan"), float("nan"), 1.0]
    return model


def test_sex_gated_model_matches_both_branches_and_blends(gated_model):
    rows = torch.tensor(
        [[3.0, 4.0, 0.0], [3.0, 4.0, 1.0], [3.0, 4.0, 0.25]],
        dtype=torch.float64,
    )
    assert gated_model(rows).ravel().tolist() == pytest.approx([11.0, 7.0, 10.0])


def test_sex_gated_model_uses_branch_specific_references(gated_model):
    rows = torch.tensor(
        [[float("nan"), float("nan"), 0.0], [float("nan"), float("nan"), 1.0]],
        dtype=torch.float64,
    )
    assert gated_model(rows).ravel().tolist() == pytest.approx([11.0, 5.0])


def test_sex_gated_model_propagates_supplied_nan_female(gated_model):
    row = torch.tensor([[3.0, 4.0, float("nan")]], dtype=torch.float64)
    assert torch.isnan(gated_model(row)).all()

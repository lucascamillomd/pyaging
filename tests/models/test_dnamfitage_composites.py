import pytest
import torch

from pyaging.models import DNAmFitAge, DNAmFitAgeGait, DNAmFitAgeGrip, LinearModel


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


def test_dnamfitage_wires_embedded_components_and_blends_numeric_female():
    model = DNAmFitAge().to(torch.float64)
    model.features = ["gait", "grip", "vo2max", "grimage", "female", "age"]
    model.Gait = _linear([0.0], 2.0)
    model.Grip = _linear([0.0], 30.0)
    model.VO2Max = _linear([0.0], 45.0)
    model.GrimAge = _linear([0.0], 50.0)
    for component, reference in zip(
        (model.Gait, model.Grip, model.VO2Max, model.GrimAge),
        (11.0, 12.0, 13.0, 14.0),
        strict=True,
    ):
        component.reference_values = [reference]

    model.features_Gait = torch.tensor([0])
    model.features_Grip = torch.tensor([1])
    model.features_VO2Max = torch.tensor([2])
    model.features_GrimAge = torch.tensor([3])
    model.female_index = 4
    model.age_index = 5
    model.reference_values = [101.0, 102.0, 103.0, 104.0, 1.0, 65.0]
    model.base_model_f = _linear([1.0, 1.0, 1.0, 1.0], 0.0)
    model.base_model_m = _linear([1.0, 1.0, 1.0, 1.0], 0.0)

    rows = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 57.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 57.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 57.0],
        ],
        dtype=torch.float64,
    )
    female_expected = sum(
        (
            (45.0 - 46.825091) / -0.13620215,
            (30.0 - 39.857718) / -0.22074456,
            (2.0 - 2.508547) / -0.01245682,
            (50.0 - 7.978487) / 0.80928530,
        )
    )
    male_expected = sum(
        (
            (45.0 - 49.836389) / -0.141862925,
            (30.0 - 57.514016) / -0.253179827,
            (2.0 - 2.349080) / -0.009380061,
            (50.0 - 9.549733) / 0.835120557,
        )
    )
    assert model(rows).ravel().tolist() == pytest.approx(
        [male_expected, female_expected, (male_expected + female_expected) / 2]
    )

    component_input = model._component_input(
        torch.tensor(
            [[float("nan"), 0.0, 0.0, 0.0, 0.0, 57.0]],
            dtype=torch.float64,
        ),
        model.features_Gait,
        model.Gait,
    )
    assert component_input.item() == 11.0

"""Runtime regressions using small local models and unchanged clock formulas."""

import math

import pytest
import torch
from scipy.stats import rankdata

from pyaging.models import Pasta, PastaMouse, PCGrimAge, PhenoAge, Reg, epiTOC1, stemTOC
from pyaging.models._base_models import pyagingModel
from pyaging.predict._transforms import mortality_to_phenoage_saopaulo


def _linear(weights):
    model = torch.nn.Linear(len(weights), 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([weights], dtype=torch.float64))
    return model


@pytest.mark.parametrize("model_class", [Pasta, Reg, PastaMouse])
def test_rank_clock_context_reproduces_full_cohort_prediction(model_class):
    model = model_class()
    model.base_model = _linear([1.0, 10.0, 100.0])
    model.postprocess_dependencies = [0.0] if model_class is Reg else [1.0, 0.0]
    if model_class is PastaMouse:
        model.set_mouse_features(["ENSMUSG1", "ENSMUSG2", "ENSG1"], [float("nan")] * 3)
        values = [[1.0, 2.0], [100.0, 200.0]]
        expected = [276.0, 132.0]
    else:
        values = [[float("nan"), 1.0, 2.0], [100.0, 200.0, 300.0]]
        expected = [213.0, 321.0]
    x = torch.tensor(values, dtype=torch.float64)
    original_state = set(vars(model))
    assert model(x).flatten().tolist() == expected
    context = model.prepare_cohort_context(x)
    result = torch.cat([model.predict_with_cohort_context(row[None], context) for row in x])
    assert result.flatten().tolist() == expected
    assert set(vars(model)) == original_state


@pytest.mark.parametrize("references", [None, [5.0, 7.0, 100.0], [float("nan")] * 3])
def test_pastamouse_context_includes_inserted_reference_values(references):
    model = PastaMouse()
    model.set_mouse_features(["ENSMUSG1", "ENSG1", "ENSG2"], references)
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    # The expanded cohorts are [1,0,0;2,0,0], [1,7,100;2,7,100], or
    # [1,NaN,NaN;2,NaN,NaN]. Keep torch's existing lower-median rule.
    expected = 0.0 if references is None else (1.0 if math.isnan(references[0]) else 7.0)
    assert model.prepare_cohort_context(x).item() == expected


@pytest.mark.parametrize("model_class", [Pasta, Reg])
def test_rank_clock_context_preserves_all_missing_zero_fallback(model_class):
    x = torch.full((2, 3), float("nan"), dtype=torch.float64)
    assert model_class().prepare_cohort_context(x).item() == 0.0


@pytest.mark.parametrize("model_class", [Pasta, Reg])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("values", [[3.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0] * 8, []])
def test_rank_average_matches_scipy_for_ties_and_ordering(model_class, dtype, values):
    x = torch.tensor(values, dtype=dtype)
    expected = torch.tensor(rankdata(values, method="average"), dtype=dtype)
    torch.testing.assert_close(model_class._rank_average(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [Pasta, Reg])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rank_average_matches_scipy_for_random_values(model_class, dtype):
    x = torch.randint(-100, 100, (8113,), generator=torch.Generator().manual_seed(0)).to(dtype)
    expected = torch.tensor(rankdata(x.numpy(), method="average"), dtype=dtype)
    torch.testing.assert_close(model_class._rank_average(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [Pasta, Reg])
def test_rank_average_does_not_extract_a_python_scalar_per_feature(model_class):
    x = torch.arange(1024, dtype=torch.float64)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        model_class._rank_average(x)
    # Every tensor comparison used as a Python condition synchronizes CUDA.
    # Bound scalar extraction independently of feature count.
    scalar_extractions = sum(event.count for event in profile.key_averages() if event.key == "aten::item")
    assert scalar_extractions <= 1


@pytest.mark.parametrize("model_class", [stemTOC, epiTOC1])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_sentinel_only_samples_preserve_model_dtype_and_device(model_class, device):
    model = model_class()
    model.base_model = _linear([1.0]).to(device)
    x = torch.full((2, 3), -1.0, dtype=torch.float64, device=device)
    result = model(x)
    assert result.dtype == x.dtype
    assert result.device == x.device
    assert torch.isnan(result).all()


def test_pcgrimage_applies_its_reference_imputer_before_pca():
    # Configure the real forward with two CpGs and one PC, avoiding the large
    # production rotation allocation while keeping the component wiring.
    model = PCGrimAge.__new__(PCGrimAge)
    pyagingModel.__init__(model)
    model.reference_values = [0.2, 0.3, 0.0, 50.0]
    model.center = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64), requires_grad=False)
    model.rotation = torch.nn.Parameter(torch.ones(2, 1, dtype=torch.float64), requires_grad=False)
    for name in ["PCPACKYRS", "PCADM", "PCB2M", "PCCystatinC", "PCGDF15", "PCLeptin", "PCPAI1", "PCTIMP1"]:
        setattr(model, name, torch.nn.Identity())
        setattr(model, f"features_{name}", [0])
    model.base_model = _linear([1.0] * 10)
    x = torch.tensor([[float("nan"), 0.3, 0.0, 50.0], [0.2, 0.3, 0.0, 50.0]], dtype=torch.float64)
    assert model(x).flatten().tolist() == [54.0, 54.0]
    assert torch.isnan(x[0, 0])


def test_phenoage_extreme_finite_predictors_do_not_overflow():
    x = torch.tensor([[-50.0], [0.0], [1000.0]], dtype=torch.float64)
    result = PhenoAge().postprocess(x)
    assert torch.isfinite(result).all()
    assert result[:2].flatten().tolist() == pytest.approx([-412.07459164738657, 142.464309256512])
    assert result[2].item() - result[1].item() == pytest.approx(1000 / 0.090165)


def test_phenoage_matches_original_expression_at_regular_predictors():
    x = torch.linspace(-12.0, -4.0, 17, dtype=torch.float64)
    # Levine 2018 supplementary methods, p. 2: Gompertz gamma, not the
    # 0.0192 penalty used only for Cox variable selection on p. 1.
    hazard = torch.exp(x) * math.expm1(120 * 0.0076927) / 0.0076927
    mortality = 1 - torch.exp(-hazard)
    original = 141.50225 + torch.log(-0.00553 * torch.log1p(-mortality)) / 0.090165
    torch.testing.assert_close(PhenoAge().postprocess(x), original, atol=1e-10, rtol=0)


def test_phenoage_matches_independent_published_formula_reference():
    # Evaluated separately in R 4.5.3 from Levine 2018 SD1, pp. 1-2.
    # Fixed outputs avoid letting a shared, erroneous constant update both
    # sides of a test unnoticed. No weight downloads are required in CI.
    x = torch.tensor([-12.0, -9.0, -6.0], dtype=torch.float64)
    expected = torch.tensor([9.37497303957636, 42.64730709381026, 75.91964114804416], dtype=torch.float64)
    torch.testing.assert_close(PhenoAge().postprocess(x), expected, atol=1e-12, rtol=0)


def test_saopaulo_extreme_finite_predictors_do_not_overflow():
    x = torch.tensor([[-50.0], [0.0], [1000.0]], dtype=torch.float64)
    constants = [torch.tensor(value, dtype=x.dtype) for value in [-2.0, 4.0, -3.0, 0.1, 7.0]]
    result = mortality_to_phenoage_saopaulo(x, *constants)
    expected = torch.tensor([[-488.94534891891834], [11.054651081081644], [10011.05465108108]], dtype=x.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-10)


def test_saopaulo_matches_original_expression_at_regular_predictors():
    x = torch.linspace(-8.0, 0.0, 17, dtype=torch.float64)
    constants = [torch.tensor(value, dtype=x.dtype) for value in [-2.0, 4.0, -3.0, 0.1, 7.0]]
    mortality = 1 - torch.exp(-2.0 * torch.exp(x) / 4.0)
    original = torch.log(-3.0 * torch.log1p(-mortality)) / 0.1 + 7.0
    torch.testing.assert_close(mortality_to_phenoage_saopaulo(x, *constants), original, atol=1e-10, rtol=0)

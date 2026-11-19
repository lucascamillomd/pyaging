"""SystemsAge projects onto its DNAm PCs blockwise without changing the result."""

import pytest
import torch

from pyaging.models import SystemsAge, SystemsAgeBlood, SystemsAgeLung


def _model(model_class, n_cpgs=29, n_pcs=6):
    generator = torch.Generator().manual_seed(3)
    model = model_class()
    model.dnam_center = torch.rand(n_cpgs, generator=generator)
    # The published rotation is stored in float32 and scored in float64.
    model.dnam_rotation = torch.randn(n_cpgs, n_pcs, generator=generator)
    model.system_vector = torch.randn(n_pcs, 5, generator=generator, dtype=torch.float64)
    model.system_component_indices = [torch.tensor([0, 1]), torch.tensor([2, 3, 4])] + [torch.tensor([4])] * 9
    for indices in model.system_component_indices:
        module = torch.nn.Linear(len(indices), 1).double()
        torch.nn.init.normal_(module.weight, generator=generator)
        model.system_modules.append(module)
    model.predicted_age_model = torch.nn.Linear(n_pcs, 1).double()
    model.predicted_age_poly = [1.0, 0.9, 0.01]
    model.systems_pca_model = torch.nn.Linear(12, 1).double()
    model.transformation_coefs = torch.rand(12, 4, generator=generator, dtype=torch.float64) + 0.5
    return model


def _full_product_forward(model, x):
    """SystemsAge's forward with the whole rotation cast to the batch dtype at once."""
    x = model.preprocess(x)
    dnam_pcs = (x - model.dnam_center.to(x.dtype)) @ model.dnam_rotation.to(x.dtype)
    system_components = dnam_pcs @ model.system_vector.to(x.dtype)
    scores = [
        module(torch.index_select(system_components, 1, indices))
        for module, indices in zip(model.system_modules, model.system_component_indices, strict=True)
    ]
    poly = torch.tensor(model.predicted_age_poly, dtype=x.dtype)
    predicted_age = model.predicted_age_model(dnam_pcs).squeeze(-1)
    predicted_age = (predicted_age * poly[1] + predicted_age.pow(2) * poly[2] + poly[0]) / 12.0
    base_outputs = torch.cat([torch.cat(scores, dim=1), predicted_age.unsqueeze(-1)], dim=1)
    if model.prediction_index == -1:
        raw, index = model.systems_pca_model(base_outputs).squeeze(-1), model.transformation_coefs.shape[0] - 1
    else:
        raw, index = base_outputs[:, model.prediction_index], model.prediction_index
    coef = model.transformation_coefs.to(x.dtype)[index]
    return ((((raw - coef[0]) / coef[1]) * coef[3] + coef[2]) / 12.0).unsqueeze(-1)


@pytest.mark.parametrize("model_class", [SystemsAge, SystemsAgeBlood, SystemsAgeLung])
@pytest.mark.parametrize("block_rows", [None, 1, 4, 29, 1000])
def test_blockwise_projection_equals_the_full_matrix_product(model_class, block_rows):
    model = _model(model_class)
    if block_rows is not None:
        model.dnam_rotation_block_rows = block_rows
    x = torch.rand(7, 29, generator=torch.Generator().manual_seed(4), dtype=torch.float64).t().contiguous().t()
    with torch.inference_mode():
        result = model(x)
        expected = _full_product_forward(model, x)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_single_block_is_bit_identical_to_the_full_matrix_product():
    model = _model(SystemsAge)
    x = torch.rand(3, 29, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with torch.inference_mode():
        assert torch.equal(model(x), _full_product_forward(model, x))


def test_float32_batches_use_the_stored_rotation():
    model = _model(SystemsAgeBlood)
    model.dnam_rotation_block_rows = 4
    x = torch.rand(3, 29, generator=torch.Generator().manual_seed(6))
    model.system_vector = model.system_vector.float()
    model.transformation_coefs = model.transformation_coefs.float()
    for module in [*model.system_modules, model.predicted_age_model, model.systems_pca_model]:
        module.float()
    with torch.inference_mode():
        result = model(x)
        expected = _full_product_forward(model, x)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

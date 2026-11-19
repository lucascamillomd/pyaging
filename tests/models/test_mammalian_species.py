"""Species-aware mammalian clocks refuse samples without a species indicator."""

import numpy as np
import pytest
import torch

from pyaging.models import (
    Mammalian2,
    Mammalian3,
    MammalianBlood2,
    MammalianBlood3,
    MammalianSkin2,
    MammalianSkin3,
)

N_SPECIES = {
    Mammalian2: 1756,
    Mammalian3: 1707,
    MammalianBlood2: 1756,
    MammalianBlood3: 1707,
    MammalianSkin2: 1756,
    MammalianSkin3: 1707,
}
SPECIES_CLASSES = list(N_SPECIES)


def _model(model_class):
    model = model_class()
    model.base_model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([[0.5, -0.25]], dtype=torch.float64))
    n_species = N_SPECIES[model_class]
    # gestation, average maturity, -, maximum lifespan; distinct per species.
    anage = np.ones((n_species, 4))
    anage[:, 0] = np.linspace(0.05, 1.0, n_species)
    anage[:, 1] = np.linspace(0.5, 5.0, n_species)
    anage[:, 3] = np.linspace(2.0, 120.0, n_species)
    model.postprocess_dependencies = [anage]
    return model


def _input(model_class, species):
    """Two CpGs followed by the species indicators; ``species`` is one column index per row or None."""
    n_species = N_SPECIES[model_class]
    x = torch.zeros(len(species), 2 + n_species, dtype=torch.float64)
    x[:, :2] = torch.tensor([0.3, 0.6], dtype=torch.float64)
    for row, column in enumerate(species):
        if column is not None:
            x[row, 2 + column] = 1.0
    return x


@pytest.mark.parametrize("model_class", SPECIES_CLASSES)
def test_sample_without_a_species_indicator_raises(model_class):
    model = _model(model_class)
    with pytest.raises(ValueError, match="species") as error:
        model(_input(model_class, [5, None]))
    assert "set to 1" in str(error.value)


@pytest.mark.parametrize("model_class", SPECIES_CLASSES)
def test_one_hot_species_selects_its_own_life_history(model_class):
    model = _model(model_class)
    result = model(_input(model_class, [5, 7, 5]))
    assert result[0, 0] == result[2, 0]
    assert result[0, 0] != result[1, 0]


@pytest.mark.parametrize("model_class", SPECIES_CLASSES)
def test_positive_species_scores_need_not_be_exactly_one_hot(model_class):
    # The gold-standard suite feeds every column a positive random value; the
    # largest indicator keeps selecting the species, as before.
    model = _model(model_class)
    x = _input(model_class, [None])
    x[0, 2:] = torch.linspace(0.1, 0.2, N_SPECIES[model_class], dtype=torch.float64)
    x[0, 2 + 9] = 0.9
    expected = model(_input(model_class, [9]))
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

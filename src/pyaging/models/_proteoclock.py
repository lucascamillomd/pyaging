"""Proteomic clocks converted from their original authors' releases."""

import torch

from ._base_models import pyagingModel


class HPS(pyagingModel):
    """Kuo et al. Healthspan Proteomic Score from 86 NPX proteins and age.

    Inputs use the original lowercase protein names and age in years. The
    returned probability of remaining free of the study's disease/death
    endpoint over ten years ranges from zero to one, with higher values better.
    No additional scaling, imputation or age residualization is performed.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("gompertz_parameters", torch.empty(2, dtype=torch.float64))

    def validate_inputs(self, adata):
        if adata.var_names.has_duplicates:
            raise ValueError("HPS requires unique feature names; aggregate duplicate proteins before prediction.")
        missing = [name for name in self.features if name not in adata.var_names]
        if missing:
            raise ValueError(
                "HPS requires all 86 proteins and chronological age; missing predictors: "
                + ", ".join(missing)
                + ". Use the original lowercase protein identifiers and age in years."
            )

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        shape, rate = self.gompertz_parameters.unbind()
        rate_new = rate * torch.exp(x)
        cdf_10_year = 1 - torch.exp(-(rate_new / shape) * (torch.exp(shape * 10) - 1))
        # Preserve the original R calculation, including rounding at the limits.
        return 1 - cdf_10_year


class PAC(pyagingModel):
    """Kuo et al. proteomic age from 128 Olink NPX proteins and age in years.

    Feature names follow the original R release's lowercase identifiers, including
    ``fut3_fut5`` and ``ntprobnp``. All 129 columns must be supplied. The model
    performs no cohort scaling or missing-value imputation. Supplied NaNs
    propagate, matching the original function; deliberate upstream imputation
    remains the caller's choice. Output is mortality-equivalent age in years.
    """

    def __init__(self):
        super().__init__()
        # shape, rate, age-only shape, age-only rate, age-only age coefficient.
        self.register_buffer("gompertz_parameters", torch.empty(5, dtype=torch.float64))

    def validate_inputs(self, adata):
        """Prevent the generic predictor from replacing absent inputs with zero."""
        if adata.var_names.has_duplicates:
            raise ValueError("PAC requires unique feature names; aggregate duplicate proteins before prediction.")
        missing = [name for name in self.features if name not in adata.var_names]
        if missing:
            raise ValueError(
                "PAC requires all 128 proteins and chronological age; missing predictors: "
                + ", ".join(missing)
                + ". Use the original lowercase protein identifiers and age in years."
            )

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        shape, rate, shape0, rate0, beta_age = self.gompertz_parameters.unbind()
        rate_new = rate * torch.exp(x)
        cdf_10_year = 1 - torch.exp(-(rate_new / shape) * (torch.exp(shape * 10) - 1))
        # Retain the original expression, including its behavior at saturated
        # mortality risks; no clipping or fitted recalibration is introduced.
        return torch.log(shape0 * torch.log(1 - cdf_10_year) / (rate0 * (1 - torch.exp(10 * shape0)))) / beta_age

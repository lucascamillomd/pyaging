"""Author-defined IntrinClock and PhenoAge variants."""

import torch

from ._base_models import pyagingModel
from ._models import IntrinClock


class IntrinClock370(IntrinClock):
    """The original cv.glmnet model at lambda.1se, as used by the author demo."""


class DNAmPhenoAgePRC(pyagingModel):
    """Polycomb-related PhenoAge contribution, with no age intercept.

    Matches the author's ``calcPRCPhenoAge(..., imputation = FALSE)``:
    missing CpGs and supplied NA beta values make zero contribution.
    """

    def preprocess(self, x):
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)

    def postprocess(self, x):
        return x


class DNAmPhenoAgeNonPRC(DNAmPhenoAgePRC):
    """Complementary non-polycomb PhenoAge contribution, without intercept."""

"""PC clocks evaluated with algebraically composed PCA/regression weights.

The builders preserve the original authors' double-precision coefficients.
Composing rotation and regression once avoids storing the large PCA rotation
again for each standalone PCGrimAge proxy. This does not retrain the models.
"""

import torch

from ._models import LinearReferenceClock


class PCBrainAge(LinearReferenceClock):
    """Thrush et al. brain chronological-age predictor, in years."""


class _PCGrimAgeProxy(LinearReferenceClock):
    """Keep the selected original demographic covariates mandatory."""

    def __init__(self):
        super().__init__()
        self.required_covariates = []
        self.register_buffer("required_covariate_indices", torch.empty(0, dtype=torch.long))

    def preprocess(self, x):
        selected = x[:, self.required_covariate_indices]
        invalid = ~torch.isfinite(selected)
        if invalid.any():
            bad_columns = invalid.any(dim=0).nonzero().flatten().tolist()
            names = ", ".join(self.required_covariates[i] for i in bad_columns)
            raise ValueError(
                f"{self.metadata.get('clock_name') or type(self).__name__} requires "
                f"supplied finite demographic inputs for every sample: {names}."
            )
        return super().preprocess(x)


class PCGrimAgePackYrs(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm smoking pack-years proxy."""


class PCGrimAgeADM(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm adrenomedullin proxy."""


class PCGrimAgeB2M(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm beta-2-microglobulin proxy."""


class PCGrimAgeCystatinC(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm cystatin C proxy."""


class PCGrimAgeGDF15(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm GDF15 proxy."""


class PCGrimAgeLeptin(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm leptin proxy."""


class PCGrimAgePAI1(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm PAI-1 proxy."""


class PCGrimAgeTIMP1(_PCGrimAgeProxy):
    """Higgins-Chen et al. PC-based DNAm TIMP-1 proxy."""

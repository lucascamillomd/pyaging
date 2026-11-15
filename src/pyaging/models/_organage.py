"""Goeminne et al. plasma-protein OrganAge models.

Original author code and coefficients:
https://github.com/ludgergoeminne/organAging
DOI: 10.1016/j.cmet.2024.10.005

The released coefficients are subject to the authors' Academic / Non-Commercial
License. Each converted artifact carries that license and its pinned provenance.
"""

from ._base_models import pyagingModel


class OrganAge(pyagingModel):
    """Score Olink NPX values with a released OrganAge coefficient vector.

    Protein names are the authors' case-sensitive symbols, including hyphens.
    NPX values are already normalized; no cohort scaling is applied here. The
    authors recommend fold one for new data. Chronological models include an
    intercept and return years; mortality models have no intercept and return
    relative natural-log mortality hazards.

    The authors' example omits absent protein columns and propagates NaN values
    in present columns. Zero-filled absent features preserve that omission in
    pyaging, while present NaNs remain NaNs. Any imputation performed before
    scoring is the caller's separate data-preparation decision.
    """

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

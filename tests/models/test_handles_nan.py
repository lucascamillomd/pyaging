"""Clocks that give a supplied NaN its own meaning receive it untouched."""

import inspect

import pytest

import pyaging.models as models
from pyaging.models._base_models import pyagingModel
from pyaging.models._models import _DNAmFitAgeSexGated
from pyaging.models._pc_extensions import _PCGrimAgeProxy

HANDLES_NAN = [
    # A cohort median fills NaN before ranking.
    models.Pasta,
    models.PastaMouse,
    models.Reg,
    # NaN questionnaire codes read as "no", NaN visit counts and lipids as zero.
    models.LinAge2,
    # NaN autosomes are left out of the sample's mean and standard deviation.
    models.SexScoreBase,
    models.XChrom,
    models.YChrom,
    # Components and sexes fill NaN from their own reference vectors.
    models.DNAmFitAge,
    _DNAmFitAgeSexGated,
    models.DNAmFitAgeGait,
    models.DNAmFitAgeGrip,
    # NaN NPX is zero abundance, and LightGBM routes NaN as zero.
    models.PAOPAC,
    # The authors set supplied NaN to zero.
    models.CellDRIFT,
    # NaN betas are omitted from the least-squares objective.
    models.MiAge,
    # The authors' scoring propagates supplied NaN.
    models.OrganAge,
    models.HPS,
    models.PAC,
    # Supplied NA contributes zero.
    models.DNAmPhenoAgePRC,
    models.DNAmPhenoAgeNonPRC,
    # NaN demographic covariates must raise, not be filled.
    _PCGrimAgeProxy,
    models.PCGrimAgePackYrs,
    models.PCGrimAgeADM,
    models.PCGrimAgeB2M,
    models.PCGrimAgeCystatinC,
    models.PCGrimAgeGDF15,
    models.PCGrimAgeLeptin,
    models.PCGrimAgePAI1,
    models.PCGrimAgeTIMP1,
]


@pytest.mark.parametrize("model_class", HANDLES_NAN, ids=lambda cls: cls.__name__)
def test_clock_receives_nan_untouched(model_class):
    assert model_class.handles_nan is True


def test_every_other_clock_lets_predict_age_fill_nan():
    others = [
        cls
        for _, cls in inspect.getmembers(models, inspect.isclass)
        if issubclass(cls, pyagingModel) and cls not in HANDLES_NAN
    ]
    assert models.DunedinPACE in others and models.epiTOC2 in others
    assert [cls.__name__ for cls in others if cls.handles_nan] == []

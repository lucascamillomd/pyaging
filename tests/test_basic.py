import pyaging as pya
import pandas as pd
import pytest

test_df = pd.read_csv('./microarray_methylation_test.csv', index_col=0)

@pytest.fixture
def test_df_to_adata():

    test_adata = pya.pp.df_to_adata(test_df)

    assert 'X_original' in test_adata.layers
    
    return test_adata

def test_predict_age(test_df_to_adata):

    clocks = ["altumage", "dnamphenoage", 'dnamtl', 'dunedinpace', 'encen100', 'encen40', 'han', 'hannum', 'horvath2013', 'hrsinchphenoage', 'knight', 'leecontrol',
         'leerefinedrobust', 'leerobust', 'lin', 'mammalian1', 'mammalian2', 'mammalian3', 'mammalianlifespan', 'mammaliansex', 'pcdnamtl', 'pcgrimage', 'pcphenoage', 'pchorvath2013',
         'pchannum', 'pcskinandblood', 'pedbe', 'replitali', 'skinandblood', 'zhangblup', 'zhangen', 'zhangmortality']

    gold_standard_results = [
        26.050183478291864,
         -56.23185903090171,
         7.839094228615214,
         0.8511561751869254,
         88.94055959158753,
         5.373412159394043,
         2.993450396035478,
         -1.9438458520215534,
         31.682096582070322,
         61.897783119109384,
         47.34468100090071,
         35.72589465024217,
         47.04132811252407,
         35.115627841154144,
         79.15939816181614,
         2.840332978447524,
         0.03749687077362242,
         -4.62243650444673,
         0.27465496459381455,
         0.7780705347885173,
         7.396056370541309,
         69.09419182228189,
         37.109878303180466,
         39.66166990875236,
         44.13897261485391,
         39.539388457487135,
         2.2742693239038707,
         62.94031975346403,
         6.424155739658539,
         20.794158327617893,
         0.11545593365738238,
         0.43941984925983346
    ]

    test_adata = pya.pred.predict_age(test_df_to_adata, clocks)
    results = test_adata.obs.iloc[0,:].tolist()
    
    assert test_adata.obs.shape[1] > 0

    tolerance=0.01
    assert len(gold_standard_results) == len(results), "Lists are not of the same length."
    for a, b in zip(gold_standard_results, results):
        assert abs(a - b) <= tolerance, f"Items {a} and {b} differ by more than {tolerance}"

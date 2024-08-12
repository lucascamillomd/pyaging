import pyaging as pya
import pandas as pd
import pytest
import numpy as np

gold_standard_dict = {
    "altumage": 91.64974762567451,
    "bitage": -76.70067243278027,
    "camilloh3k27ac": 50.757604725814055,
    "camilloh3k27me3": 36.20102332363592,
    "camilloh3k36me3": 40.17217801103032,
    "camilloh3k4me1": 40.26368976549848,
    "camilloh3k4me3": 27.711115262670287,
    "camilloh3k9ac": 46.66535426835552,
    "camilloh3k9me3": 6.612750512447683,
    "camillopanhistone": 2.146781771450619,
    "dnamphenoage": 57.21316697460426,
    "dnamtl": 9.242968475534473,
    "dunedinpace": -0.577754455730024,
    "encen100": 81.21861575842387,
    "encen40": 201.6772630773458,
    "grimage": 164.54404170150434,
    "grimage2": 137.76422699444927,
    "han": 5.024133336358977,
    "hannum": 136.00991163552106,
    "horvath2013": 235.83086467856762,
    "hrsinchphenoage": 133.77982661541225,
    "knight": -7.487121676241699,
    "leecontrol": 22.28467579012336,
    "leerefinedrobust": 39.912822644527836,
    "leerobust": 41.973395885931495,
    "lin": 10.991036388334347,
    "mammalian1": 25.700587981338234,
    "mammalian2": 2.7572620018239404,
    "mammalian3": 107.24767140561077,
    "mammalianblood2": 22.069842591638746,
    "mammalianblood3": -0.06566936993738942,
    "mammalianfemale": 0.41548384625720686,
    "mammalianlifespan": 0.12008702433972934,
    "mammalianskin2": 28.911325755269115,
    "mammalianskin3": 17.45229968406876,
    "meer": 32.79787462061331,
    "ocampoatac1": 30.036040356764108,
    "ocampoatac2": 38.427697586227694,
    "pcdnamtl": 6.688874995049472,
    "pcgrimage": 102.19417671572069,
    "pchannum": 110.44589881232716,
    "pchorvath2013": 41.555737395108835,
    "pcphenoage": 72.51714752302486,
    "pcskinandblood": 55.508428118217466,
    "pedbe": 5.947250020578089,
    "petkovich": 24.47996061655825,
    "phenoage": -64.36037459874132,
    "replitali": 94.8332866338199,
    "skinandblood": 104.75113267297903,
    "stubbs": 1.7329077738158296,
    "thompson": 164.57995856164365,
    "zhangblup": 78.76779185124363,
    "zhangen": 37.404900683228966,
    "zhangmortality": 2.8135717975793475,
    "dnamfitage": 91.03008383895092,
    "yingcausage": 195.3013578758023,
    "yingadaptage": 173.48314231920278,
    "yingdamage": -53.509282005508,
    "stoch": 136.48186449945413,
    "stocz": -21.238430415750855,
    "stocp": -48.95705647484499,
    "stemtoc": 2.060230880575066,
    "epitoc1": 0.8689615831989005,
    "retroelementagev1": 144.2884653588903,
    "retroelementagev2": 53.802897567351465,
    "intrinclock": 132.97353886880327,
}


def test_all_clocks():
    all_clocks = list(gold_standard_dict.keys())

    logger = pya.logger.Logger("test_logger")
    pya.logger.silence_logger("test_logger")
    device = "cpu"
    dir = "pyaging_data"
    indent_level = 1
    tolerance = 0.01
    for clock_name in all_clocks:
        clock = pya.pred.load_clock(
            clock_name, device, dir, logger, indent_level=indent_level
        )
        partial_clock_features = clock.features[
            0 : len(clock.features) * 2 // 3
        ]  # 1/3 dropout to simulate missing features
        np.random.seed(42)
        random_df = pd.DataFrame(
            np.abs(
                np.random.normal(
                    loc=0.5, scale=1, size=(1, len(partial_clock_features))
                )
            ),
            columns=partial_clock_features,
        )
        random_adata = pya.pp.df_to_adata(
            random_df, imputer_strategy="constant", verbose=False
        )
        pya.pred.predict_age(random_adata, clock_name, verbose=False)
        pred = random_adata.obs.iloc[0, 0]
        gold_pred = gold_standard_dict[clock_name]

        assert (
            abs(pred - gold_pred) <= tolerance
        ), f"Items {pred} and {gold_pred} differ by more than {tolerance} for clock {clock_name}"

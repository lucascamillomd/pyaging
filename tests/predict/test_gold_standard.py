import pyaging as pya
import pandas as pd
import pytest
import numpy as np

gold_standard_dict = {
    "altumage": 221.855353465906,
    "bitage": -47.462976068258286,
    "camilloh3k27ac": 52.2139525471314,
    "camilloh3k27me3": -32.08305725722438,
    "camilloh3k36me3": -23.204613344318716,
    "camilloh3k4me1": 74.4355753103286,
    "camilloh3k4me3": 106.93563878974682,
    "camilloh3k9ac": 60.32784086967423,
    "camilloh3k9me3": -1.816178293470081,
    "camillopanhistone": -125.42354004544902,
    "dnamphenoage": -867.4843089580536,
    "dnamtl": -1.305844736052677,
    "dunedinpace": 1.0299402861393916,
    "encen100": 334.9000312257558,
    "encen40": 663.2005664819444,
    "grimage": 807.2359675923526,
    "grimage2": 441.4201285301567,
    "han": 102.47735653375275,
    "hannum": 395.91179417073727,
    "horvath2013": 2304.703573714046,
    "hrsinchphenoage": 121.6780250556767,
    "knight": -327.54276215041494,
    "leecontrol": -79.85017174563836,
    "leerefinedrobust": 132.18831804022193,
    "leerobust": 117.00740800439962,
    "lin": -220.98759344220161,
    "mammalian1": 669538452400.8698,
    "mammalian2": -0.0082191780821917,
    "mammalian3": 20.62780801183159,
    "mammalianblood2": -0.0164383561643836,
    "mammalianblood3": 46.51304557684458,
    "mammalianfemale": 1.308827388489461e-13,
    "mammalianlifespan": 180.00103954876235,
    "mammalianskin2": 19.499999999930857,
    "mammalianskin3": 46.34280296801864,
    "meer": 120.39042223938054,
    "ocampoatac1": 33.52345113221555,
    "ocampoatac2": 39.056304196513516,
    "pcdnamtl": 4.001499106191858,
    "pcgrimage": 208.64970067180258,
    "pchannum": 398.5010640928993,
    "pchorvath2013": 16.8214517757981,
    "pcphenoage": 221.85806225438336,
    "pcskinandblood": 267.8075723943516,
    "pedbe": 67.73442419651838,
    "petkovich": 44.53695328962486,
    "phenoage": -58.961633346065355,
    "replitali": 17.26060414488893,
    "skinandblood": 1103.4695842499932,
    "stubbs": 0.6185829836658536,
    "thompson": 457.5872043294585,
    "zhangblup": 105.75120267204859,
    "zhangen": 41.32825956922637,
    "zhangmortality": -20.128120318055153,
}


def test_all_clocks():
    all_clocks = list(gold_standard_dict.keys())

    logger = pya.logger.Logger("test_logger")
    pya.logger.silence_logger("test_logger")
    np.random.seed(42)
    device = "cpu"
    dir = "pyaging_data"
    indent_level = 1
    tolerance = 0.01
    preds = []
    for clock_name in all_clocks:
        clock = pya.pred.load_clock(
            clock_name, device, dir, logger, indent_level=indent_level
        )
        partial_clock_features = clock.features[
            0 : len(clock.features) * 2 // 3
        ]  # 1/3 dropout to simulate missing features
        random_df = pd.DataFrame(
            np.random.randint(0, 10, (1, len(partial_clock_features))),
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

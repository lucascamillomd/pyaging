"""Turn the exported LinAge2 reference constants into ``clocks/linage2_params.json``.

The reference implementation (``linAge2.R``) ships no fitted constants: it re-derives the
normalization statistics and the Cox coefficients on every run from ``mergedDataNHANES9902.csv``,
which is deliberately not vendored here. ``consts/linAge2_export.R`` is that script patched to
dump the constants once; its output lives in ``consts/`` and is what this script reads.

This script's other job is naming. LinAge2 speaks NHANES variable codes; pyaging speaks
descriptive snake_case, one name per measurement package-wide. ``NHANES_TO_PYAGING`` below is the
full translation, and every unit was checked against ``codebook_linAge2.csv``.

Run from the repository root::

    uv run python clocks/linage2_source/extract_linage2_params.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parent
CONSTS = SOURCE / "consts"
OUTPUT = SOURCE.parent / "linage2_params.json"

# Every NHANES code LinAge2 consumes, mapped to its pyaging feature name. Units in the comments
# are the codebook's, which are also the units the extracted median/MAD assume.
NHANES_TO_PYAGING = {
    # --- examination ---
    "BPXPLS": "pulse",  # 60 sec pulse, bpm
    "BPXSAR": "systolic_blood_pressure",  # mmHg
    "BPXDAR": "diastolic_blood_pressure",  # mmHg
    "BMXBMI": "body_mass_index",  # kg/m^2
    # --- urine ---
    "URXUMASI": "urine_albumin",  # mg/L
    "URXUCRSI": "urine_creatinine",  # umol/L
    # --- iron studies ---
    "LBDIRNSI": "iron",  # umol/L
    "LBDTIBSI": "total_iron_binding_capacity",  # umol/L
    "LBXPCT": "transferrin_saturation",  # %
    "LBDFERSI": "ferritin",  # ug/L
    # --- vitamins ---
    "LBDFOLSI": "folate",  # nmol/L
    "LBDB12SI": "vitamin_b12",  # pmol/L
    # --- smoking ---
    "LBXCOT": "cotinine",  # ng/mL; digitized to `smoking_intensity` before the model sees it
    # --- complete blood count ---
    "LBXWBCSI": "white_blood_cell_count",  # 10^3 cells/uL
    "LBXLYPCT": "lymphocyte_percent",  # %
    "LBXMOPCT": "monocyte_percent",  # %
    "LBXNEPCT": "neutrophil_percent",  # %
    "LBXEOPCT": "eosinophil_percent",  # %
    "LBXBAPCT": "basophil_percent",  # %
    "LBDLYMNO": "lymphocyte_count",  # 10^3 cells/uL
    "LBDMONO": "monocyte_count",  # 10^3 cells/uL
    "LBDNENO": "neutrophil_count",  # 10^3 cells/uL
    "LBDEONO": "eosinophil_count",  # 10^3 cells/uL
    "LBDBANO": "basophil_count",  # 10^3 cells/uL
    "LBXRBCSI": "red_blood_cell_count",  # 10^6 cells/uL
    "LBXHGB": "hemoglobin",  # g/dL
    "LBXHCT": "hematocrit",  # %
    "LBXMCVSI": "mean_cell_volume",  # fL
    "LBXMCHSI": "mean_cell_hemoglobin",  # pg
    "LBXMC": "mean_cell_hemoglobin_concentration",  # g/dL
    "LBXRDW": "red_cell_distribution_width",  # %
    "LBXPLTSI": "platelet_count",  # 10^3 cells/uL
    "LBXMPSI": "mean_platelet_volume",  # fL
    # --- inflammation, glycemia, cardiac ---
    "LBXCRP": "c_reactive_protein",  # mg/dL
    "LBXGH": "hemoglobin_a1c",  # %
    "SSBNP": "nt_probnp",  # pg/mL
    # --- biochemistry panel ---
    "LBDSALSI": "albumin",  # g/L
    "LBXSATSI": "alanine_aminotransferase",  # U/L
    "LBXSASSI": "aspartate_aminotransferase",  # U/L
    "LBXSAPSI": "alkaline_phosphatase",  # IU/L, numerically U/L
    "LBDSBUSI": "blood_urea_nitrogen",  # mmol/L
    "LBDSCASI": "calcium",  # mmol/L
    "LBXSC3SI": "bicarbonate",  # mmol/L
    "LBDSGLSI": "glucose",  # mmol/L
    "LBXSLDSI": "lactate_dehydrogenase",  # U/L
    "LBDSPHSI": "phosphorus",  # mmol/L
    "LBDSTBSI": "total_bilirubin",  # umol/L
    "LBDSTPSI": "total_protein",  # g/L
    "LBDSUASI": "uric_acid",  # umol/L
    "LBDSCRSI": "creatinine",  # umol/L
    "LBXSNASI": "sodium",  # mmol/L
    "LBXSKSI": "potassium",  # mmol/L
    "LBXSCLSI": "chloride",  # mmol/L
    "LBDSGBSI": "globulin",  # g/L
    # --- lipids: consumed by the Friedewald LDL, then dropped; not model features ---
    "LBDTCSI": "total_cholesterol",  # mmol/L
    "LBDHDLSI": "hdl_cholesterol",  # mmol/L
    "LBDSTRSI": "triglycerides",  # mmol/L
    # --- questionnaire, all NHANES 1999-2000 coded categoricals ---
    "BPQ020": "told_high_blood_pressure",
    "DIQ010": "told_diabetes",
    "HUQ010": "general_health_condition",
    "HUQ020": "health_compared_to_one_year_ago",
    "HUQ050": "healthcare_visits_past_year",
    "HUQ070": "hospital_overnight_past_year",
    "KIQ020": "told_weak_or_failing_kidneys",
    "MCQ010": "told_asthma",
    "MCQ053": "treated_for_anemia_past_3_months",
    "MCQ160A": "told_arthritis",
    "MCQ160B": "told_congestive_heart_failure",  # collected by the reference code but unused
    "MCQ160C": "told_coronary_heart_disease",
    "MCQ160D": "told_angina",
    "MCQ160E": "told_heart_attack",
    "MCQ160F": "told_stroke",
    "MCQ160G": "told_emphysema",
    "MCQ160I": "told_thyroid_disease",
    "MCQ160J": "told_overweight",
    "MCQ160K": "told_chronic_bronchitis",
    "MCQ160L": "told_liver_condition",
    "MCQ220": "told_cancer",
    "OSQ010A": "fractured_hip",
    "OSQ010B": "fractured_wrist",
    "OSQ010C": "fractured_spine",
    "OSQ060": "told_osteoporosis",
    "PFQ056": "confusion_or_memory_problems",
    # --- derived model features, named for what they measure ---
    "fs1Score": "comorbidity_index",
    "fs2Score": "self_reported_health_index",
    "fs3Score": "healthcare_use_index",
    "LDLV": "ldl_cholesterol",
    "crAlbRat": "urine_albumin_creatinine_ratio",
}

# LBXCOT is the one NHANES code whose model slot is not the raw measurement: the reference code
# digitizes cotinine into a four-level smoking-intensity code before the feature vector is built.
MODEL_SLOT_OVERRIDES = {"LBXCOT": "smoking_intensity"}

# The 22 yes/no items summed by fs1Score. MCQ160B is deliberately absent -- the reference code
# reads it and never uses it, and the denominator is 22.
COMORBIDITY_ITEMS = [
    "told_high_blood_pressure",
    "told_diabetes",
    "told_weak_or_failing_kidneys",
    "told_asthma",
    "treated_for_anemia_past_3_months",
    "told_arthritis",
    "told_coronary_heart_disease",
    "told_angina",
    "told_heart_attack",
    "told_stroke",
    "told_emphysema",
    "told_thyroid_disease",
    "told_overweight",
    "told_chronic_bronchitis",
    "told_liver_condition",
    "told_cancer",
    "fractured_hip",
    "fractured_wrist",
    "fractured_spine",
    "told_osteoporosis",
    "confusion_or_memory_problems",
    "hospital_overnight_past_year",
]

DERIVED = {
    "smoking_intensity": {
        "nhanes": "LBXCOT",
        "inputs": ["cotinine"],
        "recipe": "step function on cotinine ng/mL: <10 -> 0, <100 -> 1, <200 -> 2, else 3",
    },
    "comorbidity_index": {
        "nhanes": "fs1Score",
        "inputs": COMORBIDITY_ITEMS,
        "recipe": (
            "count of the 22 items answered 'yes' (code 1), divided by 22; told_diabetes also "
            "counts code 3 (borderline); missing items default to 'no'"
        ),
    },
    "self_reported_health_index": {
        "nhanes": "fs2Score",
        "inputs": ["general_health_condition", "health_compared_to_one_year_ago"],
        "recipe": (
            "((health == 4) * 2 + (health == 5) * 4) * (1 - (versus == 1) * 0.5 + (versus == 2)); "
            "missing defaults are health = 3 and versus = 3, giving 0"
        ),
    },
    "healthcare_use_index": {
        "nhanes": "fs3Score",
        "inputs": ["healthcare_visits_past_year"],
        "recipe": "the raw HUQ050 visit-count code used as a number, with 77, 99 and missing mapped to 0",
    },
    "ldl_cholesterol": {
        "nhanes": "LDLV",
        "inputs": ["total_cholesterol", "triglycerides", "hdl_cholesterol"],
        "recipe": (
            "Friedewald: total_cholesterol - triglycerides / 5 - hdl_cholesterol, in mmol/L. The /5 "
            "divisor is the mg/dL constant applied to mmol/L inputs; that is what the reference does. "
            "The reference substitutes 0 when any input is missing"
        ),
    },
    "urine_albumin_creatinine_ratio": {
        "nhanes": "crAlbRat",
        "inputs": ["urine_albumin", "urine_creatinine"],
        "recipe": "urine_albumin / (urine_creatinine * 1.1312e-4), mg albumin per g creatinine",
    },
}

# Inputs that are read but never reach the model vector, so they are not part of the contract.
NON_INPUT_COLUMNS = {"SEQN", "RIAGENDR", "RIDAGEEX"}


def model_slot(code: str) -> str:
    """The pyaging name of the feature-vector slot NHANES variable ``code`` fills."""
    return MODEL_SLOT_OVERRIDES.get(code, NHANES_TO_PYAGING[code])


def per_sex_constants(
    suffix: str, normstats: pd.DataFrame, loadings: pd.DataFrame, cox: pd.DataFrame, beta_null: float, mean_null: float
) -> dict:
    pc_index = [int(name.removeprefix("PC")) for name in cox["name"][1:]]
    return {
        "median": normstats[f"med_{suffix}"].tolist(),
        "mad": normstats[f"mad_{suffix}"].tolist(),
        "loadings": loadings.values.tolist(),
        "pc_index": pc_index,
        "beta": cox["beta"].tolist(),
        "means": cox["mean"].tolist(),
        "beta_null": beta_null,
        "mean_null": mean_null,
        # The reference rounds the mortality rate doubling time to two decimals before scaling
        # the risk delta with it. Skipping the rounding shifts the biological age in the 4th decimal.
        "mrdt": round(float(np.log(2) / beta_null), 2),
    }


def validation_cases() -> list[dict]:
    """The paper's two published subjects, with their inputs renamed to pyaging's convention."""
    user_data = pd.read_csv(SOURCE / "userData.csv")
    published = {8881: 88.69, 9106: 64.36}
    cases = []
    for _, row in user_data.iterrows():
        seqn = int(row["SEQN"])
        inputs = {NHANES_TO_PYAGING[code]: float(row[code]) for code in row.index if code not in NON_INPUT_COLUMNS}
        cases.append(
            {
                "seqn": seqn,
                # RIDAGEEX is age at examination in months; pyaging's `age` is years.
                "age_years": float(row["RIDAGEEX"]) / 12,
                "female": int(row["RIAGENDR"]) == 2,
                "inputs": inputs,
                "biological_age": published[seqn],
            }
        )
    return cases


def main() -> None:
    features = pd.read_csv(CONSTS / "features.csv")
    normstats = pd.read_csv(CONSTS / "normstats.csv")
    cox_null = pd.read_csv(CONSTS / "coxnull.csv").set_index("sex")

    codes = features["feature"].tolist()
    if normstats["feature"].tolist() != codes:
        raise ValueError("normstats.csv is not in the same feature order as features.csv")

    # The codebook classifies every variable, so let it decide which inputs are questionnaire items
    # rather than guessing from the code prefix.
    codebook = pd.read_csv(SOURCE / "codebook_linAge2.csv").set_index("Var")
    kind = codebook["Demo/Exam/Quest/Lab/Mort"]

    # Codes that only ever exist as a derivation, so the user supplies their ingredients instead.
    computed_codes = {spec["nhanes"] for spec in DERIVED.values()} - set(MODEL_SLOT_OVERRIDES)
    numeric_inputs = [NHANES_TO_PYAGING[code] for code in codes if code not in computed_codes]
    # The three lipids are consumed by the Friedewald LDL and then dropped, so they are inputs
    # without being features.
    numeric_inputs += [NHANES_TO_PYAGING[code] for code in ("LBDTCSI", "LBDHDLSI", "LBDSTRSI")]
    questionnaire_inputs = [name for code, name in NHANES_TO_PYAGING.items() if kind.get(code) == "Q"]

    params = {
        "features": [model_slot(code) for code in codes],
        "inputs": {
            "numeric": numeric_inputs,
            "questionnaire": questionnaire_inputs,
            "female": "female",
            "age": "age",
        },
        "nhanes_to_pyaging": NHANES_TO_PYAGING,
        "derived": DERIVED,
        # lambda is either 0 (natural log) or NA (identity); no other Box-Cox power occurs.
        "log_mask": features["lam"].eq(0).tolist(),
        "skip_mask": normstats["skip"].tolist(),
        # The loading matrices are the one constant the archive does ship, so read them from it.
        "male": per_sex_constants(
            "m",
            normstats,
            pd.read_csv(SOURCE / "vMatDat99_M_pre.csv"),
            pd.read_csv(CONSTS / "coxM.csv"),
            float(cox_null.loc["M", "beta"]),
            float(cox_null.loc["M", "mean"]),
        ),
        "female": per_sex_constants(
            "f",
            normstats,
            pd.read_csv(SOURCE / "vMatDat99_F_pre.csv"),
            pd.read_csv(CONSTS / "coxF.csv"),
            float(cox_null.loc["F", "beta"]),
            float(cox_null.loc["F", "mean"]),
        ),
        "validation": validation_cases(),
    }

    if len(numeric_inputs) != 57 or len(questionnaire_inputs) != 26:
        raise ValueError(
            f"expected 57 numeric and 26 questionnaire inputs, "
            f"got {len(numeric_inputs)} and {len(questionnaire_inputs)}"
        )

    OUTPUT.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

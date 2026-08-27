import math

import torch
import torch.nn as nn

# ``predict/_pred_utils`` imports ``..models`` in turn, so these two lines close a cycle.
# It holds because ``_pred_utils`` only takes names ``models/__init__`` has already bound
# from ``_base_models`` by the time this module runs; importing a name defined here would
# break it.
from ..predict._transforms import PHENOAGE_CRP_FLOOR_MG_DL, crp_index, log1p_crp, mortality_to_phenoage_saopaulo
from ._base_models import pyagingModel

# LinAge2's ``foldOutliers`` cap, in MAD-scaled z units, applied to every feature.
LINAGE2_FOLD_CAP = 6

# The 22 yes/no items ``popPCFIfs1`` sums into fs1Score, which is also the denominator.
# ``told_congestive_heart_failure`` (MCQ160B) is read by the reference and then never
# used, so it is absent here on purpose and the divisor stays 22.
LINAGE2_COMORBIDITY_ITEMS = (
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
)


class AltumAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Scales an array based on the median and standard deviation.
        """
        median = torch.tensor(self.preprocess_dependencies[0], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.preprocess_dependencies[1], device=x.device, dtype=x.dtype)
        x = (x - median) / std
        return x

    def postprocess(self, x):
        return x


class BiTAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Binarizes an array based on the median of each row, excluding zeros.
        """
        # Create a mask for non-zero elements
        non_zero_mask = x != 0

        # Apply mask, calculate median for each row, and binarize data
        for i in range(x.size(0)):
            non_zero_elements = x[i][non_zero_mask[i]]
            if non_zero_elements.nelement() > 0:
                median_value = non_zero_elements.median()
                x[i] = (x[i] > median_value).float()
            else:
                # Handle the case where all elements are zero
                x[i] = torch.zeros_like(x[i])

        return x

    def postprocess(self, x):
        return x


class CamilloH3K27ac(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloH3K27me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloH3K36me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloH3K4me1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloH3K4me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloH3K9ac(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloH3K9me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CamilloPanHistone(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmPhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmTL(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmIC(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DunedinPACE(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Apply quantile normalization on x using gold standard means.
        """
        # Ensure gold_standard_means is a 1D tensor and sorted
        sorted_gold_standard = torch.sort(torch.tensor(self.reference_values, device=x.device, dtype=x.dtype))[0]

        # Pre-compute the quantile indices
        quantile_indices = torch.linspace(0, len(sorted_gold_standard) - 1, steps=x.size(1)).long()

        # Prepare a tensor to hold normalized data
        normalized_data = torch.empty_like(x, device=x.device, dtype=x.dtype)

        for i in range(x.size(0)):
            sorted_indices = torch.argsort(x[i, :])
            normalized_data[i, sorted_indices] = sorted_gold_standard[quantile_indices]

        # Return only the subset from x that is used in the base model
        return normalized_data[:, self.preprocess_dependencies[0]]

    def postprocess(self, x):
        return x


class ENCen100(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class ENCen40(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class Han(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class Hannum(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class Horvath2013(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class HypoClock(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Compute mean beta per sample, excluding missing (-1) values.
        """
        means = []
        for row in x:
            filtered_row = row[row != -1]
            if len(filtered_row) > 0:
                mean = torch.mean(filtered_row)
            else:
                mean = torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
            means.append(mean)
        return torch.vstack(means)

    def postprocess(self, x):
        return 1 - x


class HRSInCHPhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class HomeostaticDysregulation(pyagingModel):
    """Mahalanobis distance from a young, healthy NHANES III reference cohort."""

    def __init__(self):
        super().__init__()
        for sex in ["male", "female"]:
            for name in ["reference_mean", "reference_sd", "center", "precision", "log_hd_sd"]:
                self.register_buffer(f"{name}_{sex}", torch.empty(0))

    def preprocess(self, x):
        return log1p_crp(self.features, x)

    def postprocess(self, x):
        """Score the log Mahalanobis distance from the sex's reference cohort.

        Notes
        -----
        ``center`` is not zero. ``hd_calc`` takes each column's mean and standard
        deviation with ``na.rm = TRUE`` over the full reference column and drops
        incomplete rows only afterwards, so the surviving rows' mean is not the
        centring constant; the residual offset reaches 0.19 standard deviations
        and is what ``center`` holds.

        The distance is divided by the projection cohort's standard deviation of
        ``log(distance)``, which is how ``hd_nhanes`` reports it. That constant is
        stored per sex so a single sample can be scored without a cohort.
        """
        biomarkers, female = x[:, :-1], x[:, -1]

        def score(sex):
            mean, sd, center, precision, log_hd_sd = (
                getattr(self, f"{name}_{sex}").to(device=x.device, dtype=x.dtype)
                for name in ("reference_mean", "reference_sd", "center", "precision", "log_hd_sd")
            )
            deviation = (biomarkers - mean) / sd - center
            squared = (deviation @ precision * deviation).sum(dim=1)
            return torch.log(torch.sqrt(squared)) / log_hd_sd

        return torch.where(female == 1, score("female"), score("male")).unsqueeze(-1)


class KDMAge(pyagingModel):
    """Klemera-Doubal biological age with sex-specific NHANES III parameters."""

    def __init__(self):
        super().__init__()
        for sex in ["male", "female"]:
            for name in ["q", "k", "s"]:
                self.register_buffer(f"{name}_{sex}", torch.empty(0))
            self.register_buffer(f"s_ba2_{sex}", torch.empty(0))

    def preprocess(self, x):
        return log1p_crp(self.features, x)

    def postprocess(self, x):
        """Solve the Klemera-Doubal closed form for biological age.

        Notes
        -----
        The denominator sums over every biomarker, matching ``kdm_calc``'s
        deliberate choice not to rescale the estimate for missing markers.
        """
        biomarkers, age, female = x[:, :-2], x[:, -2], x[:, -1]

        def estimate(sex):
            q, k, s, s_ba2 = (
                getattr(self, f"{name}_{sex}").to(device=x.device, dtype=x.dtype) for name in ("q", "k", "s", "s_ba2")
            )
            numerator = ((biomarkers - q) * k / s**2).sum(dim=1)
            denominator = ((k / s) ** 2).sum()
            return (numerator + age / s_ba2) / (denominator + 1 / s_ba2)

        return torch.where(female == 1, estimate("female"), estimate("male")).unsqueeze(-1)


class Knight(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class LeeControl(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class LeeRefinedRobust(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class LeeRobust(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class Lin(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class LinAge2(pyagingModel):
    """Principal-component clinical clock trained on NHANES IV mortality (Fong et al. 2025)."""

    def __init__(self):
        super().__init__()
        # The 59 names the loadings, medians and MADs are indexed by, in SVD row order.
        # Set from the constants clocks/notebooks/linage2.ipynb derives when the clock is built.
        self.model_features = None
        for sex in ["male", "female"]:
            for name in ["median", "mad", "loadings", "beta", "means", "beta_null", "mean_null", "mrdt"]:
                self.register_buffer(f"{name}_{sex}", torch.empty(0))
            self.register_buffer(f"pc_index_{sex}", torch.empty(0, dtype=torch.long))
        self.register_buffer("log_mask", torch.empty(0, dtype=torch.bool))
        self.register_buffer("skip_mask", torch.empty(0, dtype=torch.bool))

    def preprocess(self, x):
        return x

    def _model_vector(self, x):
        """Assemble the 59-feature vector from the user-facing inputs.

        Notes
        -----
        Every missing-value default the reference implementation applies falls out
        of the comparisons themselves: a NaN questionnaire code never equals 1, so
        it counts as "no" in ``comorbidity_index``, and it never equals 4, 5, 1 or 2,
        so ``self_reported_health_index`` collapses to 0 — which is what ``popPCFIfs1``
        and ``popPCFIfs2`` get from their explicit NA defaults of "no" and "same".
        Those defaults are the *healthiest* profile, so an absent questionnaire block
        biases the biological age downward; the registry notes say so.
        """

        def column(name):
            return x[:, self.features.index(name)]

        # digiCot: a four-level step function on raw cotinine in ng/mL. NaN has no bin
        # in the reference either, so it propagates rather than reading as a non-smoker.
        cotinine = column("cotinine")
        smoking = (cotinine >= 10).to(x.dtype) + (cotinine >= 100).to(x.dtype) + (cotinine >= 200).to(x.dtype)
        smoking = torch.where(torch.isnan(cotinine), cotinine, smoking)

        yes_count = sum((column(item) == 1).to(x.dtype) for item in LINAGE2_COMORBIDITY_ITEMS)
        # DIQ010 == 3 is borderline diabetes and counts alongside DIQ010 == 1.
        comorbidity = (yes_count + (column("told_diabetes") == 3).to(x.dtype)) / len(LINAGE2_COMORBIDITY_ITEMS)

        health = column("general_health_condition")
        versus = column("health_compared_to_one_year_ago")
        severity = (health == 4).to(x.dtype) * 2 + (health == 5).to(x.dtype) * 4
        trend = 1 - (versus == 1).to(x.dtype) * 0.5 + (versus == 2).to(x.dtype)

        # The raw HUQ050 visit-count code used as a number; 77 (refused), 99 (don't
        # know) and missing all become 0.
        visits = column("healthcare_visits_past_year")
        healthcare = torch.where(
            (visits == 77) | (visits == 99) | torch.isnan(visits), torch.zeros_like(visits), visits
        )

        # Friedewald, with the reference's mg/dL divisor applied to mmol/L inputs.
        # ``populateLDL`` substitutes a hard 0 mmol/L when any of the three lipids is
        # missing, which z-scores to roughly -3.6 for a male; reference_values keep an
        # absent column away from that path, but a NaN within a present column reaches it.
        ldl = column("total_cholesterol") - column("triglycerides") / 5 - column("hdl_cholesterol")
        ldl = torch.where(torch.isnan(ldl), torch.zeros_like(ldl), ldl)

        # 1.1312e-4 converts urine creatinine from µmol/L to g/L, giving mg albumin per g.
        albumin_creatinine = column("urine_albumin") / (column("urine_creatinine") * 1.1312e-4)

        derived = {
            "smoking_intensity": smoking,
            "comorbidity_index": comorbidity,
            "self_reported_health_index": severity * trend,
            "healthcare_use_index": healthcare,
            "ldl_cholesterol": ldl,
            "urine_albumin_creatinine_ratio": albumin_creatinine,
        }
        columns = [derived[name] if name in derived else column(name) for name in self.model_features]
        return torch.stack(columns, dim=1)

    def postprocess(self, x):
        """Score the sex-specific Cox model on the folded principal components.

        Notes
        -----
        The 13 log-transformed features take a plain ``log`` with no floor. The
        reference lets ``log(0)`` reach the fold as ``-inf``, where it becomes exactly
        ``-6``; clamping the input first — as ``PhenoAge.preprocess`` does before its
        natural log — would land a below-detection reading on a different z-score, so
        this clock deliberately does not floor.

        The ±6 fold covers all 59 features, including the five that skip z-scoring:
        ``self_reported_health_index`` and ``healthcare_use_index`` both run 0-8 raw
        and saturate at 6.

        ``chronAge`` enters both the full and the null Cox model in months, and the two
        age coefficients differ, so age is a genuine covariate of the delta rather than
        an offset that cancels.
        """
        vector = self._model_vector(x)
        age_months = x[:, self.features.index("age")] * 12
        female = x[:, self.features.index("female")]

        transformed = torch.where(self.log_mask, torch.log(vector), vector)

        def estimate(sex):
            median, mad, loadings, beta, means, beta_null, mean_null, mrdt = (
                getattr(self, f"{name}_{sex}").to(device=x.device, dtype=x.dtype)
                for name in ("median", "mad", "loadings", "beta", "means", "beta_null", "mean_null", "mrdt")
            )
            normalized = torch.where(self.skip_mask, transformed, (transformed - median) / mad)
            folded = normalized.clamp(-LINAGE2_FOLD_CAP, LINAGE2_FOLD_CAP)
            components = folded @ loadings

            # The stored indices are the Cox formula's 1-based PC labels.
            selected = components[:, getattr(self, f"pc_index_{sex}") - 1]
            terms = torch.cat([age_months.unsqueeze(1), selected], dim=1)
            risk = ((terms - means) * beta).sum(dim=1)
            null_risk = beta_null * (age_months - mean_null)
            return (age_months + (risk - null_risk) / math.log(2) * mrdt) / 12

        return torch.where(female == 1, estimate("female"), estimate("male")).unsqueeze(-1)


class Mammalian1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic transformation with an offset of -2.
        """
        return torch.exp(x) - 2


class Mammalian2(pyagingModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_cpg = x[:, :-1756]  # number of species in lookup table
        x_species = x[:, -1756:]  # number of species in lookup table
        x = self.base_model(x_cpg)
        x = self.postprocess(x, x_species)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x, x_species):
        """
        Converts output of relative age to age in units of years.
        """
        indices = torch.argmax(x_species, dim=1)
        anage_array = self.postprocess_dependencies[0]
        anage_tensor = torch.tensor(anage_array, dtype=x.dtype, device=x.device)
        gestation_time = anage_tensor[indices, 0].unsqueeze(1)
        max_age = anage_tensor[indices, 3].unsqueeze(1)

        x = torch.exp(-torch.exp(-x))
        x = x * (max_age + gestation_time) - gestation_time
        return x


class Mammalian3(pyagingModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_cpg = x[:, :-1707]  # number of species in lookup table
        x_species = x[:, -1707:]  # number of species in lookup table
        x = self.base_model(x_cpg)
        x = self.postprocess(x, x_species)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x, x_species):
        """
        Converts output of to units of years.
        """
        indices = torch.argmax(x_species, dim=1)
        anage_array = self.postprocess_dependencies[0]
        anage_tensor = torch.tensor(anage_array, dtype=x.dtype, device=x.device)

        gestation_time = anage_tensor[indices, 0].unsqueeze(1)
        average_maturity_age = anage_tensor[indices, 1].unsqueeze(1)
        m_hat = 5 * (gestation_time / average_maturity_age) ** (0.38)

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        x_pos = x[mask_non_negative]
        x_neg = x[mask_negative]

        gestation_time_pos = gestation_time[mask_non_negative]
        gestation_time_neg = gestation_time[mask_negative]

        average_maturity_age_pos = average_maturity_age[mask_non_negative]
        average_maturity_age_neg = average_maturity_age[mask_negative]

        m_hat_pos = m_hat[mask_non_negative]
        m_hat_neg = m_hat[mask_negative]

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_non_negative] = (
            m_hat_pos * (average_maturity_age_pos + gestation_time_pos) * (x_pos + 1) - gestation_time_pos
        )

        # Linear transformation for non-negative values
        age_tensor[mask_negative] = (
            m_hat_neg * (average_maturity_age_neg + gestation_time_neg) * torch.exp(x_neg) - gestation_time_neg
        )

        return age_tensor


class MammalianBlood2(pyagingModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_cpg = x[:, :-1756]  # number of species in lookup table
        x_species = x[:, -1756:]  # number of species in lookup table
        x = self.base_model(x_cpg)
        x = self.postprocess(x, x_species)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x, x_species):
        """
        Converts output of relative age to age in units of years.
        """
        indices = torch.argmax(x_species, dim=1)
        anage_array = self.postprocess_dependencies[0]
        anage_tensor = torch.tensor(anage_array, dtype=x.dtype, device=x.device)
        gestation_time = anage_tensor[indices, 0].unsqueeze(1)
        max_age = anage_tensor[indices, 3].unsqueeze(1)

        x = torch.exp(-torch.exp(-x))
        x = x * (max_age + gestation_time) - gestation_time
        return x


class MammalianBlood3(pyagingModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_cpg = x[:, :-1707]  # number of species in lookup table
        x_species = x[:, -1707:]  # number of species in lookup table
        x = self.base_model(x_cpg)
        x = self.postprocess(x, x_species)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x, x_species):
        """
        Converts output of to units of years.
        """
        indices = torch.argmax(x_species, dim=1)
        anage_array = self.postprocess_dependencies[0]
        anage_tensor = torch.tensor(anage_array, dtype=x.dtype, device=x.device)

        gestation_time = anage_tensor[indices, 0].unsqueeze(1)
        average_maturity_age = anage_tensor[indices, 1].unsqueeze(1)
        m_hat = 5 * (gestation_time / average_maturity_age) ** (0.38)

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        x_pos = x[mask_non_negative]
        x_neg = x[mask_negative]

        gestation_time_pos = gestation_time[mask_non_negative]
        gestation_time_neg = gestation_time[mask_negative]

        average_maturity_age_pos = average_maturity_age[mask_non_negative]
        average_maturity_age_neg = average_maturity_age[mask_negative]

        m_hat_pos = m_hat[mask_non_negative]
        m_hat_neg = m_hat[mask_negative]

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_non_negative] = (
            m_hat_pos * (average_maturity_age_pos + gestation_time_pos) * (x_pos + 1) - gestation_time_pos
        )

        # Linear transformation for non-negative values
        age_tensor[mask_negative] = (
            m_hat_neg * (average_maturity_age_neg + gestation_time_neg) * torch.exp(x_neg) - gestation_time_neg
        )

        return age_tensor


class MammalianSkin2(pyagingModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_cpg = x[:, :-1756]  # number of species in lookup table
        x_species = x[:, -1756:]  # number of species in lookup table
        x = self.base_model(x_cpg)
        x = self.postprocess(x, x_species)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x, x_species):
        """
        Converts output of relative age to age in units of years.
        """
        indices = torch.argmax(x_species, dim=1)
        anage_array = self.postprocess_dependencies[0]
        anage_tensor = torch.tensor(anage_array, dtype=x.dtype, device=x.device)
        gestation_time = anage_tensor[indices, 0].unsqueeze(1)
        max_age = anage_tensor[indices, 3].unsqueeze(1)

        x = torch.exp(-torch.exp(-x))
        x = x * (max_age + gestation_time) - gestation_time
        return x


class MammalianSkin3(pyagingModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_cpg = x[:, :-1707]  # number of species in lookup table
        x_species = x[:, -1707:]  # number of species in lookup table
        x = self.base_model(x_cpg)
        x = self.postprocess(x, x_species)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x, x_species):
        """
        Converts output of to units of years.
        """
        indices = torch.argmax(x_species, dim=1)
        anage_array = self.postprocess_dependencies[0]
        anage_tensor = torch.tensor(anage_array, dtype=x.dtype, device=x.device)

        gestation_time = anage_tensor[indices, 0].unsqueeze(1)
        average_maturity_age = anage_tensor[indices, 1].unsqueeze(1)
        m_hat = 5 * (gestation_time / average_maturity_age) ** (0.38)

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        x_pos = x[mask_non_negative]
        x_neg = x[mask_negative]

        gestation_time_pos = gestation_time[mask_non_negative]
        gestation_time_neg = gestation_time[mask_negative]

        average_maturity_age_pos = average_maturity_age[mask_non_negative]
        average_maturity_age_neg = average_maturity_age[mask_negative]

        m_hat_pos = m_hat[mask_non_negative]
        m_hat_neg = m_hat[mask_negative]

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_non_negative] = (
            m_hat_pos * (average_maturity_age_pos + gestation_time_pos) * (x_pos + 1) - gestation_time_pos
        )

        # Linear transformation for non-negative values
        age_tensor[mask_negative] = (
            m_hat_neg * (average_maturity_age_neg + gestation_time_neg) * torch.exp(x_neg) - gestation_time_neg
        )

        return age_tensor


class MammalianLifespan(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-log transformation.
        """
        return torch.exp(x)


class MammalianFemale(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies a sigmoid transformation.
        """
        return torch.sigmoid(x)


class Meer(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Transforms age in days to age in months.
        """
        return x / 30.5


class OcampoATAC1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Normalize a PyTorch tensor of counts to TPM (Transcripts Per Million) then
        transforms with log1p.
        """

        lengths = torch.tensor(self.preprocess_dependencies[0], device=x.device, dtype=x.dtype)

        # Normalize by length
        tpm = 1000 * (x / lengths.unsqueeze(0))

        # Scale to TPM (Transcripts Per Million)
        tpm = 1e6 * (tpm / tpm.sum(dim=1, keepdim=True))

        # Apply log1p transformation
        tpm_log1p = torch.log1p(tpm)

        return tpm_log1p[:, self.preprocess_dependencies[1]]

    def postprocess(self, x):
        return x


class OcampoATAC2(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Normalize a PyTorch tensor of counts to TPM (Transcripts Per Million) then
        transforms with log1p.
        """
        lengths = torch.tensor(self.preprocess_dependencies[0], device=x.device, dtype=x.dtype)

        # Normalize by length
        tpm = 1000 * (x / lengths.unsqueeze(0))

        # Scale to TPM (Transcripts Per Million)
        tpm = 1e6 * (tpm / tpm.sum(dim=1, keepdim=True))

        # Apply log1p transformation
        tpm_log1p = torch.log1p(tpm)

        return tpm_log1p[:, self.preprocess_dependencies[1]]

    def postprocess(self, x):
        return x


class PCDNAmTL(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class PCGrimAge(pyagingModel):
    def __init__(self):
        super().__init__()

        self.center = nn.Parameter(torch.empty(78464), requires_grad=False)
        self.rotation = nn.Parameter(torch.empty((78464, 1933)), requires_grad=False)

        self.PCPACKYRS = None
        self.PCADM = None
        self.PCB2M = None
        self.PCCystatinC = None
        self.PCGDF15 = None
        self.PCLeptin = None
        self.PCPAI1 = None
        self.PCTIMP1 = None

        self.features_PCPACKYRS = None
        self.features_PCADM = None
        self.features_PCB2M = None
        self.features_PCCystatinC = None
        self.features_PCGDF15 = None
        self.features_PCLeptin = None
        self.features_PCPAI1 = None
        self.features_PCTIMP1 = None

    def forward(self, x):
        CpGs = x[:, :-2]
        Female = x[:, -2].unsqueeze(1)
        Age = x[:, -1].unsqueeze(1)

        CpGs = CpGs - self.center  # Apply centering
        PCs = torch.mm(CpGs, self.rotation)  # Apply PCA rotation

        x = torch.concat([PCs, Female, Age], dim=1)

        PCPACKYRS = self.PCPACKYRS(x[:, self.features_PCPACKYRS])
        PCADM = self.PCADM(x[:, self.features_PCADM])
        PCB2M = self.PCB2M(x[:, self.features_PCB2M])
        PCCystatinC = self.PCCystatinC(x[:, self.features_PCCystatinC])
        PCGDF15 = self.PCGDF15(x[:, self.features_PCGDF15])
        PCLeptin = self.PCLeptin(x[:, self.features_PCLeptin])
        PCPAI1 = self.PCPAI1(x[:, self.features_PCPAI1])
        PCTIMP1 = self.PCTIMP1(x[:, self.features_PCTIMP1])

        x = torch.concat(
            [
                PCPACKYRS,
                PCADM,
                PCB2M,
                PCCystatinC,
                PCGDF15,
                PCLeptin,
                PCPAI1,
                PCTIMP1,
                Age,
                Female,
            ],
            dim=1,
        )

        x = self.base_model(x)

        return x

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class PCHannum(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class PCHorvath2013(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class PCPhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class PCSkinAndBlood(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class Pasta(pyagingModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _rank_average(values):
        """
        Assign average ranks (1-based) per vector, handling ties.
        """
        sorted_vals, sorted_idx = torch.sort(values)
        ranks = torch.empty_like(sorted_vals, dtype=values.dtype)

        n = values.numel()
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_vals[end] == sorted_vals[start]:
                end += 1
            avg_rank = (start + end - 1) / 2.0 + 1.0
            ranks[sorted_idx[start:end]] = avg_rank
            start = end

        return ranks

    def preprocess(self, x):
        """
        Fill missing values with the global median then rank-normalize per sample.
        """
        median = torch.nanmedian(x)
        if torch.isnan(median):
            median = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x = torch.where(torch.isnan(x), median, x)

        ranked = torch.empty_like(x, dtype=x.dtype)
        for i in range(x.size(0)):
            ranked[i] = self._rank_average(x[i])

        return ranked

    def postprocess(self, x):
        """
        Apply linear scaling and shifting constants from the original Pasta definition.
        """
        scale = self.postprocess_dependencies[0]
        offset_factor = self.postprocess_dependencies[1]
        return x * scale + offset_factor * scale


class PastaMouse(Pasta):
    def __init__(self):
        super().__init__()
        self.base_model_features = None
        self.mouse_feature_indices = None
        self.full_reference_values = None

    def set_mouse_features(self, full_features, full_reference_values=None, mouse_prefix="ENSMUSG"):
        """
        Configure the mouse-only interface while keeping the full feature space for the base model.
        """
        self.base_model_features = list(full_features)
        self.full_reference_values = full_reference_values

        self.mouse_feature_indices = [
            i
            for i, feature in enumerate(self.base_model_features)
            if isinstance(feature, str) and feature.startswith(mouse_prefix)
        ]

        if len(self.mouse_feature_indices) == 0:
            raise ValueError("No mouse features were identified when configuring PastaMouse.")

        self.features = [self.base_model_features[i] for i in self.mouse_feature_indices]

        if self.full_reference_values is None:
            self.reference_values = None
        elif isinstance(self.full_reference_values, torch.Tensor):
            self.reference_values = self.full_reference_values[self.mouse_feature_indices].detach().clone()
        else:
            self.reference_values = [self.full_reference_values[i] for i in self.mouse_feature_indices]

    def _expand_with_reference(self, x):
        """
        Reconstruct the full 8113-length input expected by the base model by
        inserting reference values for human-only genes.
        """
        if self.base_model_features is None or self.mouse_feature_indices is None:
            raise ValueError("PastaMouse must be configured with set_mouse_features before inference.")

        if self.full_reference_values is None:
            ref_full = torch.zeros(len(self.base_model_features), device=x.device, dtype=x.dtype)
        elif isinstance(self.full_reference_values, torch.Tensor):
            ref_full = self.full_reference_values.to(device=x.device, dtype=x.dtype)
        else:
            ref_full = torch.tensor(self.full_reference_values, device=x.device, dtype=x.dtype)

        full_x = ref_full.unsqueeze(0).repeat(x.size(0), 1)
        full_x[:, self.mouse_feature_indices] = x
        return full_x

    def forward(self, x):
        # Build the full feature vector (mouse data + human reference values) before preprocessing.
        x_full = self._expand_with_reference(x)
        x_full = self.preprocess(x_full)
        x_full = self.base_model(x_full)
        x_full = self.postprocess(x_full)
        return x_full


class Reg(pyagingModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _rank_average(values):
        """
        Assign average ranks (1-based) per vector, handling ties.
        """
        sorted_vals, sorted_idx = torch.sort(values)
        ranks = torch.empty_like(sorted_vals, dtype=values.dtype)

        n = values.numel()
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_vals[end] == sorted_vals[start]:
                end += 1
            avg_rank = (start + end - 1) / 2.0 + 1.0
            ranks[sorted_idx[start:end]] = avg_rank
            start = end

        return ranks

    def preprocess(self, x):
        """
        Fill missing values with the global median then rank-normalize per sample.
        """
        median = torch.nanmedian(x)
        if torch.isnan(median):
            median = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x = torch.where(torch.isnan(x), median, x)

        ranked = torch.empty_like(x, dtype=x.dtype)
        for i in range(x.size(0)):
            ranked[i] = self._rank_average(x[i])

        return ranked

    def postprocess(self, x):
        """
        Add the REG intercept term after linear prediction.
        """
        intercept = self.postprocess_dependencies[0]
        return x + intercept


class McCartneySmoking(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class McCartneyBMI(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return torch.sigmoid(x)


class McCartneyEducation(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return torch.sigmoid(x)


class McCartneyTotalCholesterol(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return torch.sigmoid(x)


class McCartneyHDLCholesterol(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return torch.sigmoid(x)


class McCartneyLDLCholesterol(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return torch.sigmoid(x)


class McCartneyBodyFat(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return torch.sigmoid(x)


class DeconvolutionSingleCell(pyagingModel):
    def __init__(self):
        super().__init__()
        self.pseudo_inv = None
        self.cell_index = 0

    @staticmethod
    def _project_simplex(v):
        """
        Project a batch of vectors onto the probability simplex.
        """
        # v: (batch, n)
        sorted_v, _ = torch.sort(v, dim=1, descending=True)
        cssv = torch.cumsum(sorted_v, dim=1)
        inds = torch.arange(1, v.size(1) + 1, device=v.device, dtype=v.dtype)
        cond = sorted_v - (cssv - 1) / inds > 0
        rho = cond.sum(dim=1) - 1
        theta = (cssv[torch.arange(v.size(0)), rho] - 1) / (rho.to(v.dtype) + 1)
        w = torch.clamp(v - theta.unsqueeze(1), min=0)
        return w

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def forward(self, x):
        x = self.preprocess(x)

        if self.pseudo_inv is None:
            raise RuntimeError("pseudo_inv is not set for DeconvolutionSingleCell.")

        pseudo_inv = self.pseudo_inv.to(device=x.device, dtype=x.dtype)

        proportions = torch.matmul(x, pseudo_inv.T)
        proportions = self._project_simplex(proportions)

        cell = torch.index_select(proportions, 1, torch.tensor([self.cell_index], device=x.device))
        return self.postprocess(cell)

    def postprocess(self, x):
        return x


class DeconvoluteBloodEPIC(DeconvolutionSingleCell):
    def __init__(self):
        super().__init__()


class TwelveCellDeconvoluteBloodEPIC(DeconvolutionSingleCell):
    def __init__(self):
        super().__init__()


class DepressionBarbu(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class SexScoreBase(pyagingModel):
    def __init__(self):
        super().__init__()
        self.autosome_indices = None

    def preprocess(self, x):
        if self.autosome_indices is None:
            return x
        autosomes = torch.index_select(x, 1, self.autosome_indices.to(device=x.device))

        mask = ~torch.isnan(autosomes)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        sum_vals = torch.where(mask, autosomes, torch.zeros_like(autosomes)).sum(dim=1, keepdim=True)
        d_mean = sum_vals / count

        var = torch.where(mask, (autosomes - d_mean) ** 2, torch.zeros_like(autosomes)).sum(dim=1, keepdim=True)
        d_std = torch.sqrt(var / count)
        d_std = torch.where(d_std == 0, torch.ones_like(d_std), d_std)

        z = (x - d_mean) / d_std
        z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
        return z

    def postprocess(self, x):
        return x


class XChrom(SexScoreBase):
    def __init__(self):
        super().__init__()
        self.x_indices = None
        self.x_means = None
        self.x_coeffs = None

    def forward(self, x):
        z = self.preprocess(x)
        device = x.device
        dtype = x.dtype
        x_means = self.x_means.to(device=device, dtype=dtype)
        x_coeffs = self.x_coeffs.to(device=device, dtype=dtype)
        x_idx = self.x_indices.to(device=device)
        x_score = torch.sum((z.index_select(1, x_idx) - x_means) * x_coeffs, dim=1)
        return self.postprocess(x_score.unsqueeze(1))


class YChrom(SexScoreBase):
    def __init__(self):
        super().__init__()
        self.y_indices = None
        self.y_means = None
        self.y_coeffs = None

    def forward(self, x):
        z = self.preprocess(x)
        device = x.device
        dtype = x.dtype
        y_means = self.y_means.to(device=device, dtype=dtype)
        y_coeffs = self.y_coeffs.to(device=device, dtype=dtype)
        y_idx = self.y_indices.to(device=device)
        y_score = torch.sum((z.index_select(1, y_idx) - y_means) * y_coeffs, dim=1)
        return self.postprocess(y_score.unsqueeze(1))


class PedBE(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class Petkovich(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies a convertion from the output of an ElasticNet to mouse age in months.
        """
        a = 0.1666
        b = 0.4185
        c = -1.712
        age = ((x - c) / a) ** (1 / b)
        age = age / 30.5  # days to months
        return age


class PhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """Apply Levine's natural-log transform to C-reactive protein.

        The published coefficient is fit against ln(CRP in mg/dL); users supply
        the raw measurement so the same column can feed clocks that log it
        differently.

        Notes
        -----
        CRP is clamped to ``PHENOAGE_CRP_FLOOR_MG_DL`` before the log, so a zero
        — from a below-detection reading, a constant imputer, or a column the
        input never had — yields a finite age instead of ``-inf`` propagating
        into every downstream summary.
        """
        index = crp_index(self.features)
        crp = torch.clamp(x[:, index : index + 1], min=PHENOAGE_CRP_FLOOR_MG_DL)
        return torch.cat([x[:, :index], torch.log(crp), x[:, index + 1 :]], dim=1)

    def postprocess(self, x):
        """
        Applies a convertion from a CDF of the mortality score from a Gompertz
        distribution to phenotypic age.
        """
        # lambda
        lambda_ = torch.tensor(0.0192, device=x.device, dtype=x.dtype)
        mortality_score = 1 - torch.exp(-torch.exp(x) * (torch.exp(120 * lambda_) - 1) / lambda_)
        age = 141.50225 + torch.log(-0.00553 * torch.log(1 - mortality_score)) / 0.090165
        return age


class PhenoAgeSaoPaulo(pyagingModel):
    """PhenoAge refit without creatinine, albumin, and alkaline phosphatase."""

    def __init__(self):
        super().__init__()
        for name in ["m_n", "m_d", "ba_n", "ba_d", "ba_i"]:
            self.register_buffer(name, torch.empty(0))

    def preprocess(self, x):
        """Apply BioAge's log1p transform to C-reactive protein, not ``PhenoAge``'s ``ln``."""
        return log1p_crp(self.features, x)

    def postprocess(self, x):
        """Convert the Gompertz mortality score to phenotypic age.

        Notes
        -----
        The constants are refit alongside the coefficients and differ from
        Levine's published ones.
        """
        m_n, m_d, ba_n, ba_d, ba_i = (
            buffer.to(device=x.device, dtype=x.dtype)
            for buffer in (self.m_n, self.m_d, self.ba_n, self.ba_d, self.ba_i)
        )
        return mortality_to_phenoage_saopaulo(x, m_n, m_d, ba_n, ba_d, ba_i)


class RepliTali(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class SkinAndBlood(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class Stubbs(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Apply quantile normalization on x using gold standard means
        and then scale with the means and standard deviation.
        """

        gold_standard_means = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)

        # Ensure gold_standard_means is a 1D tensor and sorted
        sorted_gold_standard = torch.sort(gold_standard_means)[0]

        # Pre-compute the quantile indices
        quantile_indices = torch.linspace(0, len(sorted_gold_standard) - 1, steps=x.size(1)).long()

        # Prepare a tensor to hold normalized data
        normalized_data = torch.empty_like(x, device=x.device, dtype=x.dtype)

        for i in range(x.size(0)):
            sorted_indices = torch.argsort(x[i, :])
            normalized_data[i, sorted_indices] = sorted_gold_standard[quantile_indices]

        gold_standard_stds = torch.tensor(self.preprocess_dependencies[0], device=x.device, dtype=x.dtype)

        # Avoid division by zero in case of a column with constant value
        gold_standard_stds[torch.abs(gold_standard_stds) < 10e-10] = 1.0

        normalized_data = (normalized_data - gold_standard_means) / gold_standard_stds

        # Return only the subset from x that is used in the base model
        return normalized_data[:, self.preprocess_dependencies[1]]

    def postprocess(self, x):
        """
        Applies a convertion from the output of an ElasticNet to mouse age in months.
        """
        age = torch.exp(0.1207 * (x**2) + 1.2424 * x + 2.5440) - 3
        age = age * (7 / 30.5)  # weeks to months
        return age


class Thompson(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class ZhangBLUP(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Scales the input PyTorch tensor per row with mean 0 and std 1.
        """
        row_means = torch.mean(x, dim=1, keepdim=True)
        row_stds = torch.std(x, dim=1, keepdim=True)

        # Avoid division by zero in case of a row with constant value
        row_stds = torch.where(row_stds == 0, torch.ones_like(row_stds), row_stds)

        x_scaled = (x - row_means) / row_stds
        return x_scaled

    def postprocess(self, x):
        return x


class ZhangEN(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Scales the input PyTorch tensor per row with mean 0 and std 1.
        """
        row_means = torch.mean(x, dim=1, keepdim=True)
        row_stds = torch.std(x, dim=1, keepdim=True)

        # Avoid division by zero in case of a row with constant value
        row_stds = torch.where(row_stds == 0, torch.ones_like(row_stds), row_stds)

        x_scaled = (x - row_means) / row_stds
        return x_scaled

    def postprocess(self, x):
        return x


class ZhangMortality(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge(pyagingModel):
    def __init__(self):
        super().__init__()

        self.PACKYRS = None
        self.ADM = None
        self.B2M = None
        self.CystatinC = None
        self.GDF15 = None
        self.Leptin = None
        self.PAI1 = None
        self.TIMP1 = None

        self.features_PACKYRS = None
        self.features_ADM = None
        self.features_B2M = None
        self.features_CystatinC = None
        self.features_GDF15 = None
        self.features_Leptin = None
        self.features_PAI1 = None
        self.features_TIMP1 = None

    def forward(self, x):
        Female = x[:, -2].unsqueeze(1)
        Age = x[:, -1].unsqueeze(1)

        PACKYRS = self.PACKYRS(x[:, self.features_PACKYRS])
        ADM = self.ADM(x[:, self.features_ADM])
        B2M = self.B2M(x[:, self.features_B2M])
        CystatinC = self.CystatinC(x[:, self.features_CystatinC])
        GDF15 = self.GDF15(x[:, self.features_GDF15])
        Leptin = self.Leptin(x[:, self.features_Leptin])
        PAI1 = self.PAI1(x[:, self.features_PAI1])
        TIMP1 = self.TIMP1(x[:, self.features_TIMP1])

        x = torch.concat(
            [GDF15, B2M, CystatinC, TIMP1, ADM, PAI1, Leptin, PACKYRS, Age, Female],
            dim=1,
        )

        x = self.base_model(x)

        x = self.postprocess(x)

        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Converts from a Cox parameter to age in units of years.
        """
        cox_mean = 13.20127
        cox_std = 1.086805
        age_mean = 59.63951
        age_std = 9.049608

        # Normalize
        x = (x - cox_mean) / cox_std

        # Scale
        x = (x * age_std) + age_mean

        return x


class GrimAge2(pyagingModel):
    def __init__(self):
        super().__init__()

        self.PACKYRS = None
        self.ADM = None
        self.B2M = None
        self.CystatinC = None
        self.GDF15 = None
        self.Leptin = None
        self.PAI1 = None
        self.TIMP1 = None
        self.LogCRP = None
        self.A1C = None

        self.features_PACKYRS = None
        self.features_ADM = None
        self.features_B2M = None
        self.features_CystatinC = None
        self.features_GDF15 = None
        self.features_Leptin = None
        self.features_PAI1 = None
        self.features_TIMP1 = None
        self.features_LogCRP = None
        self.features_A1C = None

    def forward(self, x):
        Female = x[:, -2].unsqueeze(1)
        Age = x[:, -1].unsqueeze(1)

        PACKYRS = self.PACKYRS(x[:, self.features_PACKYRS])
        ADM = self.ADM(x[:, self.features_ADM])
        B2M = self.B2M(x[:, self.features_B2M])
        CystatinC = self.CystatinC(x[:, self.features_CystatinC])
        GDF15 = self.GDF15(x[:, self.features_GDF15])
        Leptin = self.Leptin(x[:, self.features_Leptin])
        PAI1 = self.PAI1(x[:, self.features_PAI1])
        TIMP1 = self.TIMP1(x[:, self.features_TIMP1])
        LogCRP = self.LogCRP(x[:, self.features_LogCRP])
        A1C = self.A1C(x[:, self.features_A1C])

        x = torch.concat(
            [
                GDF15,
                B2M,
                CystatinC,
                TIMP1,
                ADM,
                PAI1,
                Leptin,
                PACKYRS,
                LogCRP,
                A1C,
                Age,
                Female,
            ],
            dim=1,
        )

        x = self.base_model(x)

        x = self.postprocess(x)

        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Converts from a Cox parameter to age in units of years.
        """
        cox_mean = 15.370829484122
        cox_std = 1.09534876966487
        age_mean = 66.0943807965085
        age_std = 9.05974444998421

        # Normalize
        x = (x - cox_mean) / cox_std

        # Scale
        x = (x * age_std) + age_mean

        return x


class YingCausAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class YingDamAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class YingAdaptAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmFitAge(pyagingModel):
    def __init__(self):
        super().__init__()

        self.Gait = None
        self.Grip = None
        self.VO2Max = None
        self.GrimAge = None

        self.features_Gait = None
        self.features_Grip = None
        self.features_VO2Max = None
        self.features_GrimAge = None
        self.female_index = None
        self.age_index = None

    def _component_input(self, x, indices, component):
        values = x[:, indices]
        return _fill_dnamfitage_references(values, component.reference_values)

    def forward(self, x):
        female = x[:, self.female_index].unsqueeze(1)
        gait = self.Gait(self._component_input(x, self.features_Gait, self.Gait))
        grip = self.Grip(self._component_input(x, self.features_Grip, self.Grip))
        vo2max = self.VO2Max(self._component_input(x, self.features_VO2Max, self.VO2Max))
        grimage = self.GrimAge(self._component_input(x, self.features_GrimAge, self.GrimAge))

        female_components = torch.concat(
            [
                (vo2max - 46.825091) / -0.13620215,
                (grip - 39.857718) / -0.22074456,
                (gait - 2.508547) / -0.01245682,
                (grimage - 7.978487) / 0.80928530,
            ],
            dim=1,
        )
        male_components = torch.concat(
            [
                (vo2max - 49.836389) / -0.141862925,
                (grip - 57.514016) / -0.253179827,
                (gait - 2.349080) / -0.009380061,
                (grimage - 9.549733) / 0.835120557,
            ],
            dim=1,
        )
        female_prediction = self.base_model_f(female_components)
        male_prediction = self.base_model_m(male_components)
        return male_prediction + female * (female_prediction - male_prediction)

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


def _fill_dnamfitage_references(x, reference_values):
    reference = torch.as_tensor(reference_values, device=x.device, dtype=x.dtype)
    return torch.where(torch.isnan(x), reference, x)


class _DNAmFitAgeSexGated(pyagingModel):
    def __init__(self):
        super().__init__()
        self.female_model = None
        self.male_model = None
        self.female_feature_indices = None
        self.male_feature_indices = None
        self.female_reference_values = None
        self.male_reference_values = None
        self.female_index = None

    def forward(self, x):
        female = x[:, self.female_index].unsqueeze(1)
        female_x = _fill_dnamfitage_references(x[:, self.female_feature_indices], self.female_reference_values)
        male_x = _fill_dnamfitage_references(x[:, self.male_feature_indices], self.male_reference_values)
        female_prediction = self.female_model(female_x)
        male_prediction = self.male_model(male_x)
        return male_prediction + female * (female_prediction - male_prediction)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class DNAmFitAgeGait(_DNAmFitAgeSexGated):
    pass


class DNAmFitAgeGrip(_DNAmFitAgeSexGated):
    pass


class StocH(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class StocZ(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class StocP(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class stemTOC(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        # Filter out -1 values per row and calculate the 0.95 quantile per row
        quantiles = []
        for row in x:
            filtered_row = row[row != -1]
            quantile_95 = torch.quantile(filtered_row, 0.95) if len(filtered_row) > 0 else torch.tensor(float("nan"))
            quantiles.append(quantile_95)
        return torch.vstack(quantiles)

    def postprocess(self, x):
        return x


class epiTOC1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        # Filter out -1 values per row and calculate the mean per row
        means = []
        for row in x:
            filtered_row = row[row != -1]
            mean = torch.mean(filtered_row) if len(filtered_row) > 0 else torch.tensor(float("nan"))
            means.append(mean)
        return torch.vstack(means)

    def postprocess(self, x):
        return x


class epiTOC2(pyagingModel):
    def __init__(self):
        super().__init__()
        self.delta = None
        self.beta0 = None

    def preprocess(self, x):
        """
        Replace NaNs with zero; missing features should already be imputed via reference_values.
        """
        return torch.nan_to_num(x, nan=0.0)

    def forward(self, x):
        x = self.preprocess(x)

        device = x.device
        dtype = x.dtype

        delta = self.delta.to(device=device, dtype=dtype)
        beta0 = self.beta0.to(device=device, dtype=dtype)

        denom = delta * (1 - beta0)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        contrib = (x - beta0) / denom
        k = contrib.size(1)
        vals = 2.0 * torch.sum(contrib, dim=1) / k

        return self.postprocess(vals.unsqueeze(1))

    def postprocess(self, x):
        return x


class RetroelementAgeV1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class RetroelementAgeV2(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class IntrinClock(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class ABEC(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class eABEC(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class cABEC(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class PipekElasticNet(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class PipekFilteredH(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class PipekRetrainedH(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies an anti-logarithmic linear transformation to a PyTorch tensor.
        """
        adult_age = 20

        # Create a mask for negative and non-negative values
        mask_negative = x < 0
        mask_non_negative = ~mask_negative

        # Initialize the result tensor
        age_tensor = torch.empty_like(x)

        # Exponential transformation for negative values
        age_tensor[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1

        # Linear transformation for non-negative values
        age_tensor[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age

        return age_tensor


class GrimAge2ADM(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2B2M(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2CystatinC(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2GDF15(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2Leptin(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2PackYrs(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2PAI1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2TIMP1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2LogA1C(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class GrimAge2LogCRP(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmFitAgeGaitF(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmFitAgeGaitM(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmFitAgeGripF(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmFitAgeGripM(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class DNAmFitAgeVO2Max(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class CpGPTGrimAge3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        """
        Scales an array based on the median and standard deviation.
        """
        median = torch.tensor(self.preprocess_dependencies[0], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.preprocess_dependencies[1], device=x.device, dtype=x.dtype)
        x = (x - median) / std
        return x

    def postprocess(self, x):
        """
        Converts from a Cox parameter to age in units of years.
        """
        cox_mean = self.postprocess_dependencies[0]
        cox_std = self.postprocess_dependencies[1]
        age_mean = self.postprocess_dependencies[2]
        age_std = self.postprocess_dependencies[3]

        # Normalize
        x = (x - cox_mean) / cox_std

        # Scale
        x = (x * age_std) + age_mean

        return x


class CpGPTPCGrimAge3(pyagingModel):
    def __init__(self):
        super().__init__()

        self.rotation = nn.Parameter(torch.empty((30, 29)), requires_grad=False)

    def preprocess(self, x):
        """
        Scales an array based on the mean and standard deviation.
        """
        mean = torch.tensor(self.preprocess_dependencies[0], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.preprocess_dependencies[1], device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return x

    def forward(self, x):
        x = self.preprocess(x)

        age = x[:, 0].unsqueeze(1)
        proxies = x[:, 1:]

        PCs = torch.mm(proxies, self.rotation)  # Apply PCA rotation

        x = torch.concat([age, PCs], dim=1)

        # Scale
        mean = torch.tensor(self.preprocess_dependencies[2], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.preprocess_dependencies[3], device=x.device, dtype=x.dtype)
        x[:, 1:] = (x[:, 1:] - mean) / std

        x = self.base_model(x)

        x = self.postprocess(x)

        return x

    def postprocess(self, x):
        """
        Converts from a Cox parameter to age in units of years.
        """
        cox_mean = self.postprocess_dependencies[0]
        cox_std = self.postprocess_dependencies[1]
        age_mean = self.postprocess_dependencies[2]
        age_std = self.postprocess_dependencies[3]

        # Normalize
        x = (x - cox_mean) / cox_std

        # Scale
        x = (x * age_std) + age_mean

        return x


class EnsembleAgeStatic(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class EnsembleAgeStaticTop(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class SystemsAgeBase(pyagingModel):
    def __init__(self):
        super().__init__()

        self.prediction_index = None

        # DNAm PCA assets
        self.dnam_center = None
        self.dnam_rotation = None
        self.system_vector = None

        # System aggregation assets
        self.system_labels = []
        self.system_component_indices = []
        self.system_modules = nn.ModuleList()

        # Predicted age assets
        self.predicted_age_model = None
        self.predicted_age_poly = None

        # Systems PCA assets
        self.systems_pca_model = None

        # Transformation assets
        self.transformation_coefs = None
        self.transformation_labels = None

    @staticmethod
    def _as_tensor(value, device, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    def _ensure_loaded(self):
        required = [
            self.dnam_center,
            self.dnam_rotation,
            self.system_vector,
            self.predicted_age_poly,
            self.transformation_coefs,
        ]
        if any(asset is None for asset in required):
            raise RuntimeError(
                "SystemsAge parameters are not fully loaded. Please populate the model "
                "attributes using the notebook export assets before calling forward()."
            )
        if (not self.system_modules) or (not self.system_component_indices):
            raise RuntimeError(
                "SystemsAge system aggregation modules are missing. Populate "
                "system_modules and system_component_indices before use."
            )
        if self.predicted_age_model is None:
            raise RuntimeError("SystemsAge predicted age model not initialised.")
        if self.systems_pca_model is None:
            raise RuntimeError("SystemsAge systems PCA model not initialised.")

    def forward(self, x):
        self._ensure_loaded()

        x = self.preprocess(x)

        device = x.device
        dtype = x.dtype

        dnam_center = self._as_tensor(self.dnam_center, device, dtype)
        dnam_rotation = self._as_tensor(self.dnam_rotation, device, dtype)
        system_vector = self._as_tensor(self.system_vector, device, dtype)

        centered = x - dnam_center
        dnam_pcs = centered @ dnam_rotation

        system_components = dnam_pcs @ system_vector

        system_scores = []
        for module, indices in zip(self.system_modules, self.system_component_indices, strict=False):
            comps = torch.index_select(
                system_components,
                1,
                indices.to(device=device),
            )
            system_scores.append(module(comps))
        system_scores = torch.cat(system_scores, dim=1)

        predicted_age = self.predicted_age_model(dnam_pcs).squeeze(-1)
        predicted_age_poly = self._as_tensor(self.predicted_age_poly, device, dtype)
        predicted_age = (
            predicted_age * predicted_age_poly[1] + predicted_age.pow(2) * predicted_age_poly[2] + predicted_age_poly[0]
        )
        predicted_age = predicted_age / 12.0

        base_outputs = torch.cat([system_scores, predicted_age.unsqueeze(-1)], dim=1)

        if self.prediction_index == -1:
            raw_output = self.systems_pca_model(base_outputs).squeeze(-1)
            transform_idx = self.transformation_coefs.shape[0] - 1
        else:
            raw_output = base_outputs[:, self.prediction_index]
            transform_idx = self.prediction_index

        transformation_coefs = self._as_tensor(self.transformation_coefs, device, dtype)
        coef = transformation_coefs[transform_idx]
        transformed = ((raw_output - coef[0]) / coef[1]) * coef[3] + coef[2]
        transformed = transformed / 12.0

        return self.postprocess(transformed.unsqueeze(-1))

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class SystemsAgeBlood(SystemsAgeBase):
    def __init__(self):
        super().__init__()

        self.prediction_index = 0


class SystemsAgeBrain(SystemsAgeBase):
    def __init__(self):
        super().__init__()

        self.prediction_index = 1


class SystemsAgeInflammation(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 2


class SystemsAgeHeart(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 3


class SystemsAgeHormone(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 4


class SystemsAgeImmune(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 5


class SystemsAgeKidney(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 6


class SystemsAgeLiver(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 7


class SystemsAgeMetabolic(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 8


class SystemsAgeLung(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 9


class SystemsAgeMusculoSkeletal(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = 10


class SystemsAge(SystemsAgeBase):
    def __init__(self):
        super().__init__()
        self.prediction_index = -1


class LinearReferenceClock(pyagingModel):
    """
    Base class for linear methylation clocks that impute missing features
    from reference_values (when provided) and apply no output transform.
    """

    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class VidalBralo(LinearReferenceClock):
    pass


class McCartneyAlcohol(LinearReferenceClock):
    pass


class Weidner(LinearReferenceClock):
    pass


class Garagnani(LinearReferenceClock):
    pass


class Bocklandt(LinearReferenceClock):
    pass


class DNAmFILi(LinearReferenceClock):
    pass


class DNAmStress(LinearReferenceClock):
    pass


class CellPopAge(LinearReferenceClock):
    pass


class RepliTaliNorm(LinearReferenceClock):
    pass


class SenChronoAge(LinearReferenceClock):
    pass


class SenCultureAge(LinearReferenceClock):
    pass


class SenMortalityAge(LinearReferenceClock):
    pass


class DunedinPoAm38(LinearReferenceClock):
    pass


class ReedBMI(LinearReferenceClock):
    pass


class DownSyndrome(LinearReferenceClock):
    pass


class ProstateCancerKirby(LinearReferenceClock):
    pass


class HepatoXu(LinearReferenceClock):
    pass


class McCartneyTotalHDLRatio(LinearReferenceClock):
    pass


class McCartneyWHR(LinearReferenceClock):
    pass


class CompIL6(LinearReferenceClock):
    pass


class NeuSin(LinearReferenceClock):
    pass


class GliaSin(LinearReferenceClock):
    pass


class Hep(LinearReferenceClock):
    pass


class CTSLiver(LinearReferenceClock):
    pass


class EnsembleAgeHumanMouse(LinearReferenceClock):
    pass


class CorticalClock(LinearReferenceClock):
    def postprocess(self, x):
        """Horvath anti-logarithmic linear transformation (adult age = 20)."""
        adult_age = 20
        mask_negative = x < 0
        mask_non_negative = ~mask_negative
        age = torch.empty_like(x)
        age[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1
        age[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age
        return age


class Bohlin(LinearReferenceClock):
    def postprocess(self, x):
        """Model returns gestational age in days; convert to weeks."""
        return x / 7.0


class Mayne(LinearReferenceClock):
    def postprocess(self, x):
        """Gestational age in weeks (as published)."""
        return x


class EPICGA(LinearReferenceClock):
    def postprocess(self, x):
        """Model returns gestational age in days; convert to weeks."""
        return x / 7.0


class Wu(LinearReferenceClock):
    def postprocess(self, x):
        """Horvath anti-log (adult age = 48) giving months, then months to years."""
        adult_age = 48
        mask_negative = x < 0
        mask_non_negative = ~mask_negative
        age = torch.empty_like(x)
        age[mask_negative] = (1 + adult_age) * torch.exp(x[mask_negative]) - 1
        age[mask_non_negative] = (1 + adult_age) * x[mask_non_negative] + adult_age
        return age / 12.0


class CVDWesterman(LinearReferenceClock):
    def postprocess(self, x):
        """Logistic transform to a CVD risk probability."""
        return torch.sigmoid(x)


class ADBahadoSingh(LinearReferenceClock):
    def postprocess(self, x):
        """Logistic transform to an Alzheimer's disease risk probability.

        The published logit constant (-0.072) is carried in the linear intercept.
        """
        return torch.sigmoid(x)


class StemTOCvitro(stemTOC):
    pass


class EpiCMITHyper(epiTOC1):
    pass


class EpiCMITHypo(epiTOC1):
    pass


class EpiTOC3(epiTOC2):
    pass


class TAge(pyagingModel):
    def __init__(self):
        super().__init__()
        # Cohort-relative clock: ``predict_age`` runs the tAge cohort
        # preprocessing on the raw counts before aligning these features.
        self.cohort_transform = "tage"
        self.feature_range_data_type = "transcriptomics (relative)"

    def preprocess(self, x):
        """Substitute the fitted imputer statistics for genes absent from the sample.

        ``reference_values`` holds the published pipeline's ``SimpleImputer`` fill
        values, so a gene the aligned list carries but the sample does not gets the
        exact value the model was fitted to expect. For a cohort-centered clock that
        is the least-biased choice available: the gene contributes its training median
        rather than pulling the score toward zero or toward this cohort's center.
        """
        if self.reference_values is None:
            return x
        if isinstance(self.reference_values, torch.Tensor):
            reference = self.reference_values.to(device=x.device, dtype=x.dtype)
        else:
            reference = torch.tensor(self.reference_values, device=x.device, dtype=x.dtype)
        return torch.where(torch.isnan(x), reference, x)

    def postprocess(self, x):
        return x


class TAgeMortality(TAge):
    """Mortality head of the same pipeline family, with the same cohort-relative input contract."""

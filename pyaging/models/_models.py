import torch
import torch.nn as nn

from ._base_models import *


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
        return x

    def postprocess(self, x):
        return x


class CamilloH3K27me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class CamilloH3K36me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class CamilloH3K4me1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class CamilloH3K4me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class CamilloH3K9ac(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class CamilloH3K9me3(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class CamilloPanHistone(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class DNAmPhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class DNAmTL(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

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
        return x

    def postprocess(self, x):
        return x


class ENCen40(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

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
        return x

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


class HRSInCHPhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class Knight(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class LeeControl(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class LeeRefinedRobust(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class LeeRobust(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class Lin(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


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

        self.rotation = nn.Parameter(torch.empty(78464), requires_grad=False)
        self.center = nn.Parameter(torch.empty((78464, 1933)), requires_grad=False)

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
        return x

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
        return x

    def postprocess(self, x):
        """
        Applies a convertion from a CDF of the mortality score from a Gompertz
        distribution to phenotypic age.
        """
        # lambda
        l = torch.tensor(0.0192, device=x.device, dtype=x.dtype)
        mortality_score = 1 - torch.exp(-torch.exp(x) * (torch.exp(120 * l) - 1) / l)
        age = 141.50225 + torch.log(-0.00553 * torch.log(1 - mortality_score)) / 0.090165
        return age


class RepliTali(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

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
        return x

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
        return x

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
        return x

    def postprocess(self, x):
        return x


class YingDamAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class YingAdaptAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class DNAmFitAge(pyagingModel):
    def __init__(self):
        super().__init__()

        self.GaitF = None
        self.GripF = None
        self.GaitM = None
        self.GripM = None
        self.VO2Max = None

        self.features_GaitF = None
        self.features_GripF = None
        self.features_GaitM = None
        self.features_GripM = None
        self.features_VO2Max = None

    def forward(self, x):
        Female = x[:, -3]  # .unsqueeze(1)
        Age = x[:, -2]  # .unsqueeze(1)
        GrimAge = x[:, -1].unsqueeze(1)

        is_female = Female == 1
        is_male = Female == 0

        x_f = x[is_female]
        x_m = x[is_male]

        GaitF = self.GaitF(x_f[:, self.features_GaitF])
        GripF = self.GripF(x_f[:, self.features_GripF])
        VO2MaxF = self.VO2Max(x_f[:, self.features_VO2Max])
        GrimAgeF = GrimAge[is_female, :]

        GaitM = self.GaitM(x_m[:, self.features_GaitM])
        GripM = self.GripM(x_m[:, self.features_GripM])
        VO2MaxM = self.VO2Max(x_m[:, self.features_VO2Max])
        GrimAgeM = GrimAge[is_male, :]

        x_f = torch.concat(
            [
                (VO2MaxF - 46.825091) / (-0.13620215),
                (GripF - 39.857718) / (-0.22074456),
                (GaitF - 2.508547) / (-0.01245682),
                (GrimAgeF - 7.978487) / (0.80928530),
            ],
            dim=1,
        )

        x_m = torch.concat(
            [
                (VO2MaxM - 49.836389) / (-0.141862925),
                (GripM - 57.514016) / (-0.253179827),
                (GaitM - 2.349080) / (-0.009380061),
                (GrimAgeM - 9.549733) / (0.835120557),
            ],
            dim=1,
        )

        y_f = self.base_model_f(x_f)
        y_m = self.base_model_m(x_m)

        y = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        y[is_female] = y_f
        y[is_male] = y_m

        return y

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class StocH(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class StocZ(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class StocP(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

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
            if len(filtered_row) > 0:
                quantile_95 = torch.quantile(filtered_row, 0.95)
            else:
                quantile_95 = torch.tensor(float("nan"))
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
            if len(filtered_row) > 0:
                mean = torch.mean(filtered_row)
            else:
                mean = torch.tensor(float("nan"))
            means.append(mean)
        return torch.vstack(means)

    def postprocess(self, x):
        return x


class RetroelementAgeV1(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class RetroelementAgeV2(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

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

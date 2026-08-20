import math

import torch


def anti_log_linear(x, adult_age=20):
    """
    Applies an anti-logarithmic linear transformation to a value.
    """
    if x < 0:
        # Apply exponential transformation for negative values
        return (1 + adult_age) * math.exp(x) - 1
    else:
        # Apply linear transformation for non-negative values
        return (1 + adult_age) * x + adult_age


def anti_logp2(x):
    """
    Applies an anti-logarithmic transformation with an offset of -2.
    """
    # Exponential transformation with an offset
    return math.exp(x) - 2


def anti_log(x):
    """
    Applies a simple anti-logarithmic transformation.
    """
    # Simple exponential transformation
    return math.exp(x)


def anti_log_log(x):
    """
    Applies a double transformation: logarithmic followed by anti-logarithmic.
    """
    # Double transformation: logarithmic followed by anti-logarithmic
    return math.exp(-math.exp(-x))


def mortality_to_phenoage(x):
    """
    Applies a convertion from a CDF of the mortality score from a Gompertz
    distribution to phenotypic age.
    """
    # lambda
    lambda_ = 0.0192
    mortality_score = 1 - math.exp(-math.exp(x) * (math.exp(120 * lambda_) - 1) / lambda_)
    age = 141.50225 + math.log(-0.00553 * math.log(1 - mortality_score)) / 0.090165
    return age


def mortality_to_phenoage_saopaulo(x, m_n, m_d, ba_n, ba_d, ba_i):
    """
    Applies a convertion from a CDF of the mortality score from a Gompertz
    distribution to phenotypic age, using constants supplied by the caller.

    ``mortality_to_phenoage`` hardcodes Levine's 2018 published constants. A
    refit of that model fits its own mortality-to-age constants alongside its
    coefficients, so they are passed in rather than baked in.

    Parameters
    ----------
    x : torch.Tensor
        The linear predictor, one value per sample.

    m_n, m_d : torch.Tensor
        Numerator and denominator of the Gompertz mortality exponent.

    ba_n, ba_d, ba_i : torch.Tensor
        Scale, rate, and intercept converting mortality back to years.

    Returns
    -------
    torch.Tensor
        Phenotypic age in years.
    """
    mortality_score = 1 - torch.exp(m_n * torch.exp(x) / m_d)
    return torch.log(ba_n * torch.log(1 - mortality_score)) / ba_d + ba_i


def petkovichblood(x):
    """
    Applies a convertion from the output of an ElasticNet to mouse age in months.
    """
    a = 0.1666
    b = 0.4185
    c = -1.712
    age = ((x - c) / a) ** (1 / b)
    age = age / 30.5  # days to months
    return age


def stubbsmultitissue(x):
    """
    Applies a convertion from the output of an ElasticNet to mouse age in months.
    """
    age = math.exp(0.1207 * (x**2) + 1.2424 * x + 2.5440) - 3
    age = age * (7 / 30.5)  # weeks to months
    return age

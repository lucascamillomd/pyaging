import math


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
    l = 0.0192
    mortality_score = 1 - math.exp(-math.exp(x) * (math.exp(120 * l) - 1) / l)
    age = 141.50225 + math.log(-0.00553 * math.log(1 - mortality_score)) / 0.090165
    return age

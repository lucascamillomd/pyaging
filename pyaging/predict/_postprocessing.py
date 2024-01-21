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

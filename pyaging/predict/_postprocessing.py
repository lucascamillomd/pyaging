import math


def anti_log_linear(x, adult_age=20):
    """
    Applies an anti-logarithmic linear transformation to a value.

    Args:
    - x: The value to be transformed.
    - adult_age: A constant used in the transformation, default is 20.

    Returns:
    - Transformed value using a linear approach for positive values and exponential for negative values.
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

    Args:
    - x: The value to be transformed.

    Returns:
    - The exponential of x, decreased by 2.
    """
    # Exponential transformation with an offset
    return math.exp(x) - 2


def anti_log(x):
    """
    Applies a simple anti-logarithmic transformation.

    Args:
    - x: The value to be transformed.

    Returns:
    - The exponential of x.
    """
    # Simple exponential transformation
    return math.exp(x)


def anti_log_log(x):
    """
    Applies a double transformation: logarithmic followed by anti-logarithmic.

    Args:
    - x: The value to be transformed.

    Returns:
    - The exponential of the negative exponential of x.
    """
    # Double transformation: logarithmic followed by anti-logarithmic
    return math.exp(-math.exp(-x))

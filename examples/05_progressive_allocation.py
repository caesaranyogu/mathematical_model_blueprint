"""Example 5: A family of allocation functions.

This is a modeling exercise, NOT a claim that any strategy is profitable.
It intentionally leaves out real-world trading frictions and risk controls.
"""

def allocation_linear(drawdown, scale):
    """Capital grows linearly with absolute drawdown."""
    if drawdown >= 0:
        return 0
    return scale * (-drawdown)


def allocation_quadratic(drawdown, scale):
    """Capital grows quadratically with absolute drawdown."""
    if drawdown >= 0:
        return 0
    return scale * (-drawdown) ** 2


def allocation_power(drawdown, scale, exponent):
    """General power-law allocation."""
    if drawdown >= 0:
        return 0
    return scale * (-drawdown) ** exponent


if __name__ == "__main__":
    reference_high = 100
    scale = 10_000

    for price in [95, 90, 80, 70, 60]:
        drawdown = (price - reference_high) / reference_high

        print(
            f"price={price}, "
            f"drawdown={drawdown:.1%}, "
            f"linear=${allocation_linear(drawdown, scale):,.2f}, "
            f"quadratic=${allocation_quadratic(drawdown, scale):,.2f}"
        )

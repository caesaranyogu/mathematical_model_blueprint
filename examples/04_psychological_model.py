"""Example 4: A simple logistic behavioral model."""

import numpy as np


def action_probability(
    reward,
    effort,
    trust,
    beta0,
    beta_reward,
    beta_effort,
    beta_trust,
):
    """
    P(action) =
        1 / (1 + exp(-(beta0
                       + beta_reward*reward
                       - beta_effort*effort
                       + beta_trust*trust)))
    """

    z = (
        beta0
        + beta_reward * reward
        - beta_effort * effort
        + beta_trust * trust
    )

    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    p = action_probability(
        reward=8,
        effort=3,
        trust=7,
        beta0=-2,
        beta_reward=0.4,
        beta_effort=0.7,
        beta_trust=0.3,
    )

    print("Probability of action:", p)

"""Example 3: Monte Carlo business simulation."""

import numpy as np

rng = np.random.default_rng(42)

N = 10_000

customers = rng.normal(
    loc=1_000,
    scale=150,
    size=N
)

price = rng.normal(
    loc=50,
    scale=5,
    size=N
)

cost = rng.normal(
    loc=30_000,
    scale=4_000,
    size=N
)

revenue = customers * price
profit = revenue - cost

print("Expected profit:", profit.mean())
print("5th percentile:", np.percentile(profit, 5))
print("Median:", np.percentile(profit, 50))
print("95th percentile:", np.percentile(profit, 95))

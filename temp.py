from scipy.stats import binom
import numpy as np

# Probability of X ≤ 7
p_low = binom.cdf(7, 15, 0.25)

# Expected value for X > 7
high_sum = sum(k/15 * binom.pmf(k, 15, 0.25) for k in range(8, 16))

expected_score = 0.5 * p_low + high_sum

print(f"Expected score: {expected_score:.4f}")
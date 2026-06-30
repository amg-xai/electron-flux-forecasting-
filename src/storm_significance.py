import numpy as np
from scipy.stats import binomtest

# If we randomly guessed "alert" at any point in a ~10-day pre-storm window,
# what's the chance of landing in a "lead time positive" zone by chance?

# Our actual result: 4 out of 5 storms showed positive lead time
successes = 4
trials = 5

# Null hypothesis: 50% chance of random positive lead time (conservative assumption)
result = binomtest(successes, trials, p=0.5, alternative='greater')
print(f"Binomial test: {successes}/{trials} successes")
print(f"P-value (vs random 50% baseline): {result.pvalue:.4f}")

# More conservative null: classifier alerts are uniformly distributed in time,
# probability of landing before threshold crossing by chance
# Average storm window ~10 days, average lead time needed ~6-13h
# P(random alert falls in the lead window) ≈ lead_hours / window_hours
window_hours = 10 * 24
lead_hours = [8, 4, 6]  # the 3 clearly positive cases, excluding boundary case
p_by_chance = np.mean(lead_hours) / window_hours
print(f"\nEstimated P(random alert in pre-storm window by chance): {p_by_chance:.3f}")
print(f"Our classifier achieved this in 4/5 cases — binomial p-value vs this baseline:")
result2 = binomtest(successes, trials, p=p_by_chance, alternative='greater')
print(f"P-value: {result2.pvalue:.6f}")
"""
Exercise 3: Uniform vs. Distribution-Aware Quantization
=========================================================

Course: TurboQuant — Near-Optimal Vector Quantization from Theory to Practice
Module: 0 — The Vector Quantization Problem

Goal
----
In Exercises 1 and 2 you measured how badly the uniform quantizer performs on
unit sphere vectors.  At b=1, the uniform quantizer achieved MSE ≈ 52 (vs
TurboQuant's bound of 0.36) — a 145× gap — because it places centroids at ±0.5
while sphere coordinates live near 0 with std ≈ 0.0625.

Now you will implement a smarter baseline: the EQUIPROBABLE quantizer.
Instead of equal-width buckets, equiprobable quantization places boundaries so
that each bucket captures equal probability mass.  For a Gaussian distribution
N(0, σ²), this means boundaries at the quantiles of the distribution.

This is a stepping stone toward the Lloyd-Max optimal quantizer (Module 1).
Equiprobable quantization gets bucket placement right (equal mass) but uses
bucket MIDPOINTS as centroids, not conditional means.  Lloyd-Max optimizes
BOTH boundaries and centroids simultaneously for minimum MSE.

Key insight of this exercise:
    "The key insight of TurboQuant is that random rotation CREATES a known
     distribution (Beta/Gaussian), enabling precomputed optimal codebooks."
    — Module 0 Lesson

By using the KNOWN distribution of sphere coordinates (Beta → Gaussian in high
d), we can precompute better codebooks.  This exercise demonstrates the first
step of that improvement.

Background from Exercise 1:
  - Uniform quantizer at b=1 on d=256 sphere vectors: MSE ≈ 52.2
  - Uniform quantizer at b=4: MSE ≈ 0.34
  - These are the baselines you will now beat.

Background from Exercise 2:
  - b=1 magnitude bias: 7.99× inflation (centroids at ±0.5, data at std≈0.06)
  - The distribution-aware quantizer will dramatically reduce this bias.

Dependencies
------------
    pip install numpy scipy

Usage
-----
    python ex03_uniform_vs_distributionaware_quantization.py
"""

import numpy as np
from scipy import stats, special


# ---------------------------------------------------------------------------
# PROVIDED: Uniform quantizer from Exercise 1
# ---------------------------------------------------------------------------

def uniform_quantize(x: np.ndarray, b: int) -> np.ndarray:
    """Partition [-1,1] into 2^b equal buckets; return bucket indices.

    Parameters
    ----------
    x : np.ndarray
        Values to quantize, assumed in [-1, 1].
    b : int
        Bit-width; produces 2^b bucket levels.

    Returns
    -------
    np.ndarray of int, same shape as x
        Bucket indices in [0, 2^b - 1].
    """
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    indices = np.floor((x + 1.0) / delta).astype(int)
    return np.clip(indices, 0, n_buckets - 1)


def uniform_dequantize(indices: np.ndarray, b: int) -> np.ndarray:
    """Map bucket indices to bucket centroids.

    Parameters
    ----------
    indices : np.ndarray of int
        Bucket indices in [0, 2^b - 1].
    b : int
        Bit-width used during quantization.

    Returns
    -------
    np.ndarray of float, same shape as indices
        Bucket centroid values (midpoints of uniform buckets).
    """
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    return -1.0 + (indices + 0.5) * delta


# ---------------------------------------------------------------------------
# PROVIDED: Data generation and display helpers
# ---------------------------------------------------------------------------

def generate_gaussian_data(n: int, d: int, sigma: float, seed: int = 42) -> np.ndarray:
    """Generate n random vectors from N(0, sigma^2 * I_d).

    This simulates the distribution of coordinates of unit sphere vectors in
    high dimensions.  For d=256, coordinates of S^{d-1} vectors follow a
    Beta distribution that converges to N(0, 1/256) ≈ N(0, 0.0625^2).

    Parameters
    ----------
    n : int
        Number of sample scalars to generate (simulating n*d coordinates).
    d : int
        Dimension (controls the shape of the output for display purposes).
    sigma : float
        Standard deviation.  For d=256 sphere coordinates: sigma = 1/√256
        ≈ 0.0625.  For general Gaussian data: sigma can be 1.0 or other values.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    np.ndarray, shape (n, d)
        Gaussian samples drawn from N(0, sigma^2).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=(n, d))


def compute_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Compute mean per-vector squared L2 error (MSE distortion).

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Original vectors.
    x_hat : np.ndarray, shape (n, d)
        Reconstructed vectors.

    Returns
    -------
    float
        Mean over n of ||x_i - x_hat_i||_2^2.
    """
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=1)))


def print_comparison_table(
    bit_widths: list,
    mse_uniform: list,
    mse_equiprobable: list,
    sigma: float,
) -> None:
    """Print a side-by-side MSE comparison table.

    Parameters
    ----------
    bit_widths : list of int
        Bit-widths tested.
    mse_uniform : list of float
        MSE values for the uniform quantizer.
    mse_equiprobable : list of float
        MSE values for the equiprobable quantizer.
    sigma : float
        Standard deviation of the data distribution.
    """
    print(f"\n  σ = {sigma:.4f}  (distribution: N(0, {sigma:.4f}^2))")
    print(f"  {'b':>4}  {'Uniform MSE':>14}  {'Equiprobable MSE':>18}  {'improvement':>12}")
    print("  " + "-" * 54)
    for b, mse_u, mse_e in zip(bit_widths, mse_uniform, mse_equiprobable):
        improvement = (mse_u - mse_e) / mse_u * 100
        marker = " ←" if improvement > 5 else ""
        print(f"  {b:>4}  {mse_u:>14.6f}  {mse_e:>18.6f}  "
              f"{improvement:>11.1f}%{marker}")


# ---------------------------------------------------------------------------
# YOUR CODE — implement the three functions below
# ---------------------------------------------------------------------------

def equiprobable_boundaries(b: int, sigma: float) -> np.ndarray:
    """Compute quantile-based bucket boundaries for N(0, sigma^2) data.

    Places 2^b - 1 interior boundaries such that each of the 2^b buckets
    captures equal probability mass under N(0, sigma^2).

    The k-th interior boundary (k = 1, ..., 2^b - 1) is the k/(2^b)-th
    quantile of N(0, sigma^2):
        t_k = sigma * Phi^{-1}(k / 2^b)
    where Phi^{-1} is the inverse CDF (percent point function) of N(0, 1).

    The first boundary is implicitly -inf (or -np.inf) and the last is +inf.
    Return only the 2^b - 1 INTERIOR boundaries.

    Parameters
    ----------
    b : int
        Bit-width; produces 2^b equal-mass buckets with 2^b - 1 interior
        boundaries.
    sigma : float
        Standard deviation of the Gaussian distribution N(0, sigma^2).

    Returns
    -------
    np.ndarray, shape (2^b - 1,)
        Interior boundary values in increasing order, in the same units as
        the data (NOT normalized by sigma).

    Notes
    -----
    Use scipy.stats.norm.ppf(p, loc=0, scale=sigma) to compute quantiles.

    For b=1 (2 buckets): one interior boundary at t_1 = 0 (median of N(0,σ²)).
    For b=2 (4 buckets): three interior boundaries at the 25th, 50th, 75th
    percentiles: t_1 ≈ -0.674σ, t_2 = 0, t_3 ≈ +0.674σ.

    The key advantage over uniform boundaries: for very small σ (like 0.0625
    for d=256 sphere vectors), these boundaries are clustered around 0, while
    the uniform quantizer wastes most of its range on [-1, -0.125] and [0.125, 1]
    which almost no data ever reaches.
    """
    ###########################################################
    # YOUR CODE HERE - 5-8 lines                              #
    #                                                         #
    # Steps:                                                  #
    # 1. Compute n_buckets = 2^b                              #
    # 2. Create an array of n_buckets - 1 quantile levels:   #
    #    p = [1/n_buckets, 2/n_buckets, ..., (n-1)/n_buckets]#
    # 3. Use stats.norm.ppf(p, loc=0, scale=sigma) to get    #
    #    the boundary positions                               #
    # 4. Return the resulting boundary array                  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def equiprobable_centroids(
    boundaries: np.ndarray, sigma: float
) -> np.ndarray:
    """Compute bucket centroids as conditional means within each bucket.

    Given equiprobable bucket boundaries, the optimal centroid for bucket k
    (spanning from a_k to b_k) is the conditional mean of X under N(0, σ²):

        E[X | a_k < X ≤ b_k] = σ * (φ(a_k/σ) - φ(b_k/σ)) / (Φ(b_k/σ) - Φ(a_k/σ))

    where:
        φ(z) = exp(-z²/2) / √(2π)  is the standard normal PDF
        Φ(z) = ∫_{-∞}^z φ(t)dt     is the standard normal CDF

    Boundary conventions for the outermost buckets:
        First bucket: a_0 = -∞  → φ(-∞) = 0  → centroid = -σ·φ(b_0/σ) / Φ(b_0/σ)
        Last bucket:  b_K = +∞  → φ(+∞) = 0  → centroid = σ·φ(a_K/σ) / (1 - Φ(a_K/σ))

    Parameters
    ----------
    boundaries : np.ndarray, shape (2^b - 1,)
        Interior bucket boundaries, as returned by equiprobable_boundaries.
        Does NOT include -inf or +inf at the edges.
    sigma : float
        Standard deviation of the Gaussian distribution N(0, sigma^2).

    Returns
    -------
    np.ndarray, shape (2^b,)
        Conditional means (centroids) for each of the 2^b buckets.

    Notes
    -----
    Use stats.norm.pdf and stats.norm.cdf (with loc=0, scale=sigma) to
    compute φ and Φ at boundary values.

    Common mistakes:
    - Using bucket midpoints instead of conditional means: midpoints ignore
      skewness within each bucket.  For uniform mass distribution this matters
      less, but for Gaussian mass distribution the conditional mean can be
      significantly different from the midpoint.
    - Dividing by σ² instead of σ in the formula (off-by-one in scaling).
    - Forgetting the outermost buckets (using only the interior ones).
    """
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Steps:                                                  #
    # 1. Construct the full boundary array with -inf and +inf:#
    #    full_boundaries = [-inf] + boundaries.tolist() + [+inf]          #
    # 2. For each consecutive pair (a_k, b_k) in full_boundaries:        #
    #    a. Compute pdf(a_k/sigma) and pdf(b_k/sigma)         #
    #       (use 0 for inf boundaries)                        #
    #    b. Compute cdf(b_k/sigma) - cdf(a_k/sigma) = P(a<X≤b)          #
    #    c. Centroid = sigma * (pdf_a - pdf_b) / P(a<X≤b)    #
    # 3. Return array of 2^b centroids                        #
    #                                                         #
    # Hint: np.concatenate([-inf...], boundaries, [+inf...]) #
    # or use np.r_[-np.inf, boundaries, np.inf]              #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def equiprobable_quantize(
    x: np.ndarray, boundaries: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Quantize values using precomputed equiprobable boundaries and centroids.

    Assigns each value in x to the bucket whose boundary interval contains it,
    then returns the corresponding centroid as the reconstructed value.

    Bucket assignment:
        x falls in bucket k if  boundaries[k-1] < x ≤ boundaries[k]
        (with boundaries[-1] = -∞ and boundaries[n_buckets] = +∞)

    Parameters
    ----------
    x : np.ndarray, shape (n, d) or (d,)
        Input values to quantize (any shape, values can be outside the
        boundary range — they are clipped to the outermost buckets).
    boundaries : np.ndarray, shape (2^b - 1,)
        Interior boundary values (NOT including -inf and +inf endpoints),
        in increasing order, as returned by equiprobable_boundaries.
    centroids : np.ndarray, shape (2^b,)
        Centroid values for each bucket, as returned by equiprobable_centroids.

    Returns
    -------
    np.ndarray, same shape as x
        Reconstructed values: each element replaced by its bucket's centroid.

    Notes
    -----
    Use np.searchsorted(boundaries, x) to find which bucket each element
    belongs to.  searchsorted with default side='left' returns:
        0 if x ≤ boundaries[0]
        k if boundaries[k-1] < x ≤ boundaries[k]
        len(boundaries) if x > boundaries[-1]
    This maps exactly to bucket indices in [0, 2^b - 1].  No clipping needed
    because the outermost buckets have centroids for all extreme values.
    """
    ###########################################################
    # YOUR CODE HERE - 3-5 lines                              #
    #                                                         #
    # Steps:                                                  #
    # 1. Use np.searchsorted(boundaries, x) to get indices   #
    # 2. Clip indices to [0, len(centroids)-1] for safety    #
    # 3. Return centroids[indices] (fancy indexing)           #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# MAIN BLOCK — fully provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Exercise 3: Uniform vs. Distribution-Aware Quantization")
    print("  Module 0 — The Vector Quantization Problem")
    print("=" * 60)

    d = 256
    n = 5_000
    bit_widths = [1, 2, 3, 4]

    # -----------------------------------------------------------------------
    # Case 1: sigma = 1/√d ≈ 0.0625 (sphere coordinate distribution)
    # This models the actual coordinates of unit vectors in R^256.
    # -----------------------------------------------------------------------
    sigma_sphere = 1.0 / np.sqrt(d)
    print(f"\n--- Case 1: Sphere coordinate distribution (sigma={sigma_sphere:.4f}) ---")
    print(f"    (Models coordinates of random unit vectors in R^{d})")

    X_sphere = generate_gaussian_data(n, d, sigma=sigma_sphere, seed=42)
    print(f"    Data std: {X_sphere.std():.4f} (theory: {sigma_sphere:.4f})")

    mse_uniform_sphere = []
    mse_equiprobable_sphere = []

    for b in bit_widths:
        # Uniform quantizer
        uni_idx = uniform_quantize(X_sphere, b)
        X_uni = uniform_dequantize(uni_idx, b)
        mse_u = compute_mse(X_sphere, X_uni)

        # Equiprobable quantizer
        bounds = equiprobable_boundaries(b, sigma_sphere)
        cents = equiprobable_centroids(bounds, sigma_sphere)
        X_eq = equiprobable_quantize(X_sphere, bounds, cents)
        mse_e = compute_mse(X_sphere, X_eq)

        mse_uniform_sphere.append(mse_u)
        mse_equiprobable_sphere.append(mse_e)

    print_comparison_table(bit_widths, mse_uniform_sphere, mse_equiprobable_sphere, sigma_sphere)

    # -----------------------------------------------------------------------
    # Case 2: sigma = 1.0 (standard Gaussian data)
    # This shows the improvement on "normal" scale data.
    # -----------------------------------------------------------------------
    sigma_std = 1.0
    print(f"\n--- Case 2: Standard Gaussian data (sigma={sigma_std:.4f}) ---")
    print(f"    (Standard normal distribution N(0, 1))")

    X_std = generate_gaussian_data(n, d, sigma=sigma_std, seed=42)

    mse_uniform_std = []
    mse_equiprobable_std = []

    for b in bit_widths:
        # Uniform quantizer (designed for [-1,1] but data lives in ~[-3,3])
        uni_idx = uniform_quantize(np.clip(X_std, -1, 1), b)
        X_uni = uniform_dequantize(uni_idx, b)
        mse_u = compute_mse(X_std, X_uni)

        # Equiprobable quantizer
        bounds = equiprobable_boundaries(b, sigma_std)
        cents = equiprobable_centroids(bounds, sigma_std)
        X_eq = equiprobable_quantize(X_std, bounds, cents)
        mse_e = compute_mse(X_std, X_eq)

        mse_uniform_std.append(mse_u)
        mse_equiprobable_std.append(mse_e)

    print_comparison_table(bit_widths, mse_uniform_std, mse_equiprobable_std, sigma_std)

    # -----------------------------------------------------------------------
    # Codebook inspection: boundaries and centroids at b=2 for sigma_sphere
    # -----------------------------------------------------------------------
    print(f"\n--- Codebook inspection: b=2, sigma={sigma_sphere:.4f} ---")
    bounds_inspect = equiprobable_boundaries(2, sigma_sphere)
    cents_inspect = equiprobable_centroids(bounds_inspect, sigma_sphere)
    uniform_cents = np.array([-0.75, -0.25, 0.25, 0.75])  # uniform b=2 centroids

    print(f"  Uniform boundaries: -1.0, -0.5, 0.0, +0.5, +1.0")
    print(f"  Equiprob boundaries: [-inf, {bounds_inspect[0]:.5f}, "
          f"{bounds_inspect[1]:.5f}, {bounds_inspect[2]:.5f}, +inf]")
    print(f"  Uniform centroids:     {uniform_cents}")
    print(f"  Equiprob centroids:    {cents_inspect}")
    print(f"  → Equiprobable centroids are clustered near 0 (where data is),")
    print(f"    while uniform centroids are spread evenly across [-1, 1].")

    # -----------------------------------------------------------------------
    # improvement summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  improvement Summary")
    print("=" * 60)

    for b, mse_u, mse_e in zip(bit_widths, mse_uniform_sphere, mse_equiprobable_sphere):
        improvement = (mse_u - mse_e) / mse_u * 100
        ratio = mse_u / mse_e
        print(f"  b={b}: Uniform={mse_u:.4f}, Equiprobable={mse_e:.6f}, "
              f"improvement={improvement:.1f}% ({ratio:.0f}× better)")

    print()
    print("  Key insight: equiprobable quantization places ALL resolution near 0")
    print("  (where sphere coordinates actually live) rather than spreading")
    print("  it evenly across [-1, 1] where almost no data exists.")
    print()
    print("  This improvement gap will be measured again in Module 1 when we")
    print("  add the Lloyd-Max centroid optimization (equiPROBABLE → equalMSE).")
    print()
    print("  The full TurboQuant pipeline adds random rotation to GUARANTEE")
    print("  the Gaussian/Beta distribution BEFORE applying optimal codebooks,")
    print("  making the improvement hold for ANY input vector, not just Gaussian.")

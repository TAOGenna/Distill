"""
Exercise 3: Uniform vs. Distribution-Aware Quantization — SOLUTION
====================================================================

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

Dependencies
------------
    pip install numpy scipy

Usage
-----
    python _solutions/ex03_uniform_vs_distributionaware_quantization.py
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

    Parameters
    ----------
    n : int
        Number of sample vectors.
    d : int
        Dimension of each vector.
    sigma : float
        Standard deviation.
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
    """Print a side-by-side MSE comparison table."""
    print(f"\n  σ = {sigma:.4f}  (distribution: N(0, {sigma:.4f}^2))")
    print(f"  {'b':>4}  {'Uniform MSE':>14}  {'Equiprobable MSE':>18}  {'improvement':>12}")
    print("  " + "-" * 54)
    for b, mse_u, mse_e in zip(bit_widths, mse_uniform, mse_equiprobable):
        improvement = (mse_u - mse_e) / mse_u * 100
        marker = " ←" if improvement > 5 else ""
        print(f"  {b:>4}  {mse_u:>14.6f}  {mse_e:>18.6f}  "
              f"{improvement:>11.1f}%{marker}")


# ---------------------------------------------------------------------------
# SOLUTION IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def equiprobable_boundaries(b: int, sigma: float) -> np.ndarray:
    """Compute quantile-based bucket boundaries for N(0, sigma^2) data.

    Places 2^b - 1 interior boundaries at equal probability mass quantiles.
    The k-th interior boundary is:  t_k = sigma * Phi^{-1}(k / 2^b)

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
        Interior boundary values in increasing order.
    """
    n_buckets = 2 ** b
    # Quantile levels: 1/n, 2/n, ..., (n-1)/n — these are the interior cuts
    p = np.arange(1, n_buckets) / n_buckets
    # Inverse CDF of N(0, sigma^2) at each level
    boundaries = stats.norm.ppf(p, loc=0.0, scale=sigma)
    return boundaries


def equiprobable_centroids(
    boundaries: np.ndarray, sigma: float
) -> np.ndarray:
    """Compute bucket centroids as conditional means within each bucket.

    E[X | a_k < X ≤ b_k] = σ * (φ(a_k/σ) - φ(b_k/σ)) / (Φ(b_k/σ) - Φ(a_k/σ))

    where φ is the standard normal PDF and Φ is the standard normal CDF.

    Parameters
    ----------
    boundaries : np.ndarray, shape (2^b - 1,)
        Interior bucket boundaries (excluding -inf and +inf).
    sigma : float
        Standard deviation of the Gaussian distribution N(0, sigma^2).

    Returns
    -------
    np.ndarray, shape (2^b,)
        Conditional means (centroids) for each of the 2^b buckets.
    """
    # Extend boundaries with -inf and +inf for the outermost edges
    full_boundaries = np.r_[-np.inf, boundaries, np.inf]
    n_buckets = len(full_boundaries) - 1

    centroids = np.zeros(n_buckets)
    for k in range(n_buckets):
        a = full_boundaries[k]      # lower boundary
        b = full_boundaries[k + 1]  # upper boundary

        # φ(a/σ) and φ(b/σ): standard normal PDF at normalized boundaries
        # (0 at ±inf by convention, which scipy handles correctly)
        pdf_a = stats.norm.pdf(a, loc=0.0, scale=sigma)  # = φ(a/σ) / σ × σ
        pdf_b = stats.norm.pdf(b, loc=0.0, scale=sigma)

        # Φ(b/σ) - Φ(a/σ): probability mass in this bucket
        prob = stats.norm.cdf(b, loc=0.0, scale=sigma) - \
               stats.norm.cdf(a, loc=0.0, scale=sigma)

        # Conditional mean formula:
        # E[X | a < X ≤ b] = σ² × (f(a) - f(b)) / P(a < X ≤ b)
        # where f(x) = φ(x/σ)/σ is the N(0,σ²) PDF
        # = σ² × (φ(a/σ)/σ - φ(b/σ)/σ) / prob
        # = σ × (φ(a/σ) - φ(b/σ)) / prob... but scipy returns N(0,σ²) PDF
        # stats.norm.pdf(x, 0, sigma) = φ(x/σ)/σ
        # So: E[X|a<X≤b] = sigma^2 * (pdf_a - pdf_b) / prob
        centroids[k] = sigma ** 2 * (pdf_a - pdf_b) / prob

    return centroids


def equiprobable_quantize(
    x: np.ndarray, boundaries: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Quantize values using precomputed equiprobable boundaries and centroids.

    Parameters
    ----------
    x : np.ndarray, shape (n, d) or (d,)
        Input values to quantize.
    boundaries : np.ndarray, shape (2^b - 1,)
        Interior boundary values (excluding -inf and +inf endpoints).
    centroids : np.ndarray, shape (2^b,)
        Centroid values for each bucket.

    Returns
    -------
    np.ndarray, same shape as x
        Reconstructed values: each element replaced by its bucket's centroid.
    """
    # searchsorted returns the bucket index for each element:
    # 0 if x <= boundaries[0], k if boundaries[k-1] < x <= boundaries[k],
    # len(boundaries) if x > boundaries[-1]
    indices = np.searchsorted(boundaries, x, side='left')
    # Clip to valid range in case of numerical edge cases
    indices = np.clip(indices, 0, len(centroids) - 1)
    return centroids[indices]


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
    print("  Note: lower MSE does NOT eliminate inner product bias from Exercise 2.")
    print("  bias in similarity estimation persists regardless of codebook placement;")
    print("  only TurboQuant's QJL component (Module 2) achieves unbiased IP estimation.")
    print()
    print("  The full TurboQuant pipeline adds random rotation to GUARANTEE")
    print("  the Gaussian/Beta distribution BEFORE applying optimal codebooks,")
    print("  making the improvement hold for ANY input vector, not just Gaussian.")

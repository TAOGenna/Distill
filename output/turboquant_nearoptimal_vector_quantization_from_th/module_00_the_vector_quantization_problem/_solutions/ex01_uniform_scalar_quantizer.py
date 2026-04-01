"""
Exercise 1: Uniform Scalar Quantizer — SOLUTION
=================================================

Course: TurboQuant — Near-Optimal Vector Quantization from Theory to Practice
Module: 0 — The Vector Quantization Problem

Goal
----
Implement the most basic vector quantizer: divide the range [-1, 1] into 2^b
equal-width buckets, map each coordinate to the nearest bucket, and reconstruct
from bucket centroids.  Then measure the MSE distortion at b = 1, 2, 3, 4 bits
for random unit vectors in d = 256 dimensions.

These numbers are your baselines for the entire course. Every subsequent module
will beat them. By the time you reach Module 2 (full TurboQuant_mse pipeline),
you will be able to see exactly how much the random-rotation + Lloyd-Max codebook
improves on what you implement here.

Key concept from the paper (verbatim):
  "our goal is to design a quantization map Q: R^d → {0,1}^B that transforms
   d-dimensional vectors to a binary string of B bits. If we set B = b*d for
   some b ≥ 0, this quantizer will have a bit-width of b. Crucially, we require
   an inverse map Q^{-1}: {0,1}^B → R^d that performs dequantization."

Dependencies
------------
    pip install numpy

Usage
-----
    python _solutions/ex01_uniform_scalar_quantizer.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# PROVIDED HELPERS — do not modify these
# ---------------------------------------------------------------------------

def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """Project each row of X onto the unit sphere S^{d-1}.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input vectors.  May have arbitrary non-zero norm.

    Returns
    -------
    np.ndarray, shape (n, d)
        Row-normalized version of X: each row has L2 norm exactly 1.0.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def generate_test_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate n random unit vectors in R^d.

    Vectors are drawn from an isotropic Gaussian and then normalized, which
    produces a uniform distribution on the hypersphere S^{d-1}. This is the
    standard way to sample uniformly from a high-dimensional sphere.

    Parameters
    ----------
    n : int
        Number of vectors to generate.
    d : int
        Dimension of each vector.
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    np.ndarray, shape (n, d)
        Matrix of unit vectors; each row has L2 norm 1.0.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return normalize_vectors(X)


def print_results_table(bit_widths: list, mse_values: list) -> None:
    """Print a formatted table of bit-width vs MSE distortion.

    Parameters
    ----------
    bit_widths : list of int
        Bit-widths tested (e.g., [1, 2, 3, 4]).
    mse_values : list of float
        Corresponding average MSE distortion for each bit-width.
    """
    print("\n" + "=" * 55)
    print(f"  {'Bits (b)':>10}  {'MSE Distortion':>16}  {'bits/coord':>10}")
    print("=" * 55)
    for b, mse in zip(bit_widths, mse_values):
        print(f"  {b:>10d}  {mse:>16.6f}  {b:>10d}")
    print("=" * 55)
    print("\n  Reference (TurboQuant upper bounds, from paper):")
    paper_bounds = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    for b, ub in paper_bounds.items():
        print(f"    b={b}: TurboQuant_mse ≤ {ub}")
    print()


# ---------------------------------------------------------------------------
# SOLUTION IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def uniform_quantize(x: np.ndarray, b: int) -> np.ndarray:
    """Quantize values in x to integer bucket indices using a uniform grid.

    Partitions the interval [-1, 1] into 2^b equal-width buckets of width
    Δ = 2 / 2^b.  Each value in x is mapped to the index of the bucket it
    falls into.  Values at the boundary x = 1.0 exactly are assigned to the
    last bucket (index 2^b - 1).

    Bucket k spans the interval:
        [-1 + k*Δ,  -1 + (k+1)*Δ)   for k = 0, 1, ..., 2^b - 2
        [-1 + (2^b-1)*Δ,  1]         for k = 2^b - 1  (closed on right)

    Parameters
    ----------
    x : np.ndarray, shape (n, d) or (d,)
        Input values.  Should be in [-1, 1]; values outside this range are
        clipped to the nearest bucket.
    b : int
        Bit-width.  The number of distinct bucket indices is 2^b.

    Returns
    -------
    np.ndarray of int, same shape as x
        Bucket indices in the range [0, 2^b - 1].
    """
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    # Shift from [-1, 1] to [0, 2], divide by bucket width, take floor
    indices = np.floor((x + 1.0) / delta).astype(int)
    # Clip to [0, n_buckets - 1] to handle boundary x = 1.0 exactly
    return np.clip(indices, 0, n_buckets - 1)


def uniform_dequantize(indices: np.ndarray, b: int) -> np.ndarray:
    """Reconstruct values from bucket indices by returning bucket centroids.

    Given a bucket index k (from uniform_quantize), returns the centroid of
    bucket k: the midpoint of the interval [-1 + k*Δ, -1 + (k+1)*Δ].

    The centroid formula is:
        centroid(k) = -1 + (k + 0.5) * Δ    where Δ = 2 / 2^b

    Parameters
    ----------
    indices : np.ndarray of int, shape (n, d) or (d,)
        Bucket indices in the range [0, 2^b - 1], as returned by
        uniform_quantize.
    b : int
        Bit-width, same value used when calling uniform_quantize.

    Returns
    -------
    np.ndarray of float, same shape as indices
        Reconstructed values (bucket centroids) in the range (-1, 1).
    """
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    # Centroid of bucket k: left edge + half bucket width
    return -1.0 + (indices + 0.5) * delta


def measure_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Compute the mean squared error between original and reconstructed vectors.

    Implements the distortion metric from the paper:
        D_mse = E_Q[ ||x - Q^{-1}(Q(x))||_2^2 ]

    Here we take the expectation over vectors (rows) rather than over the
    quantizer's randomness (the uniform quantizer is deterministic, so there is
    no quantizer randomness to average over).

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Original unit vectors (before quantization).
    x_hat : np.ndarray, shape (n, d)
        Reconstructed vectors (after quantize → dequantize).

    Returns
    -------
    float
        Average per-vector MSE: mean over n vectors of ||x_i - x_hat_i||^2.
    """
    # Squared L2 error per vector (sum over d coords), then average over n
    return np.mean(np.sum((x - x_hat) ** 2, axis=1))


# ---------------------------------------------------------------------------
# MAIN BLOCK — fully provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Exercise 1: Uniform Scalar Quantizer")
    print("  Module 0 — The Vector Quantization Problem")
    print("=" * 55)

    # Configuration
    d = 256          # vector dimension (like a small embedding)
    n = 5_000        # number of test vectors
    bit_widths = [1, 2, 3, 4]

    # Generate random unit vectors on S^{d-1}
    print(f"\nGenerating {n} random unit vectors in R^{d}...")
    X = generate_test_vectors(n=n, d=d, seed=42)

    # Verify unit norm
    norms = np.linalg.norm(X, axis=1)
    print(f"Norm statistics — mean: {norms.mean():.6f}, std: {norms.std():.2e}")
    assert np.allclose(norms, 1.0, atol=1e-10), "Vectors must be unit norm!"

    # Quick coordinate distribution check
    first_coord = X[:, 0]
    sigma_theory = 1.0 / np.sqrt(d)
    print(f"\nCoordinate x_0 statistics (d={d}):")
    print(f"  Empirical std: {first_coord.std():.4f}")
    print(f"  Theoretical std (1/√d): {sigma_theory:.4f}")
    print(f"  Max absolute value: {np.abs(first_coord).max():.4f}")
    print(f"  Fraction in [-0.125, 0.125]: "
          f"{np.mean(np.abs(first_coord) < 0.125):.3f}")

    # -----------------------------------------------------------------------
    # Quantize at each bit-width and measure MSE
    # -----------------------------------------------------------------------
    print("\nRunning uniform quantization at each bit-width...")
    mse_values = []

    for b in bit_widths:
        n_buckets = 2 ** b
        delta = 2.0 / n_buckets

        # Step 1: Quantize — map each coordinate to a bucket index
        indices = uniform_quantize(X, b)

        # Sanity checks on indices
        assert indices.shape == X.shape, \
            f"Expected shape {X.shape}, got {indices.shape}"
        assert indices.dtype in [np.int32, np.int64, int], \
            f"Indices must be integer type, got {indices.dtype}"
        assert indices.min() >= 0 and indices.max() <= n_buckets - 1, \
            f"Indices out of range [0, {n_buckets-1}]: [{indices.min()}, {indices.max()}]"

        # Step 2: Dequantize — map indices back to centroid values
        X_hat = uniform_dequantize(indices, b)

        assert X_hat.shape == X.shape, \
            f"Dequantized shape {X_hat.shape} != input shape {X.shape}"
        assert X_hat.dtype in [np.float32, np.float64], \
            f"Dequantized values must be float, got {X_hat.dtype}"

        # Step 3: Measure MSE distortion
        mse = measure_mse(X, X_hat)
        mse_values.append(mse)

        print(f"  b={b}: n_buckets={n_buckets:3d}, Δ={delta:.4f}, MSE={mse:.6f}")

    # Print formatted results table
    print_results_table(bit_widths, mse_values)

    # -----------------------------------------------------------------------
    # Sanity checks with expected approximate values
    # -----------------------------------------------------------------------
    print("Sanity checks (MSE distortion):")
    # The uniform quantizer places centroids at ±0.5 (b=1), but sphere
    # coordinates in d=256 have std ≈ 1/√256 ≈ 0.0625 and are concentrated near 0.
    # Almost every coordinate maps to centroid ±0.5, but its true value is near 0,
    # giving per-coord MSE ≈ 0.5² ≈ 0.25. Total MSE ≈ 256 × 0.25 ≈ 64 at b=1.
    # The exact expected values (for d=256 sphere vectors):
    expected_rough = {1: (40.0, 70.0), 2: (8.0, 14.0), 3: (1.0, 3.0), 4: (0.20, 0.55)}
    all_ok = True
    for b, mse in zip(bit_widths, mse_values):
        lo, hi = expected_rough[b]
        ok = lo <= mse <= hi
        status = "OK" if ok else "WARNING"
        print(f"  b={b}: MSE={mse:.3f} in [{lo}, {hi}]? {status}")
        if not ok:
            all_ok = False
    if all_ok:
        print("\n  All MSE values in expected range. Implementation looks correct!")
    else:
        print("\n  Some values outside expected range — check your implementation.")

    # -----------------------------------------------------------------------
    # Key observation printout
    # -----------------------------------------------------------------------
    print("\nKey observation (MSE distortion comparison):")
    print(f"  At b=1, uniform quantizer achieves MSE ≈ {mse_values[0]:.1f}.")
    print(f"  TurboQuant_mse achieves D_mse ≤ 0.36 at b=1 (from the paper).")
    ratio = mse_values[0] / 0.36
    print(f"  Gap: {ratio:.0f}× worse than TurboQuant_mse!")
    print(f"  improvement potential: {ratio:.0f}× MSE reduction achievable with TurboQuant_mse at b=1.")
    print()
    print("  WHY? The uniform quantizer places centroids at ±0.5, but sphere")
    print(f"  coordinates in d={d} have std ≈ {1/d**0.5:.4f} — concentrated near 0.")
    print("  Most data lands far from the ±0.5 centroids, wasting all resolution.")
    print()
    print("  At b=4: 14 of 16 buckets span [-1, -0.125] and [0.125, 1], yet")
    print(f"  only {1 - np.mean(np.abs(X[:, 0]) < 0.125):.3f} of data lives there.")
    print()
    print("  The Lloyd-Max codebook (Module 1) fixes this by concentrating")
    print("  buckets where the Beta distribution actually has mass.")
    print()
    print("  Next: Exercise 2 — measure inner product distortion and bias.")

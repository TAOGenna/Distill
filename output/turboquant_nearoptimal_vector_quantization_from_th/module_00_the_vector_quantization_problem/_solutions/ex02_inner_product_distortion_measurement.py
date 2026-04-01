"""
Exercise 2: Inner Product Distortion Measurement — SOLUTION
============================================================

Course: TurboQuant — Near-Optimal Vector Quantization from Theory to Practice
Module: 0 — The Vector Quantization Problem

Goal
----
In Exercise 1 you measured MSE distortion of the uniform quantizer (b=1: ~52,
b=4: ~0.34 for d=256 unit vectors).  Now measure a DIFFERENT kind of error:
inner product distortion.  When we compress a database of vectors and then
compute dot products with query vectors, how wrong are the similarity scores?

Two failure modes appear:
  1. VARIANCE: Inner product estimates fluctuate around the true value.
  2. BIAS: Inner product estimates are SYSTEMATICALLY too large or too small.

Bias is the more dangerous failure mode.  A biased estimator is wrong even
with infinite averaging, meaning no amount of additional computation can fix it.
This motivates TurboQuant's explicit unbiasedness requirement:
    E_Q[ <y, Q^{-1}(Q(x))> ] = <y, x>  for all y, x.

Key formula from the paper (verbatim):
    D_prod := E_Q[ |<y, x> - <y, Q^{-1}(Q(x))>|^2 ]
    "Furthermore, for inner-product quantizers, we require unbiasedness of the
     inner product estimator."

Dependencies
------------
    pip install numpy

Usage
-----
    python _solutions/ex02_inner_product_distortion_measurement.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# PROVIDED: Working uniform quantizer from Exercise 1
# ---------------------------------------------------------------------------

def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """Project each row of X onto the unit sphere S^{d-1}.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input vectors with arbitrary non-zero norms.

    Returns
    -------
    np.ndarray, shape (n, d)
        Unit-norm version of X (each row has ||x||_2 = 1).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def uniform_quantize(x: np.ndarray, b: int) -> np.ndarray:
    """Partition [-1,1] into 2^b equal buckets; return bucket indices.

    Parameters
    ----------
    x : np.ndarray, shape (n, d) or (d,)
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
    indices : np.ndarray of int, shape (n, d) or (d,)
        Bucket indices in [0, 2^b - 1].
    b : int
        Bit-width used during quantization.

    Returns
    -------
    np.ndarray of float, same shape as indices
        Bucket centroid values in (-1, 1).
    """
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    return -1.0 + (indices + 0.5) * delta


# ---------------------------------------------------------------------------
# PROVIDED: Data generation and display helpers
# ---------------------------------------------------------------------------

def generate_database_and_queries(
    n_db: int, n_query: int, d: int, seed: int = 42
) -> tuple:
    """Generate a database of unit vectors and a set of query vectors.

    Parameters
    ----------
    n_db : int
        Number of database vectors to generate.
    n_query : int
        Number of query vectors to generate.
    d : int
        Dimension of all vectors.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    database : np.ndarray, shape (n_db, d)
        Unit-norm database vectors drawn uniformly from S^{d-1}.
    queries : np.ndarray, shape (n_query, d)
        Unit-norm query vectors drawn uniformly from S^{d-1}.
    """
    rng = np.random.default_rng(seed)
    raw_db = rng.standard_normal((n_db, d))
    database = raw_db / np.linalg.norm(raw_db, axis=1, keepdims=True)
    raw_q = rng.standard_normal((n_query, d))
    queries = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)
    return database, queries


def compute_true_inner_products(
    queries: np.ndarray, database: np.ndarray
) -> np.ndarray:
    """Compute exact inner products between all queries and database vectors.

    Parameters
    ----------
    queries : np.ndarray, shape (n_query, d)
        Query vectors.
    database : np.ndarray, shape (n_db, d)
        Database vectors (original, unquantized).

    Returns
    -------
    np.ndarray, shape (n_query, n_db)
        True inner products: result[i, j] = <queries[i], database[j]>.
    """
    return queries @ database.T


def print_distortion_summary(
    b: int,
    true_ip: np.ndarray,
    estimated_ip: np.ndarray,
    distortion: float,
    bias: float,
) -> None:
    """Print a formatted summary of inner product distortion for one bit-width."""
    errors = (estimated_ip - true_ip).ravel()
    print(f"  b={b}:")
    print(f"    D_prod (MSE of IP estimates): {distortion:.6f}")
    print(f"    bias  = mean(estimated - true): {bias:+.6f}")
    print(f"    std   = std(estimated - true):  {errors.std():.6f}")
    print(f"    max |error|:                    {np.abs(errors).max():.4f}")


# ---------------------------------------------------------------------------
# SOLUTION IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def compute_estimated_inner_products(
    queries: np.ndarray, database: np.ndarray, b: int
) -> np.ndarray:
    """Estimate inner products using a uniformly quantized database.

    Parameters
    ----------
    queries : np.ndarray, shape (n_query, d)
        Original (non-quantized) query vectors.
    database : np.ndarray, shape (n_db, d)
        Original database vectors (unit norm on S^{d-1}).
    b : int
        Bit-width for uniform scalar quantization of the database.

    Returns
    -------
    np.ndarray, shape (n_query, n_db)
        Estimated inner products using the quantized database.
    """
    # Quantize the database (stored at compressed bit-width)
    db_indices = uniform_quantize(database, b)
    # Dequantize: approximate reconstruction from bucket centroids
    db_hat = uniform_dequantize(db_indices, b)
    # Compute inner products: queries are unquantized (online, never stored)
    return queries @ db_hat.T


def compute_inner_product_distortion(
    true_ip: np.ndarray, estimated_ip: np.ndarray
) -> float:
    """Compute mean squared inner product error (D_prod) averaged over all pairs.

    Parameters
    ----------
    true_ip : np.ndarray, shape (n_query, n_db)
        Exact inner products <queries[i], database[j]>.
    estimated_ip : np.ndarray, shape (n_query, n_db)
        Estimated inner products from the quantized database.

    Returns
    -------
    float
        Mean squared error of inner product estimates, averaged over all pairs.
    """
    errors = estimated_ip - true_ip
    return np.mean(errors ** 2)


def check_bias(true_ip: np.ndarray, estimated_ip: np.ndarray) -> float:
    """Compute the systematic bias of the inner product estimator.

    Parameters
    ----------
    true_ip : np.ndarray, shape (n_query, n_db)
        Exact inner products.
    estimated_ip : np.ndarray, shape (n_query, n_db)
        Estimated inner products from quantized database.

    Returns
    -------
    float
        Mean signed error: mean(estimated - true).  Zero = unbiased.
    """
    return np.mean(estimated_ip - true_ip)


def check_magnitude_bias(true_ip: np.ndarray, estimated_ip: np.ndarray) -> float:
    """Check if the quantizer inflates the MAGNITUDE of inner product estimates.

    Parameters
    ----------
    true_ip : np.ndarray, shape (n_query, n_db)
        Exact inner products.
    estimated_ip : np.ndarray, shape (n_query, n_db)
        Estimated inner products from quantized database.

    Returns
    -------
    float
        Scale factor: mean(|estimated_ip|) / mean(|true_ip|).
        Value > 1 means systematic magnitude inflation.
    """
    mean_abs_estimated = np.mean(np.abs(estimated_ip))
    mean_abs_true = np.mean(np.abs(true_ip))
    return mean_abs_estimated / mean_abs_true


# ---------------------------------------------------------------------------
# MAIN BLOCK — fully provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Exercise 2: Inner Product Distortion Measurement")
    print("  Module 0 — The Vector Quantization Problem")
    print("=" * 60)

    # Configuration
    d = 256
    n_db = 2_000
    n_query = 200
    bit_widths = [1, 2, 3, 4]

    # Generate data
    print(f"\nSetup: d={d}, {n_db} database vectors, {n_query} queries")
    database, queries = generate_database_and_queries(n_db, n_query, d, seed=42)

    # Verify unit norms
    db_norms = np.linalg.norm(database, axis=1)
    q_norms = np.linalg.norm(queries, axis=1)
    print(f"Database norm: mean={db_norms.mean():.6f}, std={db_norms.std():.2e}")
    print(f"Query norm:    mean={q_norms.mean():.6f}, std={q_norms.std():.2e}")

    # True (exact) inner products — reference
    true_ip = compute_true_inner_products(queries, database)
    print(f"\nTrue inner products: shape={true_ip.shape}")
    print(f"  min={true_ip.min():.4f}, max={true_ip.max():.4f}")
    print(f"  mean|IP|={np.abs(true_ip).mean():.4f}")

    # -----------------------------------------------------------------------
    # Measure distortion and bias at each bit-width
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Inner Product Distortion Summary")
    print("=" * 60)

    distortions = []
    biases = []
    mag_biases = []

    for b in bit_widths:
        estimated_ip = compute_estimated_inner_products(queries, database, b)

        assert estimated_ip.shape == true_ip.shape, \
            f"Shape mismatch: {estimated_ip.shape} != {true_ip.shape}"

        distortion = compute_inner_product_distortion(true_ip, estimated_ip)
        bias = check_bias(true_ip, estimated_ip)
        mag_bias = check_magnitude_bias(true_ip, estimated_ip)

        distortions.append(distortion)
        biases.append(bias)
        mag_biases.append(mag_bias)

        print_distortion_summary(b, true_ip, estimated_ip, distortion, bias)
        print(f"    magnitude scale factor:             {mag_bias:.3f}×")
        print()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"  {'b':>4}  {'D_prod':>12}  {'bias':>10}  {'mag_scale':>10}")
    print("=" * 60)
    for b, d_prod, bias, ms in zip(bit_widths, distortions, biases, mag_biases):
        print(f"  {b:>4}  {d_prod:>12.6f}  {bias:>+10.6f}  {ms:>10.3f}×")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Key observations
    # -----------------------------------------------------------------------
    print("\nKey observations:")
    print(f"  1. Magnitude bias at b=1: {mag_biases[0]:.2f}× inflation.")
    print(f"     The quantized vectors have L2 norm ≈ {mag_biases[0]:.1f} instead of 1!")
    print(f"     (Centroids at ±0.5 vs true std≈{1/d**0.5:.4f}: ratio≈{0.5/(1/d**0.5):.1f}×)")
    print()
    print(f"  2. Even at b=4, magnitude bias = {mag_biases[3]:.3f}×.")
    print(f"     The uniform quantizer NEVER fully removes the inflation.")
    print()
    print(f"  3. D_prod at b=4: {distortions[3]:.4f}")
    print(f"     TurboQuant_prod bound at b=4: 0.047/d = {0.047/d:.6f}")
    ratio_4 = distortions[3] / (0.047 / d)
    print(f"     Uniform is {ratio_4:.0f}× worse than TurboQuant_prod!")
    print(f"     improvement factor: {ratio_4:.0f}× better D_prod achievable with TurboQuant_prod.")
    print()
    print("  4. The bias in the uniform quantizer is NOT just variance —")
    print("     it is a SYSTEMATIC inflation of similarity scores.")
    print("     This corrupts nearest-neighbor rankings.")
    print()
    print("  This bias motivates TurboQuant's 2-stage design:")
    print("  → (b-1)-bit MSE quantizer + 1-bit QJL on residual")
    print("  → Achieves unbiased IP estimation with near-optimal variance.")
    print()
    print("  Next: Exercise 3 — compare uniform vs distribution-aware quantization.")

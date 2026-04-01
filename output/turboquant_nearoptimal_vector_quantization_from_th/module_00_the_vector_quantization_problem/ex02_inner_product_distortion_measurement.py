"""
Exercise 2: Inner Product Distortion Measurement
==================================================

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

Background
----------
The paper proves that MSE-optimal quantizers ARE biased for inner products.
Specifically, at 1-bit width, TurboQuant_mse produces a sign quantizer with
multiplicative bias 2/π ≈ 0.637 (the arc-cosine kernel bias).  The uniform
quantizer has even larger bias because its centroids are far from where the
data actually lives (as you saw in Exercise 1).

In Exercise 4 (Module 1), you will build a 2-stage quantizer (TurboQuant_prod)
that combines an MSE quantizer with a 1-bit QJL residual to achieve UNBIASED
inner product estimation.  The bias numbers you measure here are the baseline.

Dependencies
------------
    pip install numpy

Usage
-----
    python ex02_inner_product_distortion_measurement.py
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

    Query vectors are NOT constrained to unit norm — this tests D_prod's
    dependence on ||y||^2 (the paper shows D_prod ≤ (√3π²·||y||²/d) · 1/4^b).

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
        Query vectors drawn from N(0, I_d) — NOT normalized, so they
        have varying norms to test the ||y||^2 scaling in D_prod.
    """
    rng = np.random.default_rng(seed)
    # Database: uniform on unit sphere
    raw_db = rng.standard_normal((n_db, d))
    database = raw_db / np.linalg.norm(raw_db, axis=1, keepdims=True)
    # Queries: unit sphere too (for clean comparison)
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
    """Print a formatted summary of inner product distortion for one bit-width.

    Parameters
    ----------
    b : int
        Bit-width.
    true_ip : np.ndarray, shape (n_query, n_db)
        True inner products.
    estimated_ip : np.ndarray, shape (n_query, n_db)
        Estimated inner products from quantized database.
    distortion : float
        Mean squared inner product error (D_prod).
    bias : float
        Mean signed error (positive = overestimate, negative = underestimate).
    """
    errors = (estimated_ip - true_ip).ravel()
    print(f"  b={b}:")
    print(f"    D_prod (MSE of IP estimates): {distortion:.6f}")
    print(f"    bias  = mean(estimated - true): {bias:+.6f}")
    print(f"    std   = std(estimated - true):  {errors.std():.6f}")
    print(f"    max |error|:                    {np.abs(errors).max():.4f}")


# ---------------------------------------------------------------------------
# YOUR CODE — implement the three functions below
# ---------------------------------------------------------------------------

def compute_estimated_inner_products(
    queries: np.ndarray, database: np.ndarray, b: int
) -> np.ndarray:
    """Estimate inner products using a uniformly quantized database.

    Quantize each database vector at b bits, reconstruct (dequantize),
    then compute dot products with the query vectors.

    Note: in a real KV cache system, the database is quantized ONCE at
    storage time and the query is applied later without re-quantizing.
    We simulate this by quantizing the database and then computing:
        <query_i, dequantize(quantize(database_j))>

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
        Estimated inner products: result[i, j] ≈ <queries[i], database[j]>.

    Notes
    -----
    Common mistake: quantizing the queries instead of (or in addition to)
    the database.  In retrieval, queries arrive online and are NOT quantized;
    only the stored database is compressed.
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                              #
    #                                                         #
    # Steps:                                                  #
    # 1. Quantize the database: uniform_quantize(database, b) #
    # 2. Dequantize: uniform_dequantize(indices, b)           #
    # 3. Compute queries @ database_hat.T                     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_inner_product_distortion(
    true_ip: np.ndarray, estimated_ip: np.ndarray
) -> float:
    """Compute mean squared inner product error (D_prod) averaged over all pairs.

    Implements the paper's distortion metric:
        D_prod = E_Q[ |<y, x> - <y, Q^{-1}(Q(x))>|^2 ]

    Here the expectation is approximated by averaging over all query-database
    pairs.  Since the uniform quantizer is deterministic, there is no
    quantizer randomness to average over.

    Parameters
    ----------
    true_ip : np.ndarray, shape (n_query, n_db)
        Exact inner products <queries[i], database[j]>.
    estimated_ip : np.ndarray, shape (n_query, n_db)
        Estimated inner products from the quantized database.

    Returns
    -------
    float
        Mean squared error of inner product estimates, averaged over all
        (query, database) pairs.  Scalar value.

    Notes
    -----
    Do NOT average only over database vectors (axis=1); also average over
    queries (axis=0), or equivalently, call .mean() on the full 2D error
    array.  The distortion should decrease as bit-width increases.

    Reference values for b=1,2,3,4 with d=256 unit sphere vectors (uniform
    quantizer, not TurboQuant): expect much larger values than TurboQuant's
    bounds because the uniform quantizer is far from optimal for sphere data.
    """
    ###########################################################
    # YOUR CODE HERE - 3-5 lines                              #
    #                                                         #
    # Steps:                                                  #
    # 1. Compute the error matrix: estimated_ip - true_ip     #
    # 2. Square element-wise                                  #
    # 3. Take the mean over all elements (both axes)          #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def check_bias(true_ip: np.ndarray, estimated_ip: np.ndarray) -> float:
    """Compute the systematic bias of the inner product estimator.

    Bias is the mean signed error: mean(estimated - true).
    An unbiased estimator has bias = 0 for all query-database pairs.
    A positive bias means we consistently overestimate inner products.
    A negative bias means we consistently underestimate.

    The paper requires:
        E_Q[ <y, Q^{-1}(Q(x))> ] = <y, x>   (unbiasedness)

    The uniform quantizer violates this because its centroids are at ±0.5
    (for b=1), while the actual sphere coordinates are near 0.  When we
    compute <y, x_hat>, we are computing inner products with inflated
    coordinate values, systematically overestimating |<y, x>|.

    Parameters
    ----------
    true_ip : np.ndarray, shape (n_query, n_db)
        Exact inner products.
    estimated_ip : np.ndarray, shape (n_query, n_db)
        Estimated inner products from the quantized database.

    Returns
    -------
    float
        Mean signed error: mean over all (query, database) pairs of
        (estimated_ip - true_ip).  Zero means unbiased, positive means
        systematic overestimate, negative means systematic underestimate.

    Notes
    -----
    The bias depends on the ABSOLUTE values of inner products, not their
    signs.  Since E[sign(x_j)] = 0 for symmetric distributions, and the
    uniform quantizer centroids are symmetric around 0, the bias in the
    raw inner product estimates may appear small on average over all pairs.

    To see the systematic nature of the bias, also compute the bias in
    |<y, x>| (absolute values) — the uniform quantizer ALWAYS inflates
    magnitudes, even if the sign of the bias in signed inner products averages
    to near zero due to symmetry.
    """
    ###########################################################
    # YOUR CODE HERE - 2-3 lines                              #
    #                                                         #
    # Steps:                                                  #
    # 1. Compute signed error: estimated_ip - true_ip         #
    # 2. Return the mean over all elements                    #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def check_magnitude_bias(true_ip: np.ndarray, estimated_ip: np.ndarray) -> float:
    """Check if the quantizer inflates the MAGNITUDE of inner product estimates.

    While the signed bias may be near zero by symmetry, the uniform quantizer
    systematically inflates |<y, x>| because its centroids are at ±0.5 (for
    b=1) while true sphere coordinates are near 0.

    The ratio E[|estimated_ip|] / E[|true_ip|] reveals this scale inflation.

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
        Value > 1 means systematic magnitude inflation (overconfident scores).
        Value = 1 means no magnitude bias.

    Notes
    -----
    At b=1, the uniform quantizer replaces each coordinate x_j (which has
    std ≈ 0.0625 in d=256) with ±0.5.  So the "reconstructed" vectors have
    L2 norm ≈ √(256 × 0.25) = 8 instead of 1.  Inner products are inflated
    by factor ≈ 8, corresponding to the scale ratio.
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                              #
    #                                                         #
    # Hint: compute mean(|estimated_ip|) / mean(|true_ip|)   #
    # Use np.abs() and np.mean()                              #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


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
    n_db = 2_000        # database vectors (quantized and stored)
    n_query = 200       # query vectors (never quantized)
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
        # Compute estimated inner products from quantized database
        estimated_ip = compute_estimated_inner_products(queries, database, b)

        # Verify shape
        assert estimated_ip.shape == true_ip.shape, \
            f"Shape mismatch: {estimated_ip.shape} != {true_ip.shape}"

        # Compute distortion and bias
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

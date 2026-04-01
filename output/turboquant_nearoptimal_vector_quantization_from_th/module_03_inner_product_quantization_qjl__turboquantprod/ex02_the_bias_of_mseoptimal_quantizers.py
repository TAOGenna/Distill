"""
Exercise 2: The Bias of MSE-Optimal Quantizers
===============================================

In Exercise 1 you confirmed that QJL is unbiased — the relative bias
was just 0.04% even with 200 trials per pair.

Now we investigate what happens when we use TurboQuant_mse for inner
product estimation.  The paper states (Section 3.2):

    "for large enough d, we have
     E[<y, Q_mse^{-1}(Q_mse(x))>] = (2/π) · <y, x>
     which has a multiplicative bias of 2/π"

This bias arises because the MSE dequantization scale (√(2/(πd))) was
chosen to minimize ||x - x̃||², NOT to produce unbiased inner products.
The optimal MSE scale is exactly (π/2) times smaller than QJL's scale.

Your Tasks
----------
1. estimate_ip_mse_quantizer(quantizer, queries, database)
   - Quantize database with TurboQuant_mse, compute inner products with queries
   - (~6-8 lines)

2. compute_multiplicative_bias(true_ip, estimated_ip)
   - Measure the systematic scaling factor E[estimated]/E[|true|]
   - (~4-6 lines)

3. compute_bias_vs_bitwidth(d, bitwidths, n_vectors=5000)
   - Sweep over bit-widths, compute bias for each
   - Return dict mapping b → (mse_bias, qjl_bias)
   - (~8-12 lines)

The __main__ block below prints a comparison table showing the 2/pi bias
at b=1 and its diminishing effect at higher bit-widths.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Import TurboQuantMSE from module 2's solution
# ---------------------------------------------------------------------------
_module2_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "module_02_optimal_scalar_quantization__turboquantmse", "_solutions"
)
sys.path.insert(0, _module2_dir)

try:
    from ex03_full_turboquantmse_pipeline import TurboQuantMSE, CODEBOOKS
except ImportError:
    raise ImportError(
        "Could not import TurboQuantMSE from module 2.  "
        "Make sure module_02/_solutions/ex03_full_turboquantmse_pipeline.py exists."
    )

# Import QJL from this module's solution (exercise 1)
_module3_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_solutions"
)
sys.path.insert(0, _module3_dir)

try:
    from ex01_qjl_1bit_inner_product_quantizer import QJL
except ImportError:
    # Fallback: import from current directory (if running as solution)
    from ex01_qjl_1bit_inner_product_quantizer import QJL


# ---------------------------------------------------------------------------
# PROVIDED: Data generation
# ---------------------------------------------------------------------------

def generate_database_queries(n_db, n_q, d, seed=42):
    """Generate unit-norm database vectors and (non-unit) query vectors.

    Parameters
    ----------
    n_db : int
        Number of database vectors to quantize.
    n_q : int
        Number of query vectors for inner product estimation.
    d : int
        Dimension.  Must be a key in CODEBOOKS (128, 256, or 512).
    seed : int
        Random seed.

    Returns
    -------
    database : np.ndarray, shape (n_db, d)
        Unit-norm database vectors.
    queries : np.ndarray, shape (n_q, d)
        Query vectors (unit-norm, for clean bias measurement).
    """
    rng = np.random.default_rng(seed)
    # Database: unit-norm vectors (standard TurboQuant input)
    X = rng.standard_normal((n_db, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Queries: unit-norm for clean bias measurement
    Y = rng.standard_normal((n_q, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return X, Y


def compute_true_inner_products(queries, database):
    """Compute exact inner products between all query-database pairs.

    Parameters
    ----------
    queries : np.ndarray, shape (n_q, d)
    database : np.ndarray, shape (n_db, d)

    Returns
    -------
    np.ndarray, shape (n_q, n_db)
        Entry [i, j] = <queries[i], database[j]>.
    """
    return queries @ database.T


# ---------------------------------------------------------------------------
# YOUR CODE: Inner product estimation with TurboQuantMSE
# ---------------------------------------------------------------------------

def estimate_ip_mse_quantizer(quantizer, queries, database):
    """Estimate inner products using TurboQuant_mse as an inner product estimator.

    Quantizes each database vector (MSE quantization), then computes inner
    products between query vectors and the dequantized database vectors.

    This is the "naive" approach of using an MSE quantizer for inner products.
    It introduces the multiplicative bias described in the lesson.

    Parameters
    ----------
    quantizer : TurboQuantMSE
        An initialized TurboQuantMSE instance (already has rotation matrix and codebook).
    queries : np.ndarray, shape (n_q, d)
        Query vectors.
    database : np.ndarray, shape (n_db, d)
        Unit-norm database vectors to quantize.

    Returns
    -------
    np.ndarray, shape (n_q, n_db)
        Estimated inner products, entry [i, j] = <queries[i], Q_mse^{-1}(Q_mse(database[j]))>.

    Hints
    -----
    - Use quantizer.quantize_batch(database) → indices shape (n_db, d)
    - Use quantizer.dequantize_batch(indices) → X_tilde shape (n_db, d)
    - Return queries @ X_tilde.T  (shape n_q × n_db)
    """
    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Hint: quantize the whole database in one batch call     #
    # Hint: dequantize to get the reconstructed vectors       #
    # Hint: compute inner products as a matrix multiply       #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_multiplicative_bias(true_ip, estimated_ip):
    """Compute the multiplicative bias factor of an inner product estimator.

    The multiplicative bias is defined as:
        bias = E[estimated] / E[|true|]

    For an unbiased estimator (QJL), bias ≈ 1.0.
    For TurboQuant_mse at b=1, bias ≈ 2/π ≈ 0.637.

    To avoid cancellation, we compute the bias using the sign of the true
    inner product:  we want to measure how much the estimate is scaled,
    not whether it has the wrong sign.

    Parameters
    ----------
    true_ip : np.ndarray, shape (n_q, n_db) or (n,)
        True inner product values (may be positive or negative).
    estimated_ip : np.ndarray, shape (n_q, n_db) or (n,)
        Estimated inner product values.

    Returns
    -------
    float
        Multiplicative bias factor.  Ideal value: 1.0.
        Value of 2/π ≈ 0.637 indicates severe systematic underestimation.

    Hints
    -----
    - Flatten both arrays: true_flat = true_ip.ravel(), est_flat = estimated_ip.ravel()
    - Consider only pairs where |true| > 0.01 (avoid near-zero instability)
    - Multiply both by sign(true) to make all true values positive
    - Then bias = mean(sign(true)*estimated) / mean(|true|)
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                              #
    #                                                         #
    # Hint: true_flat = true_ip.ravel()                       #
    # Hint: est_flat = estimated_ip.ravel()                   #
    # Hint: mask = np.abs(true_flat) > 0.01                   #
    # Hint: signs = np.sign(true_flat[mask])                  #
    # Hint: return mean(signs * est_flat[mask]) / mean(|true_flat[mask]|)
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_bias_vs_bitwidth(d, bitwidths, n_vectors=5000):
    """Sweep over bit-widths and measure inner product bias for each method.

    For each bit-width b in bitwidths:
      1. Create TurboQuantMSE(d, b, seed=42)
      2. Create QJL(d, seed=42)
      3. Generate n_vectors database + query vectors
      4. Compute true inner products
      5. Estimate with TurboQuantMSE → compute bias
      6. Estimate with QJL (single trial) → compute bias

    Parameters
    ----------
    d : int
        Dimension.  Must be 128, 256, or 512 (codebook available).
    bitwidths : list of int
        Bit-widths to test, e.g. [1, 2, 3, 4].
    n_vectors : int
        Number of database/query vectors (use same count for both).

    Returns
    -------
    dict
        Mapping b → {"mse_bias": float, "qjl_bias": float}.

    Hints
    -----
    - For QJL, use a single QJL instance (one S matrix) for all vectors.
      The bias is the same regardless of how many trials you average.
    - Use a SMALLER number of pairs for efficiency:
        n_db = min(n_vectors, 500); n_q = min(n_vectors, 200)
    - For QJL estimates: quantize each db vector, get z, then estimate
      inner products against all queries at once.
    """
    n_db = min(n_vectors, 500)
    n_q = min(n_vectors, 200)
    database, queries = generate_database_queries(n_db, n_q, d, seed=100)
    true_ips = compute_true_inner_products(queries, database)

    results = {}
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint: for b in bitwidths:                               #
    #   quantizer = TurboQuantMSE(d, b, seed=42)             #
    #   estimated_mse = estimate_ip_mse_quantizer(...)        #
    #   mse_bias = compute_multiplicative_bias(true_ips, estimated_mse) #
    #                                                         #
    #   qjl = QJL(d, seed=42)                                #
    #   X_tilde_qjl = np.vstack([qjl.dequantize(qjl.quantize(database[i])) #
    #                             for i in range(n_db)])      #
    #   estimated_qjl = queries @ X_tilde_qjl.T              #
    #   qjl_bias = compute_multiplicative_bias(true_ips, estimated_qjl) #
    #                                                         #
    #   results[b] = {"mse_bias": mse_bias, "qjl_bias": qjl_bias}  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################
    return results


# ---------------------------------------------------------------------------
# PROVIDED: Table formatter
# ---------------------------------------------------------------------------

def print_bias_comparison_table(results, d):
    """Print a formatted bias comparison table.

    Parameters
    ----------
    results : dict
        From compute_bias_vs_bitwidth: b → {"mse_bias": float, "qjl_bias": float}.
    d : int
        Dimension used in the experiment.
    """
    print()
    print(f"  Multiplicative Bias: TurboQuant_mse vs QJL  (d={d})")
    print(f"  {'b':>3}  {'TurboQuant_mse bias':>20}  {'QJL bias':>12}  {'2/pi ratio':>12}  Notes")
    print(f"  {'─'*3}  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*25}")
    two_over_pi = 2.0 / np.pi
    for b in sorted(results.keys()):
        mse_b = results[b]["mse_bias"]
        qjl_b = results[b]["qjl_bias"]
        # How close is mse_bias to 2/pi?  (only meaningful at b=1)
        closeness = abs(mse_b - two_over_pi) / two_over_pi
        if b == 1:
            note = f"← 2/pi = {two_over_pi:.4f} (diff: {closeness*100:.1f}%)"
        elif b == 4:
            note = "← bias nearly gone"
        else:
            note = ""
        print(f"  {b:>3}  {mse_b:>20.4f}  {qjl_b:>12.4f}  {two_over_pi:>12.4f}  {note}")
    print()
    print(f"  Key result: At b=1, TurboQuant_mse has multiplicative bias ≈ 2/pi ≈ {two_over_pi:.4f}")
    print(f"  QJL maintains bias ≈ 1.00 at all bit-widths (unbiased by construction).")


# ---------------------------------------------------------------------------
# TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Bias of MSE-Optimal Quantizers for Inner Product Estimation")
    print("=" * 70)

    d = 128
    bitwidths = [1, 2, 3, 4]
    two_over_pi = 2.0 / np.pi

    # -----------------------------------------------------------------
    # TEST 1: Bias at b=1 (should be ≈ 2/π)
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 1: Bias at b=1 — confirming 2/π factor")
    print("─" * 50)

    n_db, n_q = 2000, 500
    database, queries = generate_database_queries(n_db, n_q, d, seed=1)
    true_ips = compute_true_inner_products(queries, database)

    qmse_b1 = TurboQuantMSE(d=d, b=1, seed=42)
    estimated_b1 = estimate_ip_mse_quantizer(qmse_b1, queries, database)
    bias_b1 = compute_multiplicative_bias(true_ips, estimated_b1)

    print(f"  TurboQuant_mse b=1 bias:   {bias_b1:.4f}")
    print(f"  Theoretical (2/π):         {two_over_pi:.4f}")
    print(f"  Difference:                {abs(bias_b1 - two_over_pi):.4f}")
    if abs(bias_b1 - two_over_pi) < 0.05:
        print(f"  → PASSES: bias matches 2/pi within 5%")
    else:
        print(f"  → WARNING: bias deviates more than 5% from 2/pi")

    # -----------------------------------------------------------------
    # TEST 2: Bias sweep across all bit-widths
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 2: Bias across bit-widths  (2000 db, 200 queries)")
    print("─" * 50)

    results = compute_bias_vs_bitwidth(d=d, bitwidths=bitwidths, n_vectors=2000)
    print_bias_comparison_table(results, d)

    # -----------------------------------------------------------------
    # TEST 3: Verify QJL is always unbiased
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 3: QJL unbiasedness at all bit-widths")
    print("─" * 50)
    print("  (QJL ignores bit-width — always 1 bit — but bias should always ≈ 1.0)")
    print()
    for b in bitwidths:
        qjl_bias = results[b]["qjl_bias"]
        status = "OK" if abs(qjl_bias - 1.0) < 0.05 else "CHECK"
        print(f"    b={b}  QJL bias = {qjl_bias:.4f}  [{status}]")

    # -----------------------------------------------------------------
    # TEST 4: Error distribution at b=1 (should be skewed for MSE)
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 4: Error distribution at b=1")
    print("─" * 50)

    errors_mse_b1 = (estimated_b1 - true_ips).ravel()
    # For an unbiased estimator, mean(error) ≈ 0
    # For TurboQuant_mse at b=1, mean(error) ≈ -(1 - 2/π) · mean(|true|)
    mean_true = np.mean(np.abs(true_ips))
    expected_mean_error = -(1.0 - two_over_pi) * mean_true
    print(f"  Mean IP error (TurboQuant_mse b=1):  {np.mean(errors_mse_b1):.5f}")
    print(f"  Expected mean error (-(1-2/π)·E|IP|): {expected_mean_error:.5f}")
    print(f"  Std of errors:                         {np.std(errors_mse_b1):.5f}")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    mse_b1 = results[1]["mse_bias"]
    mse_b4 = results[4]["mse_bias"]
    qjl_b1 = results[1]["qjl_bias"]
    print(f"  b=1: TurboQuant_mse bias = {mse_b1:.4f}  (theory: 2/pi = {two_over_pi:.4f})")
    print(f"  b=4: TurboQuant_mse bias = {mse_b4:.4f}  (approaches 1.0 as b→∞)")
    print(f"  QJL: always unbiased    = {qjl_b1:.4f}  (by construction)")
    print()
    print("  Conclusion: MSE-optimal quantizers introduce multiplicative bias")
    print("  in inner product estimation.  At b=1, the bias factor is exactly 2/pi.")
    print("  TurboQuant_prod (Exercise 3) fixes this with a two-stage approach.")

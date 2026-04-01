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
2. compute_multiplicative_bias(true_ip, estimated_ip)
3. compute_bias_vs_bitwidth(d, bitwidths, n_vectors=5000)
"""

import sys
import os
import numpy as np

_module2_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "module_02_optimal_scalar_quantization__turboquantmse", "_solutions"
)
sys.path.insert(0, _module2_dir)

try:
    from ex03_full_turboquantmse_pipeline import TurboQuantMSE, CODEBOOKS
except ImportError:
    raise ImportError(
        "Could not import TurboQuantMSE from module 2.  "
        "Make sure module_02/_solutions/ex03_full_turboquantmse_pipeline.py exists."
    )

# Import QJL from exercise 1 solution
_sol_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _sol_dir)

from ex01_qjl_1bit_inner_product_quantizer import QJL


# ---------------------------------------------------------------------------
# PROVIDED: Data generation
# ---------------------------------------------------------------------------

def generate_database_queries(n_db, n_q, d, seed=42):
    """Generate unit-norm database vectors and (unit-norm) query vectors.

    Parameters
    ----------
    n_db : int
    n_q : int
    d : int
    seed : int

    Returns
    -------
    database : np.ndarray, shape (n_db, d)
    queries : np.ndarray, shape (n_q, d)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_db, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n_q, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return X, Y


def compute_true_inner_products(queries, database):
    """Compute exact inner products, shape (n_q, n_db)."""
    return queries @ database.T


# ---------------------------------------------------------------------------
# SOLUTION: Inner product estimation with TurboQuantMSE
# ---------------------------------------------------------------------------

def estimate_ip_mse_quantizer(quantizer, queries, database):
    """Estimate inner products using TurboQuant_mse as an inner product estimator.

    Parameters
    ----------
    quantizer : TurboQuantMSE
    queries : np.ndarray, shape (n_q, d)
    database : np.ndarray, shape (n_db, d)

    Returns
    -------
    np.ndarray, shape (n_q, n_db)
    """
    indices = quantizer.quantize_batch(database)       # (n_db, d)
    X_tilde = quantizer.dequantize_batch(indices)      # (n_db, d)
    return queries @ X_tilde.T                         # (n_q, n_db)


def compute_multiplicative_bias(true_ip, estimated_ip):
    """Compute the multiplicative bias factor of an inner product estimator.

    Parameters
    ----------
    true_ip : np.ndarray
    estimated_ip : np.ndarray

    Returns
    -------
    float
        Multiplicative bias factor ≈ 1.0 for unbiased, ≈ 0.637 for MSE at b=1.
    """
    true_flat = true_ip.ravel()
    est_flat = estimated_ip.ravel()
    # Only consider pairs where |true| > 0.01 to avoid near-zero instability
    mask = np.abs(true_flat) > 0.01
    signs = np.sign(true_flat[mask])
    # Signed estimate vs absolute true: bias = E[sign*estimated] / E[|true|]
    return float(np.mean(signs * est_flat[mask]) / np.mean(np.abs(true_flat[mask])))


def compute_bias_vs_bitwidth(d, bitwidths, n_vectors=5000):
    """Sweep over bit-widths and measure inner product bias for each method.

    Parameters
    ----------
    d : int
    bitwidths : list of int
    n_vectors : int

    Returns
    -------
    dict : b → {"mse_bias": float, "qjl_bias": float}
    """
    n_db = min(n_vectors, 500)
    n_q = min(n_vectors, 200)
    database, queries = generate_database_queries(n_db, n_q, d, seed=100)
    true_ips = compute_true_inner_products(queries, database)

    results = {}
    for b in bitwidths:
        # MSE quantizer bias
        quantizer = TurboQuantMSE(d, b, seed=42)
        estimated_mse = estimate_ip_mse_quantizer(quantizer, queries, database)
        mse_bias = compute_multiplicative_bias(true_ips, estimated_mse)

        # QJL bias (single trial per vector — QJL is unbiased even without averaging)
        qjl = QJL(d, seed=42)
        X_tilde_qjl = np.vstack([
            qjl.dequantize(qjl.quantize(database[i]))
            for i in range(n_db)
        ])
        estimated_qjl = queries @ X_tilde_qjl.T
        qjl_bias = compute_multiplicative_bias(true_ips, estimated_qjl)

        results[b] = {"mse_bias": mse_bias, "qjl_bias": qjl_bias}

    return results


# ---------------------------------------------------------------------------
# PROVIDED: Table formatter
# ---------------------------------------------------------------------------

def print_bias_comparison_table(results, d):
    """Print a formatted bias comparison table."""
    print()
    print(f"  Multiplicative Bias: TurboQuant_mse vs QJL  (d={d})")
    print(f"  {'b':>3}  {'TurboQuant_mse bias':>20}  {'QJL bias':>12}  {'2/pi ratio':>12}  Notes")
    print(f"  {'─'*3}  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*25}")
    two_over_pi = 2.0 / np.pi
    for b in sorted(results.keys()):
        mse_b = results[b]["mse_bias"]
        qjl_b = results[b]["qjl_bias"]
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
    # TEST 4: Error distribution at b=1
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 4: Error distribution at b=1")
    print("─" * 50)

    errors_mse_b1 = (estimated_b1 - true_ips).ravel()
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
    print("  TurboQuant_prod (Exercise 3) fixes this by directly minimizing D_prod,")
    print("  the inner product distortion, instead of MSE reconstruction error.")

"""
Exercise 4: Inner Product Distortion — Prod vs MSE Comparison
=============================================================

In Exercise 3 you saw TurboQuant_prod achieve empirical D_prod values close
to the theoretical predictions:
  b=1: D_prod ≈ 0.012219  (theory: 0.012266 = 1.57/128)
  b=2: D_prod ≈ 0.004419  (theory: 0.004375 = 0.56/128)
  b=3: D_prod ≈ 0.001421  (theory: 0.001406 = 0.18/128)
  b=4: D_prod ≈ 0.000418  (theory: 0.000367 = 0.047/128)

Now we do a comprehensive comparison between TurboQuant_prod and
TurboQuant_mse for inner product estimation.

Key insight: At high bit-widths (b≥3), TurboQuant_mse catches up despite
being biased — all b bits go to MSE minimization, and the residual bias
at b=3 is only ~3%.
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
    from ex03_full_turboquantmse_pipeline import TurboQuantMSE
except ImportError:
    raise ImportError("Cannot import TurboQuantMSE from module 2 solutions.")

_sol_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _sol_dir)

from ex03_twostage_turboquantprod import TurboQuantProd


# ---------------------------------------------------------------------------
# PROVIDED: Data generation
# ---------------------------------------------------------------------------

def generate_database_queries(n_db, n_q, d, seed=42):
    """Generate realistic embedding-like vectors for comparison.

    Parameters
    ----------
    n_db, n_q, d : int
    seed : int

    Returns
    -------
    database : np.ndarray, shape (n_db, d), unit-norm rows
    queries : np.ndarray, shape (n_q, d), unit-norm rows
    """
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal((10, d))
    shared /= np.linalg.norm(shared, axis=1, keepdims=True)

    def make_vectors(n):
        base = rng.standard_normal((n, d))
        mix_idx = rng.integers(0, 10, size=n)
        mix = 0.2 * shared[mix_idx]
        v = base + mix
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    return make_vectors(n_db), make_vectors(n_q)


def compute_true_inner_products(queries, database):
    """Compute exact inner products, shape (n_q, n_db)."""
    return queries @ database.T


# ---------------------------------------------------------------------------
# SOLUTION: Error collection and statistics
# ---------------------------------------------------------------------------

def collect_ip_errors(quantizer_class, queries, database, b, d, n_trials=30):
    """Collect inner product errors for a given quantizer class.

    Parameters
    ----------
    quantizer_class : type
        Either TurboQuantProd or TurboQuantMSE.
    queries : np.ndarray, shape (n_q, d)
    database : np.ndarray, shape (n_db, d)
    b : int
    d : int
    n_trials : int

    Returns
    -------
    errors : np.ndarray, shape (n_q * n_db,)
    """
    true_ips = compute_true_inner_products(queries, database)
    n_q, n_db = len(queries), len(database)

    if quantizer_class is TurboQuantMSE:
        q = TurboQuantMSE(d, b, seed=42)
        idx = q.quantize_batch(database)
        X_tilde = q.dequantize_batch(idx)
        estimated = queries @ X_tilde.T
        return (estimated - true_ips).ravel()

    elif quantizer_class is TurboQuantProd:
        sum_est = np.zeros((n_q, n_db))
        for seed in range(n_trials):
            qp = TurboQuantProd(d, b, seed=seed * 13)
            for j in range(n_db):
                idx, z, gamma = qp.quantize(database[j])
                if idx is not None:
                    x_mse = qp.mse_quantizer.dequantize(idx)
                else:
                    x_mse = np.zeros(d)
                ip_mse = queries @ x_mse
                scale = np.sqrt(np.pi / 2) / d
                ip_qjl = gamma * scale * (queries @ qp.qjl.S.T @ z)
                sum_est[:, j] += ip_mse + ip_qjl
        return ((sum_est / n_trials) - true_ips).ravel()

    else:
        raise ValueError(f"Unknown quantizer class: {quantizer_class}")


def compute_error_statistics(errors):
    """Compute summary statistics for inner product errors.

    Parameters
    ----------
    errors : np.ndarray, shape (n,)

    Returns
    -------
    dict with keys: "mean", "std", "mse", "bias_sq", "variance"
    """
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    mse = float(np.mean(errors ** 2))
    bias_sq = mean ** 2
    variance = mse - bias_sq
    return {
        "mean": mean,
        "std": std,
        "mse": mse,
        "bias_sq": bias_sq,
        "variance": max(0.0, variance),  # clip numerical noise
    }


def compute_theoretical_prod_bound(b, d, y_norm_sq=1.0):
    """Compute the theoretical D_prod upper bound for TurboQuant_prod.

    Parameters
    ----------
    b : int
    d : int
    y_norm_sq : float

    Returns
    -------
    float
    """
    EXACT_VALUES = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}
    if b in EXACT_VALUES:
        return EXACT_VALUES[b] * y_norm_sq / d
    else:
        return (np.sqrt(3) * np.pi ** 2 * y_norm_sq / d) / (4 ** b)


def format_comparison_row(method, b, d, stats, bound):
    """Format one row of the comparison table.

    Parameters
    ----------
    method : str
    b : int
    d : int
    stats : dict
    bound : float or None

    Returns
    -------
    str
    """
    short = method[:16].ljust(16)
    if bound is not None:
        bound_str = f"{bound:.6f}"
        ratio_str = f"{stats['mse']/bound:.2f}×"
    else:
        bound_str = "  N/A   "
        ratio_str = "  N/A"
    return (f"{short}  {b:>2}  {stats['mean']:>+10.5f}  {stats['std']:>10.5f}  "
            f"{stats['mse']:>14.6f}  {bound_str:>10}  {ratio_str:>7}")


# ---------------------------------------------------------------------------
# TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("Comprehensive Inner Product Distortion: TurboQuant_prod vs TurboQuant_mse")
    print("=" * 80)

    d = 128
    bitwidths = [1, 2, 3, 4]

    n_db, n_q = 200, 50
    database, queries = generate_database_queries(n_db, n_q, d, seed=99)
    true_ips = compute_true_inner_products(queries, database)

    print(f"\n  Dataset: n_db={n_db}, n_q={n_q}, d={d}")
    print(f"  True IP range: [{true_ips.min():.3f}, {true_ips.max():.3f}]")
    print(f"  True IP mean |IP|: {np.mean(np.abs(true_ips)):.4f}")

    # -----------------------------------------------------------------
    # TEST 1: Full comparison table
    # -----------------------------------------------------------------
    print()
    print("─" * 80)
    print("Test 1: Inner Product Error comparison table")
    print("─" * 80)
    print()
    print(f"  {'Method':<18}  {'b':>2}  {'mean err':>10}  {'std err':>10}  "
          f"{'MSE (D_prod)':>14}  {'theory':>10}  {'ratio':>7}")
    print(f"  {'─'*18}  {'─'*2}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*7}")

    all_rows = []
    for b in bitwidths:
        for cls, name in [(TurboQuantProd, "TurboQuant_prod"), (TurboQuantMSE, "TurboQuant_mse")]:
            errors = collect_ip_errors(cls, queries, database, b, d, n_trials=30)
            stats = compute_error_statistics(errors)
            bound = compute_theoretical_prod_bound(b, d) if cls is TurboQuantProd else None
            row = format_comparison_row(name, b, d, stats, bound)
            print(f"  {row}")
            all_rows.append({
                "method": name, "b": b, "stats": stats, "bound": bound
            })
        print()

    # -----------------------------------------------------------------
    # TEST 2: Verify TurboQuant_prod is unbiased
    # -----------------------------------------------------------------
    print()
    print("─" * 80)
    print("Test 2: Unbiasedness check — TurboQuant_prod mean error")
    print("─" * 80)
    print()
    prod_rows = [r for r in all_rows if r["method"] == "TurboQuant_prod"]
    mean_abs_true = float(np.mean(np.abs(true_ips)))
    all_unbiased = True
    for r in prod_rows:
        b = r["b"]
        mean_err = r["stats"]["mean"]
        rel_bias = abs(mean_err) / mean_abs_true
        status = "OK" if rel_bias < 0.05 else "CHECK"
        if status != "OK":
            all_unbiased = False
        print(f"  b={b}: mean error = {mean_err:+.5f}  "
              f"(rel bias = {rel_bias:.4f})  [{status}]")

    # -----------------------------------------------------------------
    # TEST 3: TurboQuant_mse bias at b=1
    # -----------------------------------------------------------------
    print()
    print("─" * 80)
    print("Test 3: Bias check — TurboQuant_mse should have bias at b=1")
    print("─" * 80)
    print()
    mse_rows = [r for r in all_rows if r["method"] == "TurboQuant_mse"]
    for r in mse_rows:
        b = r["b"]
        mean_err = r["stats"]["mean"]
        expected_bias = (1.0 - 2.0 / np.pi) * mean_abs_true
        print(f"  b={b}: mean error = {mean_err:+.5f}  "
              f"(expected |bias| ≈ {expected_bias:.4f} at b=1)")

    # -----------------------------------------------------------------
    # TEST 4: Theory vs empirical for TurboQuant_prod
    # -----------------------------------------------------------------
    print()
    print("─" * 80)
    print("Test 4: Theoretical bound verification (ratio ≤ 2.0 = within 2×)")
    print("─" * 80)
    print()
    all_within_bound = True
    for r in prod_rows:
        b = r["b"]
        mse = r["stats"]["mse"]
        bound = r["bound"]
        ratio = mse / bound
        status = "OK" if ratio <= 2.0 else "CHECK"
        if status != "OK":
            all_within_bound = False
        print(f"  b={b}: D_prod = {mse:.6f}  bound = {bound:.6f}  "
              f"ratio = {ratio:.3f}  [{status}]")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print()
    print("=" * 80)
    print("Summary: Which method wins for inner product estimation?")
    print("=" * 80)
    print()
    print("  Insight from the comparison:")
    print("  - At b=1,2: TurboQuant_prod has LOWER distortion (unbiased + small variance)")
    print("  - At b=1: TurboQuant_mse has multiplicative bias = 2/pi ≈ 0.637")
    print("  - At b=3,4: TurboQuant_mse 'catches up' because its bias shrinks to ~3%")
    print("  - TurboQuant_prod is always unbiased; TurboQuant_mse converges to unbiased")
    print()

    print(f"  {'b':>3}  {'prod MSE':>12}  {'mse MSE':>12}  Winner")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*20}")
    prod_by_b = {r["b"]: r["stats"]["mse"] for r in prod_rows}
    mse_by_b = {r["b"]: r["stats"]["mse"] for r in mse_rows}
    for b in bitwidths:
        p = prod_by_b[b]
        m = mse_by_b[b]
        winner = "TurboQuant_prod" if p < m else "TurboQuant_mse"
        print(f"  {b:>3}  {p:>12.6f}  {m:>12.6f}  {winner}")

    print()
    print("  The comparison table above demonstrates the key result: for")
    print("  inner product applications at low bit-widths, use TurboQuant_prod.")
    print("  At high bit-widths, either method works — choose based on")
    print("  whether unbiasedness or simplicity is more important.")
    print()
    if all_unbiased and all_within_bound:
        print("  All checks PASSED — comparison complete.")
    else:
        print("  Some checks need review — see details above.")

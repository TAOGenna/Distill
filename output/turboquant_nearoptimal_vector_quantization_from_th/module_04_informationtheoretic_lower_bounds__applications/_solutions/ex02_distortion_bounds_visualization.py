"""
Exercise 2: Distortion Bounds Visualization
=============================================

In Exercise 1 you computed the Shannon Lower Bound and confirmed:
  - b=1: TurboQuant D_mse = 0.363, lower bound = 0.250, gap = 1.45×
  - b=2: TurboQuant D_mse = 0.117, lower bound = 0.0625, gap = 1.88×
  - b=3: TurboQuant D_mse = 0.034, lower bound = 0.0156, gap = 2.21×
  - b=4: TurboQuant D_mse = 0.0095, lower bound = 0.0039, gap = 2.43×

Now you will analyze these bounds more carefully and answer two questions:

  Q1: Is the gap fundamental or an artifact of the Panter-Dite approximation?
  Q2: When does TurboQuant_mse become competitive with TurboQuant_prod
      for inner product estimation (as b increases, the bias 2/π shrinks)?

Your Tasks
----------
1. compute_all_mse_bounds(b_range, d)          — gather lower/empirical/upper bounds
2. compute_all_prod_bounds(b_range, d, y_norm_sq) — same for inner product distortion
3. compute_optimality_gap(empirical, lower)    — ratio empirical/lower per bit-width
4. analyze_crossover(mse_distortion, prod_distortion, bitwidths)
                                               — find crossover bit-width

Key Insight
-----------
The gap between TurboQuant and the lower bound comes from two sources:
  1. Coordinate independence assumption: quantizing each coordinate independently
     instead of using joint multi-dimensional vector quantization
  2. Panter-Dite high-resolution approximation: asymptotically tight but loose
     for small b.  At b=1, we use the exact codebook, so this gap is absent —
     the 1.45× gap is the *minimum achievable* with scalar quantization.

At b≥3, TurboQuant_mse's bias (2/π factor) drops below 3%, meaning the
MSE-optimal quantizer becomes nearly as good as TurboQuant_prod for inner
products.  This analysis identifies the crossover point.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# PROVIDED: Import from Exercise 1 (Shannon bounds)
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from ex01_shannon_lower_bound_computation import (
    compute_exact_slb,
    compute_simplified_slb,
)

# ---------------------------------------------------------------------------
# PROVIDED: TurboQuant implementations from previous modules
# ---------------------------------------------------------------------------

_mod2_sol = os.path.join(
    _this_dir, "..", "..",
    "module_02_optimal_scalar_quantization__turboquantmse",
    "_solutions"
)
_mod3_sol = os.path.join(
    _this_dir, "..", "..",
    "module_03_inner_product_quantization_qjl__turboquantprod",
    "_solutions"
)
sys.path.insert(0, os.path.normpath(_mod2_sol))
sys.path.insert(0, os.path.normpath(_mod3_sol))

try:
    from ex03_full_turboquantmse_pipeline import TurboQuantMSE, CODEBOOKS
    from ex03_twostage_turboquantprod import TurboQuantProd
except ImportError as e:
    raise ImportError(
        f"Could not import TurboQuant classes from previous modules: {e}\n"
        "Make sure modules 2 and 3 solutions exist."
    )

# ---------------------------------------------------------------------------
# PROVIDED: Test data generation
# ---------------------------------------------------------------------------

def generate_test_data(n, d, seed=42):
    """Generate n uniformly random unit vectors in R^d.

    Parameters
    ----------
    n : int
        Number of vectors.
    d : int
        Dimension.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (n, d)
        Matrix of unit-norm vectors.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def measure_mse_distortion(quantizer, vectors):
    """Measure average MSE distortion of a TurboQuantMSE quantizer.

    Parameters
    ----------
    quantizer : TurboQuantMSE
    vectors : np.ndarray, shape (n, d)

    Returns
    -------
    float
        Average ||x - x̃||² over all vectors.
    """
    idx_batch = quantizer.quantize_batch(vectors)
    X_tilde = quantizer.dequantize_batch(idx_batch)
    return float(np.mean(np.sum((vectors - X_tilde) ** 2, axis=1)))


def measure_prod_distortion_fast(d, b, vectors, queries, n_seeds=20):
    """Estimate D_prod = E[|<y, x> - <y, x̃>|²] for TurboQuantProd.

    Averages over multiple seeds for reliability.

    Parameters
    ----------
    d : int
        Dimension.
    b : int
        Bit-width.
    vectors : np.ndarray, shape (n_db, d)
        Database vectors.
    queries : np.ndarray, shape (n_q, d)
        Query vectors.
    n_seeds : int
        Number of random seeds to average over.

    Returns
    -------
    float
        Estimated inner product distortion.
    """
    true_ips = queries @ vectors.T  # (n_q, n_db)
    total_sq_err = 0.0
    count = 0

    for seed in range(n_seeds):
        qp = TurboQuantProd(d, b, seed=seed * 17)
        for j, x in enumerate(vectors):
            idx, z, gamma = qp.quantize(x)
            # Use dequantize for inner product estimation
            x_approx = qp.dequantize(idx, z, gamma)
            ip_est = queries @ x_approx  # (n_q,)
            total_sq_err += float(np.sum((ip_est - true_ips[:, j]) ** 2))
            count += len(queries)

    return total_sq_err / count


# ---------------------------------------------------------------------------
# SOLUTION: Four functions implemented
# ---------------------------------------------------------------------------

def compute_all_mse_bounds(b_range, d):
    """Gather lower bound, empirical MSE, and theoretical upper bound for each b.

    The theoretical upper bound is (√3π/2) · (1/4)^b (Panter-Dite formula).
    The empirical MSE is measured using TurboQuantMSE on random unit vectors.

    Parameters
    ----------
    b_range : iterable of int
        Bit-widths to evaluate (e.g. range(1, 5)).
    d : int
        Dimension. Must be in CODEBOOKS (128, 256, or 512).

    Returns
    -------
    dict with keys:
        "lower"     : list of float — lower bounds 1/4^b
        "empirical" : list of float — measured D_mse for TurboQuantMSE
        "upper"     : list of float — Panter-Dite upper bounds (√3π/2)/4^b
        "bitwidths" : list of int   — the b values used
    """
    lower_list = []
    empirical_list = []
    upper_list = []
    bw_list = []

    panter_dite = np.sqrt(3) * np.pi / 2

    for b in b_range:
        lb = compute_simplified_slb(b)
        lower_list.append(lb)

        # Empirical measurement
        vectors = generate_test_data(n=2000, d=d, seed=b * 7)
        qmse = TurboQuantMSE(d, b, seed=42)
        emp = measure_mse_distortion(qmse, vectors)
        empirical_list.append(emp)

        upper_list.append(panter_dite * lb)
        bw_list.append(b)

    return {
        "lower": lower_list,
        "empirical": empirical_list,
        "upper": upper_list,
        "bitwidths": bw_list,
    }


def compute_all_prod_bounds(b_range, d, y_norm_sq=1.0):
    """Gather lower bound, empirical D_prod, and theoretical upper bound.

    The inner product lower bound from Shannon + Yao:
        D_prod_lower = ||y||² / (d · 4^b)

    The theoretical upper bound:
        D_prod_upper = (√3 π² · ||y||²) / (d · 4^b)

    Parameters
    ----------
    b_range : iterable of int
        Bit-widths to evaluate.
    d : int
        Dimension. Must be in CODEBOOKS.
    y_norm_sq : float
        ||y||². Default 1.0 (unit-norm queries).

    Returns
    -------
    dict with keys:
        "lower"     : list of float — lower bounds ||y||²/(d·4^b)
        "empirical" : list of float — measured D_prod (averaged over seeds)
        "upper"     : list of float — upper bounds (√3π²·||y||²)/(d·4^b)
        "bitwidths" : list of int   — the b values used
    """
    lower_list = []
    empirical_list = []
    upper_list = []
    bw_list = []

    upper_factor = np.sqrt(3) * np.pi ** 2

    # Generate database and queries once
    vectors = generate_test_data(n=50, d=d, seed=200)
    queries = generate_test_data(n=20, d=d, seed=201)

    for b in b_range:
        lb_mse = compute_simplified_slb(b)
        # Inner product lower bound: ||y||² / (d * 4^b)
        lb_prod = y_norm_sq * lb_mse / d
        lower_list.append(lb_prod)

        # Upper bound: (sqrt(3)*pi^2 * ||y||^2) / (d * 4^b)
        ub_prod = upper_factor * y_norm_sq * lb_mse / d
        upper_list.append(ub_prod)

        # Empirical
        emp = measure_prod_distortion_fast(d, b, vectors, queries, n_seeds=15)
        empirical_list.append(emp)

        bw_list.append(b)

    return {
        "lower": lower_list,
        "empirical": empirical_list,
        "upper": upper_list,
        "bitwidths": bw_list,
    }


def compute_optimality_gap(empirical, lower):
    """Compute the optimality gap ratio empirical / lower bound for each b.

    Parameters
    ----------
    empirical : list of float
        Empirical distortion values (one per bit-width).
    lower : list of float
        Lower bound values (one per bit-width, same length).

    Returns
    -------
    list of float
        Ratios empirical[i] / lower[i], one per bit-width.
    """
    assert len(empirical) == len(lower), "Lists must have the same length"
    eps = 1e-15
    return [e / (l + eps) for e, l in zip(empirical, lower)]


def analyze_crossover(mse_distortion, prod_distortion, bitwidths):
    """Find the bit-width where TurboQuant_mse becomes competitive with
    TurboQuant_prod for inner product estimation.

    At b=1, TurboQuant_mse has multiplicative bias 2/π ≈ 0.637, so its
    inner product estimates are systematically wrong.  As b increases, the
    bias shrinks. This function finds the crossover where the bias is < 3%.

    Parameters
    ----------
    mse_distortion : list of float
        D_prod values when using TurboQuant_mse for inner product estimation.
    prod_distortion : list of float
        D_prod values when using TurboQuant_prod (unbiased).
    bitwidths : list of int
        Corresponding bit-widths.

    Returns
    -------
    dict with keys:
        "bias_at_b"     : list of float — MSE/Prod distortion ratio per b
        "crossover_b"   : int or None   — first b where ratio <= 1.1
        "gap_reduction" : list of float — prod/mse ratio (how much QJL helps)
    """
    bias_at_b = [m / (p + 1e-15) for m, p in zip(mse_distortion, prod_distortion)]
    gap_reduction = [p / (m + 1e-15) for p, m in zip(prod_distortion, mse_distortion)]

    crossover_b = None
    for i, b in enumerate(bitwidths):
        if bias_at_b[i] <= 1.1:
            crossover_b = b
            break

    return {
        "bias_at_b": bias_at_b,
        "crossover_b": crossover_b,
        "gap_reduction": gap_reduction,
    }


# ---------------------------------------------------------------------------
# __main__ TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TurboQuant Distortion Bounds: Analysis vs Lower Bound")
    print("=" * 70)

    d = 128
    b_range = [1, 2, 3, 4]

    # ------------------------------------------------------------------
    # Part 1: MSE distortion bounds
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print(f"Part 1: MSE Bounds  (d={d})")
    print("─" * 60)
    print()

    mse_data = compute_all_mse_bounds(b_range, d)

    print(f"  {'b':>3}  {'lower 1/4^b':>14}  {'empirical':>12}  {'upper PD':>10}  {'gap':>8}")
    print(f"  {'─'*3}  {'─'*14}  {'─'*12}  {'─'*10}  {'─'*8}")

    mse_gaps = compute_optimality_gap(mse_data["empirical"], mse_data["lower"])
    for i, b in enumerate(mse_data["bitwidths"]):
        print(
            f"  {b:3d}  {mse_data['lower'][i]:14.6f}"
            f"  {mse_data['empirical'][i]:12.6f}"
            f"  {mse_data['upper'][i]:10.6f}"
            f"  {mse_gaps[i]:8.3f}×"
        )

    # ------------------------------------------------------------------
    # Part 2: Inner product distortion bounds
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print(f"Part 2: Inner Product Bounds  (d={d}, ||y||²=1)")
    print("─" * 60)
    print()

    prod_data = compute_all_prod_bounds(b_range, d, y_norm_sq=1.0)

    print(f"  {'b':>3}  {'lower':>12}  {'empirical':>12}  {'upper':>12}  {'gap':>8}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")

    prod_gaps = compute_optimality_gap(prod_data["empirical"], prod_data["lower"])
    for i, b in enumerate(prod_data["bitwidths"]):
        print(
            f"  {b:3d}  {prod_data['lower'][i]:12.7f}"
            f"  {prod_data['empirical'][i]:12.7f}"
            f"  {prod_data['upper'][i]:12.7f}"
            f"  {prod_gaps[i]:8.3f}×"
        )

    # ------------------------------------------------------------------
    # Part 3: MSE vs Prod for inner product — crossover analysis
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 3: Crossover — when does TurboQuant_mse match TurboQuant_prod?")
    print("─" * 60)
    print()

    # Compute D_prod using TurboQuantMSE (biased) vs TurboQuantProd (unbiased)
    db_cross = generate_test_data(n=50, d=d, seed=100)
    q_cross = generate_test_data(n=20, d=d, seed=101)
    true_ips_cross = q_cross @ db_cross.T

    mse_ip_distortions = []
    prod_ip_distortions = []

    for b in b_range:
        # TurboQuantMSE as inner product estimator (biased)
        qmse = TurboQuantMSE(d, b, seed=42)
        idx_batch = qmse.quantize_batch(db_cross)
        X_tilde = qmse.dequantize_batch(idx_batch)
        ip_mse = q_cross @ X_tilde.T
        mse_ip_dist = float(np.mean((ip_mse - true_ips_cross) ** 2))
        mse_ip_distortions.append(mse_ip_dist)

        # TurboQuantProd (unbiased)
        prod_ip_dist = measure_prod_distortion_fast(d, b, db_cross, q_cross, n_seeds=10)
        prod_ip_distortions.append(prod_ip_dist)

    crossover_info = analyze_crossover(mse_ip_distortions, prod_ip_distortions, b_range)

    print(f"  {'b':>3}  {'D_prod (MSE q.)':>16}  {'D_prod (Prod q.)':>17}  {'ratio MSE/Prod':>15}  Better?")
    print(f"  {'─'*3}  {'─'*16}  {'─'*17}  {'─'*15}  {'─'*8}")

    for i, b in enumerate(b_range):
        ratio = crossover_info["bias_at_b"][i]
        better = "prod" if ratio > 1.0 else "≈equal"
        print(
            f"  {b:3d}  {mse_ip_distortions[i]:16.7f}"
            f"  {prod_ip_distortions[i]:17.7f}"
            f"  {ratio:15.3f}×"
            f"  {better}"
        )

    crossover_b = crossover_info["crossover_b"]
    print()
    if crossover_b is not None:
        print(f"  Crossover at b={crossover_b}: TurboQuant_mse and _prod have similar")
        print(f"  inner product distortion (< 10% difference).  For b >= {crossover_b},")
        print(f"  the 2/π bias in TurboQuant_mse is negligible.")
    else:
        print("  No crossover in b=1..4 range; TurboQuant_prod remains better.")

    # ------------------------------------------------------------------
    # Part 4: Answering the key analytical question
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 4: Is the gap fundamental or an artifact?")
    print("─" * 60)
    print()
    print("  The optimality gap (empirical / lower bound) for MSE distortion:")
    for i, b in enumerate(b_range):
        gap = mse_gaps[i]
        explanation = "exact codebook (not PD approx.)" if b <= 4 else "PD approx."
        print(f"    b={b}: gap = {gap:.2f}×  [{explanation}]")
    print()
    print("  Answer: The gap at b=1 (1.45×) is NOT an artifact of Panter-Dite.")
    print("  It uses the exact numerically-optimal codebook. The gap is fundamental")
    print("  to the scalar quantization architecture — treating coordinates as")
    print("  independent. Only joint vector quantization could close this gap,")
    print("  at exponential computational cost.")
    print()
    print("  compression implications: b=4 → 4× compression vs FP16,")
    print("  b=3 → 5.3×, b=2 → 8×. The MSE distortion bounds shown above")
    print("  directly determine the recall@k achievable in nearest-neighbor search")
    print("  (Exercise 3): lower distortion → higher recall at each compression level.")
    print()
    print("DONE — distortion bounds analysis complete.")

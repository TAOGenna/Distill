"""
Exercise 3: Two-Stage TurboQuant_prod
======================================

In Exercise 2 you confirmed that TurboQuant_mse has a multiplicative bias of
2/pi ≈ 0.6366 at b=1, shrinking to 0.99 at b=4.

Now you will implement TurboQuant_prod — the two-stage quantizer that
achieves unbiased inner product estimates while using only b bits total:

  Stage 1: Apply TurboQuant_mse with (b-1) bits
           → get MSE reconstruction x̃_mse and residual r = x - x̃_mse
  Stage 2: Apply QJL to the unit-normalized residual r/||r||
           → get 1-bit sign vector for unbiased correction

The inner product estimator is (paper Section 3.2, Algorithm 2):

    <y, x> ≈ <y, x̃_mse>  +  ||r|| · <y, Q_qjl^{-1}(Q_qjl(r/||r||))>

This is unbiased because:
  - <y, x̃_mse> is computed exactly (no estimation error in the first term)
  - E[||r|| · <y, Q_qjl^{-1}(Q_qjl(r/||r||))>] = ||r|| · <y, r/||r||> = <y, r>
  - Sum = <y, x̃_mse> + <y, r> = <y, x>  ✓
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

_sol_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _sol_dir)

from ex01_qjl_1bit_inner_product_quantizer import QJL


# ---------------------------------------------------------------------------
# PROVIDED: Data utilities
# ---------------------------------------------------------------------------

def generate_unit_vectors(n, d, seed=None):
    """Generate n uniformly random unit vectors in R^d."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ---------------------------------------------------------------------------
# SOLUTION: TurboQuantProd class
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """TurboQuant_prod: two-stage unbiased inner product quantizer.

    Implements Algorithm 2 from the TurboQuant paper.

    Parameters
    ----------
    d : int
        Vector dimension.  Must be in {128, 256, 512}.
    b : int
        Total bit-width (≥ 1).
    seed : int
        Random seed.
    """

    def __init__(self, d, b, seed=42):
        self.d = d
        self.b = b
        self.mse_b = b - 1
        if self.mse_b >= 1:
            self.mse_quantizer = TurboQuantMSE(d, self.mse_b, seed=seed)
        else:
            self.mse_quantizer = None
        self.qjl = QJL(d, seed=seed + 1)

    def quantize(self, x):
        """Quantize x using the two-stage TurboQuant_prod procedure.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Unit-norm input vector.

        Returns
        -------
        idx : np.ndarray of int or None
        qjl_bits : np.ndarray, shape (d,)
        gamma : float
        """
        if self.mse_quantizer is not None:
            idx = self.mse_quantizer.quantize(x)
            x_mse = self.mse_quantizer.dequantize(idx)
        else:
            idx = None
            x_mse = np.zeros(self.d)

        r = x - x_mse
        gamma = float(np.linalg.norm(r))

        if gamma > 1e-12:
            qjl_bits = self.qjl.quantize(r / gamma)
        else:
            qjl_bits = np.ones(self.d)

        return idx, qjl_bits, gamma

    def dequantize(self, idx, qjl_bits, gamma):
        """Reconstruct approximate vector from two-stage representation.

        Parameters
        ----------
        idx : np.ndarray of int or None
        qjl_bits : np.ndarray, shape (d,)
        gamma : float

        Returns
        -------
        np.ndarray, shape (d,)
        """
        if idx is not None:
            x_mse = self.mse_quantizer.dequantize(idx)
        else:
            x_mse = np.zeros(self.d)
        x_qjl = gamma * self.qjl.dequantize(qjl_bits)
        return x_mse + x_qjl

    def estimate_inner_product(self, y, idx, qjl_bits, gamma):
        """Estimate <y, x> from quantized representation.

        Parameters
        ----------
        y : np.ndarray, shape (d,)
        idx : np.ndarray of int or None
        qjl_bits : np.ndarray, shape (d,)
        gamma : float

        Returns
        -------
        float
        """
        if idx is not None:
            x_mse = self.mse_quantizer.dequantize(idx)
        else:
            x_mse = np.zeros(self.d)
        ip_mse = float(y @ x_mse)
        ip_qjl = gamma * self.qjl.estimate_inner_product(y, qjl_bits)
        return ip_mse + ip_qjl


# ---------------------------------------------------------------------------
# SOLUTION: Distortion measurement (vectorized for speed)
# ---------------------------------------------------------------------------

def measure_prod_distortion(quantizer, queries, database, n_trials=50):
    """Measure inner product distortion D_prod of TurboQuantProd.

    D_prod = E[|<y, x> - estimated|^2]

    Parameters
    ----------
    quantizer : TurboQuantProd
    queries : np.ndarray, shape (n_q, d)
    database : np.ndarray, shape (n_db, d)
    n_trials : int

    Returns
    -------
    float
        Estimated D_prod.
    """
    d = quantizer.d
    b = quantizer.b
    n_q, n_db = len(queries), len(database)
    true_ips = queries @ database.T  # (n_q, n_db)

    total_sq_error = 0.0
    count = 0

    for trial in range(n_trials):
        fresh_q = TurboQuantProd(d, b, seed=trial * 100)
        for j in range(n_db):
            idx, z, gamma = fresh_q.quantize(database[j])
            # Vectorized: estimate IPs for all queries at once
            if idx is not None:
                x_mse = fresh_q.mse_quantizer.dequantize(idx)
            else:
                x_mse = np.zeros(d)
            ip_mse_all = queries @ x_mse                         # (n_q,)
            # QJL part: scale * (S @ y) for all queries, dotted with z
            scale = np.sqrt(np.pi / 2) / d
            Sy_all = queries @ fresh_q.qjl.S.T                   # (n_q, d)
            ip_qjl_all = gamma * scale * (Sy_all @ z)            # (n_q,)
            estimated_col = ip_mse_all + ip_qjl_all              # (n_q,)
            sq_errors = (estimated_col - true_ips[:, j]) ** 2    # (n_q,)
            total_sq_error += float(np.sum(sq_errors))
            count += n_q

    return total_sq_error / count


# ---------------------------------------------------------------------------
# TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TurboQuant_prod: Two-Stage Unbiased Inner Product Quantizer")
    print("=" * 70)

    d = 128
    D_PROD_THEORY = {1: 1.57 / d, 2: 0.56 / d, 3: 0.18 / d, 4: 0.047 / d}

    # -----------------------------------------------------------------
    # TEST 1: Unbiasedness — average over many seeds to reduce noise
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 1: Unbiasedness  (b=1,2,3,4; 30 pairs, 300 seeds)")
    print("─" * 50)

    # Use small database/queries but many seeds for reliable bias estimate
    n_db_bias, n_q_bias = 30, 10
    database_b = generate_unit_vectors(n_db_bias, d, seed=1)
    queries_b = generate_unit_vectors(n_q_bias, d, seed=2)
    true_ips_b = queries_b @ database_b.T  # (10, 30)

    print()
    print(f"  {'b':>3}  {'mean error':>12}  {'mean |IP|':>12}  {'rel bias':>10}  Status")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")

    for b in [1, 2, 3, 4]:
        n_seeds = 300
        sum_estimates = np.zeros((n_q_bias, n_db_bias))
        for seed in range(n_seeds):
            q_prod = TurboQuantProd(d, b, seed=seed * 7)
            for j in range(n_db_bias):
                idx, z, gamma = q_prod.quantize(database_b[j])
                if idx is not None:
                    x_mse = q_prod.mse_quantizer.dequantize(idx)
                else:
                    x_mse = np.zeros(d)
                ip_mse_all = queries_b @ x_mse
                scale = np.sqrt(np.pi / 2) / d
                ip_qjl_all = gamma * scale * (queries_b @ q_prod.qjl.S.T @ z)
                sum_estimates[:, j] += ip_mse_all + ip_qjl_all
        mean_estimates = sum_estimates / n_seeds
        errors = mean_estimates - true_ips_b
        mean_err = float(np.mean(errors))
        mean_abs_true = float(np.mean(np.abs(true_ips_b)))
        rel_bias = abs(mean_err) / mean_abs_true
        status = "OK" if rel_bias < 0.02 else "CHECK"
        print(f"  {b:>3}  {mean_err:>12.5f}  {mean_abs_true:>12.5f}  {rel_bias:>10.4f}  [{status}]")

    # -----------------------------------------------------------------
    # TEST 2: D_prod distortion at b=1,2,3,4
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 2: D_prod vs theoretical bound  (50 trials, 50 db, 20 queries)")
    print("─" * 50)

    db_small = generate_unit_vectors(50, d, seed=10)
    q_small = generate_unit_vectors(20, d, seed=11)

    print()
    print(f"  {'b':>3}  {'D_prod empirical':>18}  {'D_prod theory':>15}  {'ratio':>8}  Status")
    print(f"  {'─'*3}  {'─'*18}  {'─'*15}  {'─'*8}  {'─'*10}")

    for b in [1, 2, 3, 4]:
        q_prod = TurboQuantProd(d, b, seed=42)
        d_prod = measure_prod_distortion(q_prod, q_small, db_small, n_trials=50)
        theory = D_PROD_THEORY[b]
        ratio = d_prod / theory
        status = "OK" if ratio <= 2.0 else "CHECK"
        print(f"  {b:>3}  {d_prod:>18.6f}  {theory:>15.6f}  {ratio:>8.3f}  [{status}]")

    # -----------------------------------------------------------------
    # TEST 3: Compare TurboQuant_prod vs TurboQuant_mse mean error
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 3: Prod vs MSE — unbiasedness comparison")
    print("─" * 50)

    db_cmp = generate_unit_vectors(100, d, seed=20)
    q_cmp = generate_unit_vectors(30, d, seed=21)
    true_cmp = q_cmp @ db_cmp.T  # (30, 100)

    print()
    print(f"  {'b':>3}  {'prod mean err':>15}  {'mse mean err':>15}  Prod unbiased?")
    print(f"  {'─'*3}  {'─'*15}  {'─'*15}  {'─'*14}")

    for b in [1, 2, 3, 4]:
        # TurboQuant_prod: average over 100 seeds
        sum_prod = np.zeros((30, 100))
        n_prod_seeds = 100
        for seed in range(n_prod_seeds):
            qp = TurboQuantProd(d, b, seed=seed)
            for j in range(100):
                idx, z, gamma = qp.quantize(db_cmp[j])
                if idx is not None:
                    x_mse = qp.mse_quantizer.dequantize(idx)
                else:
                    x_mse = np.zeros(d)
                ip_mse_all = q_cmp @ x_mse
                scale = np.sqrt(np.pi / 2) / d
                ip_qjl_all = gamma * scale * (q_cmp @ qp.qjl.S.T @ z)
                sum_prod[:, j] += ip_mse_all + ip_qjl_all
        prod_mean = sum_prod / n_prod_seeds
        prod_err = float(np.mean(prod_mean - true_cmp))

        # TurboQuant_mse: single pass (deterministic given seed)
        qmse = TurboQuantMSE(d, b, seed=42)
        idx_batch = qmse.quantize_batch(db_cmp)
        X_tilde = qmse.dequantize_batch(idx_batch)
        mse_est = q_cmp @ X_tilde.T
        mse_err = float(np.mean(mse_est - true_cmp))

        prod_ok = "YES" if abs(prod_err) < abs(mse_err) * 0.2 or abs(prod_err) < 1e-4 else "CHECK"
        print(f"  {b:>3}  {prod_err:>15.5f}  {mse_err:>15.5f}  {prod_ok}")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary: D_prod values at d=128")
    print("=" * 70)
    print()
    print(f"  {'b':>3}  {'Empirical D_prod':>18}  {'Theory D_prod':>15}  {'Lower bound':>13}")
    print(f"  {'─'*3}  {'─'*18}  {'─'*15}  {'─'*13}")
    for b in [1, 2, 3, 4]:
        q_prod = TurboQuantProd(d, b, seed=42)
        d_prod = measure_prod_distortion(q_prod, q_small, db_small, n_trials=50)
        theory = D_PROD_THEORY[b]
        lower = 1.0 / (d * 4**b)
        print(f"  {b:>3}  {d_prod:>18.6f}  {theory:>15.6f}  {lower:>13.8f}")
    print()
    print("  TurboQuant_prod: unbiased AND near-optimal inner product distortion.")

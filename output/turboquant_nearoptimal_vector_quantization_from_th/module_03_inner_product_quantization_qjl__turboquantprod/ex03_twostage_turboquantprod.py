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

Your Tasks
----------
1. TurboQuantProd.quantize(self, x)
   - Run TurboQuantMSE, compute residual, store norm, QJL-quantize unit residual
   - (~8-12 lines)

2. TurboQuantProd.dequantize(self, idx, qjl_bits, gamma)
   - Combine MSE reconstruction with QJL residual correction
   - (~4-6 lines)

3. TurboQuantProd.estimate_inner_product(self, y, idx, qjl_bits, gamma)
   - Compute exact MSE IP + scaled QJL IP estimate
   - (~4-6 lines)

4. measure_prod_distortion(quantizer, queries, database, n_trials=50)
   - Compute E[|<y,x> - estimated|^2] averaging over vector pairs and QJL trials
   - (~6-8 lines)
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

# Import QJL from exercise 1 solution
_sol_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_solutions")
sys.path.insert(0, _sol_dir)

try:
    from ex01_qjl_1bit_inner_product_quantizer import QJL
except ImportError:
    from ex01_qjl_1bit_inner_product_quantizer import QJL


# ---------------------------------------------------------------------------
# PROVIDED: Data utilities
# ---------------------------------------------------------------------------

def generate_unit_vectors(n, d, seed=None):
    """Generate n uniformly random unit vectors in R^d.

    Parameters
    ----------
    n : int
    d : int
    seed : int or None

    Returns
    -------
    np.ndarray, shape (n, d), each row has ||row||=1
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ---------------------------------------------------------------------------
# YOUR CODE: TurboQuantProd class
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """TurboQuant_prod: two-stage unbiased inner product quantizer.

    Implements Algorithm 2 from the TurboQuant paper.

    Stage 1: TurboQuant_mse with (b-1) bits → minimizes ||residual||²
    Stage 2: QJL applied to the unit-normalized residual → unbiased correction

    The total bit-width is (b-1) + 1 = b bits per coordinate,
    plus O(1) bits for the residual norm γ (negligible per coordinate).

    Guarantee (Theorem 2):
      E[<y, x_hat>] = <y, x>  (unbiased)
      D_prod ≤ (√3 π² ||y||² / d) · (1/4^b)

    For b=1,2,3,4:  D_prod ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d

    Parameters
    ----------
    d : int
        Vector dimension.  Must be in {128, 256, 512} (precomputed codebooks).
    b : int
        Total bit-width (≥ 1).  Uses (b-1) bits for MSE, 1 bit for QJL.
        At b=1: uses 0-bit MSE (zero vector) + 1-bit QJL → reduces to pure QJL.
    seed : int
        Random seed for both the rotation matrix (MSE) and S (QJL).
    """

    def __init__(self, d, b, seed=42):
        self.d = d
        self.b = b
        # Stage 1: (b-1)-bit MSE quantizer.
        self.mse_b = b - 1
        if self.mse_b >= 1:
            self.mse_quantizer = TurboQuantMSE(d, self.mse_b, seed=seed)
        else:
            self.mse_quantizer = None   # b=1: no MSE stage
        # Stage 2: QJL for the residual.  Use a different seed than MSE.
        self.qjl = QJL(d, seed=seed + 1)

    def quantize(self, x):
        """Quantize x using the two-stage TurboQuant_prod procedure.

        Procedure (Algorithm 2, TurboQuant paper):
          1.  idx = Quant_mse(x)           (b-1 bits; None if b=1)
          2.  x̃_mse = DeQuant_mse(idx)     (zero vector if b=1)
          3.  r = x - x̃_mse               (residual in original space)
          4.  γ = ||r||₂                   (residual norm — stored explicitly)
          5.  If γ > 0: qjl_bits = sign(S · (r/γ))
              Else:     qjl_bits = ones(d)  (degenerate case)

        IMPORTANT: the residual r must be computed in the ORIGINAL
        coordinate system, not the rotated system.  r = x - x̃_mse where
        both x and x̃_mse are in the standard basis.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Unit-norm input vector.

        Returns
        -------
        idx : np.ndarray of int, shape (d,), or None
            MSE quantization indices.  None when b=1 (no MSE stage).
        qjl_bits : np.ndarray, shape (d,)
            Sign vector from QJL applied to the unit residual.
        gamma : float
            L2 norm of the residual r = x - x̃_mse.

        Hints
        -----
        - If self.mse_quantizer is None (b=1): x̃_mse = zeros(d), idx = None
        - Otherwise: idx = self.mse_quantizer.quantize(x)
                     x_mse = self.mse_quantizer.dequantize(idx)
        - r = x - x_mse  (NOT the rotated version)
        - gamma = float(np.linalg.norm(r))
        - if gamma > 1e-12: qjl_bits = self.qjl.quantize(r / gamma)
        - else:             qjl_bits = np.ones(self.d)
        """
        ###########################################################
        # YOUR CODE HERE - 8-12 lines                             #
        #                                                         #
        # Hint: handle the b=1 case (self.mse_quantizer is None)  #
        # Hint: compute residual r = x - x̃_mse in original space  #
        # Hint: normalize residual before QJL quantization         #
        # Hint: store gamma = ||r|| to enable rescaling in decode  #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def dequantize(self, idx, qjl_bits, gamma):
        """Reconstruct an approximate vector from the two-stage representation.

        Combines the MSE reconstruction with the QJL-estimated residual:
          x̃ = x̃_mse  +  γ · Q_qjl^{-1}(qjl_bits)

        Parameters
        ----------
        idx : np.ndarray of int, shape (d,), or None
            MSE quantization indices (None if b=1).
        qjl_bits : np.ndarray, shape (d,)
            QJL sign vector from quantize().
        gamma : float
            Residual norm from quantize().

        Returns
        -------
        np.ndarray, shape (d,)
            Approximate reconstruction x̃ ≈ x.

        Hints
        -----
        - If idx is None: x̃_mse = zeros(d)
        - x̃_qjl = gamma * self.qjl.dequantize(qjl_bits)
        - return x̃_mse + x̃_qjl
        """
        ###########################################################
        # YOUR CODE HERE - 4-6 lines                              #
        #                                                         #
        # Hint: if idx is None: x_mse = np.zeros(self.d)         #
        # Hint: else: x_mse = self.mse_quantizer.dequantize(idx)  #
        # Hint: x_qjl = gamma * self.qjl.dequantize(qjl_bits)    #
        # Hint: return x_mse + x_qjl                             #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def estimate_inner_product(self, y, idx, qjl_bits, gamma):
        """Estimate <y, x> from quantized representation (idx, qjl_bits, gamma).

        Computes:
          <y, x̃_mse>  +  γ · <y, Q_qjl^{-1}(qjl_bits)>

        Parameters
        ----------
        y : np.ndarray, shape (d,)
        idx : np.ndarray of int, shape (d,), or None
        qjl_bits : np.ndarray, shape (d,)
        gamma : float

        Returns
        -------
        float
            Unbiased estimate of <y, x>.

        Hints
        -----
        - ip_mse = y @ x_mse (exact inner product with MSE reconstruction)
        - ip_qjl = gamma * qjl.estimate_inner_product(y, qjl_bits)
        - return float(ip_mse + ip_qjl)
        """
        ###########################################################
        # YOUR CODE HERE - 4-6 lines                              #
        #                                                         #
        # Hint: if idx is None: x_mse = np.zeros(self.d)         #
        # Hint: ip_mse = float(y @ x_mse)                        #
        # Hint: ip_qjl = gamma * self.qjl.estimate_inner_product(y, qjl_bits)
        # Hint: return ip_mse + ip_qjl                           #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################


# ---------------------------------------------------------------------------
# YOUR CODE: Distortion measurement
# ---------------------------------------------------------------------------

def measure_prod_distortion(quantizer, queries, database, n_trials=50):
    """Measure the inner product distortion D_prod of TurboQuantProd.

    D_prod = E[|<y, x> - estimated|²]

    Parameters
    ----------
    quantizer : TurboQuantProd
    queries : np.ndarray, shape (n_q, d)
    database : np.ndarray, shape (n_db, d)
    n_trials : int
        Number of independent TurboQuantProd instances to average over.

    Returns
    -------
    float
        Estimated D_prod = mean squared error over all (query, db, trial) triples.

    Hints
    -----
    - For each trial, create a fresh TurboQuantProd(d, b, seed=trial*100)
    - Quantize each db vector, estimate all IPs against all queries
    - For vectorized estimates: ip_mse_all = queries @ x_mse  (shape n_q)
      and scale * (queries @ qjl.S.T @ z) for QJL part
    - D_prod = total squared error / (n_trials * n_q * n_db)
    """
    d = quantizer.d
    b = quantizer.b
    n_q, n_db = len(queries), len(database)
    true_ips = queries @ database.T   # (n_q, n_db)

    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Hint: total_sq_error = 0.0; count = 0                   #
    # Hint: for trial in range(n_trials):                     #
    #           fresh_q = TurboQuantProd(d, b, seed=trial*100)#
    #           for j in range(n_db):                         #
    #               idx, z, gamma = fresh_q.quantize(database[j])    #
    #               # vectorized over all queries:            #
    #               x_mse = fresh_q.mse_quantizer.dequantize(idx) \  #
    #                        if idx is not None else np.zeros(d)      #
    #               ip_mse = queries @ x_mse                  #
    #               scale = np.sqrt(np.pi/2) / d              #
    #               ip_qjl = gamma * scale * (queries @ fresh_q.qjl.S.T @ z) #
    #               sq_errors = (ip_mse + ip_qjl - true_ips[:, j])**2 #
    #               total_sq_error += float(np.sum(sq_errors)) #
    #               count += n_q                              #
    # Hint: return total_sq_error / count                     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


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
        # TurboQuant_prod: average over 100 seeds (vectorized)
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

        # TurboQuant_mse: single pass
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

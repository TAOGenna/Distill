"""
Exercise 1: QJL — 1-Bit Inner Product Quantizer
=================================================

The Quantized Johnson-Lindenstrauss (QJL) transform is the foundation of
TurboQuant_prod.  It provides **unbiased** inner product estimates from just
1 bit per coordinate, using the arc-cosine kernel identity:

    E[s_i^T y · sign(s_i^T x)] = (2/π) · <x, y>    for s_i ~ N(0, I_d)

The QJL algorithm (from Definition 1 of the TurboQuant paper):
  - Quantization:    Q_qjl(x) := sign(S · x)
  - Dequantization:  Q_qjl^{-1}(z) := sqrt(π/2)/d · S^T · z

The sqrt(π/2) scaling exactly cancels the 2/π factor from the arc-cosine
identity, making the estimator unbiased.

Your Tasks
----------
1. Implement QJL.quantize(x):        z = sign(S · x)            (~3-4 lines)
2. Implement QJL.dequantize(z):      x̃ = sqrt(π/2)/d · S^T · z  (~3-4 lines)
3. Implement QJL.estimate_inner_product(y, z): <y, x̃>            (~3-5 lines)
4. Implement measure_qjl_variance(qjl, x, y, n_trials):          (~6-10 lines)

The __main__ block below verifies:
  - Relative bias < 2%  (prints "unbiased")
  - Empirical variance within 20% of theoretical bound π/(2d) · ||y||²
"""

import numpy as np


# ---------------------------------------------------------------------------
# PROVIDED: Vector generation utilities
# ---------------------------------------------------------------------------

def generate_random_unit_vectors(n, d, seed=None):
    """Generate n uniformly random unit vectors in R^d.

    Samples from an isotropic Gaussian and normalizes each row.

    Parameters
    ----------
    n : int
        Number of vectors to generate.
    d : int
        Dimension of each vector.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, d)
        Matrix where each row has unit L2 norm.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def generate_query_vectors(n_q, d, seed=None):
    """Generate n_q query vectors with random norms in [0.5, 2.0].

    Query vectors do NOT need to be unit-norm (QJL works for any y).

    Parameters
    ----------
    n_q : int
        Number of query vectors.
    d : int
        Dimension.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_q, d)
        Query vectors with varying norms.
    """
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n_q, d))
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = Y / norms
    # Scale norms to [0.5, 2.0] for variety
    scale = 0.5 + 1.5 * rng.random((n_q, 1))
    return Y * scale


def compute_true_inner_products(queries, database):
    """Compute the exact inner products between all query-database pairs.

    Parameters
    ----------
    queries : np.ndarray, shape (n_q, d)
        Query vectors.
    database : np.ndarray, shape (n_db, d)
        Database vectors (should be unit-norm for TurboQuant).

    Returns
    -------
    np.ndarray, shape (n_q, n_db)
        Matrix of true inner products, entry [i, j] = <queries[i], database[j]>.
    """
    return queries @ database.T


# ---------------------------------------------------------------------------
# YOUR CODE: QJL class
# ---------------------------------------------------------------------------

class QJL:
    """Quantized Johnson-Lindenstrauss (QJL) 1-bit inner product quantizer.

    Implements Definition 1 and Lemma 2 from the TurboQuant paper:

      Quantize:    Q_qjl(x) := sign(S · x)
      Dequantize:  Q_qjl^{-1}(z) := sqrt(π/2)/d · S^T · z

    The quantizer is unbiased:
      E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x>

    And has variance bounded by:
      Var(<y, Q_qjl^{-1}(Q_qjl(x))>) ≤ π/(2d) · ||y||²

    Parameters
    ----------
    d : int
        Vector dimension.  S has shape (d, d).
    seed : int or None
        Random seed for generating S.  Fixed seed = fixed quantizer instance.
    """

    def __init__(self, d, seed=None):
        self.d = d
        rng = np.random.default_rng(seed)
        # S is a d×d matrix with i.i.d. N(0,1) entries.
        # IMPORTANT: S is fixed for one QJL instance.  To get independent
        # estimates you must create separate QJL instances (see measure_qjl_variance).
        self.S = rng.standard_normal((d, d))

    def quantize(self, x):
        """Quantize x to a sign vector via the random projection S.

        Computes z = sign(S · x) entry-wise.  Each coordinate of x is
        "voted on" by a random hyperplane through the origin (the i-th row
        of S defines one such hyperplane).

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Input vector.  Should be unit-norm for the unbiasedness guarantee,
            but sign(S·x) is well-defined for any x ≠ 0.

        Returns
        -------
        np.ndarray, shape (d,), dtype float64
            Sign vector z ∈ {-1, +1}^d.
            Convention: sign(0) = +1 (np.sign returns 0 for 0; handle below).

        Hints
        -----
        - Compute S @ x (a length-d vector).
        - Apply np.sign; then ensure no zeros remain (set 0 → +1).
        """
        ###########################################################
        # YOUR CODE HERE - 3-4 lines                              #
        #                                                         #
        # Hint: z = np.sign(self.S @ x)                          #
        # Hint: replace any zeros with +1 to ensure z ∈ {-1,+1}  #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def dequantize(self, z):
        """Reconstruct an approximate vector from the sign representation.

        Computes x̃ = sqrt(π/2)/d · S^T · z.

        The scaling factor sqrt(π/2)/d is the key: it exactly compensates
        for the 2/π factor introduced by the arc-cosine kernel identity,
        making the inner product estimator <y, x̃> unbiased for <y, x>.

        Parameters
        ----------
        z : np.ndarray, shape (d,)
            Sign vector from quantize(), values in {-1, +1}.

        Returns
        -------
        np.ndarray, shape (d,)
            Approximate reconstruction of the original vector x.
            Note: this is NOT a good MSE reconstruction (the quantizer
            was designed for inner products, not L2 fidelity).

        Hints
        -----
        - Scale factor: np.sqrt(np.pi / 2) / self.d
        - S^T · z is self.S.T @ z
        """
        ###########################################################
        # YOUR CODE HERE - 3-4 lines                              #
        #                                                         #
        # Hint: scale = np.sqrt(np.pi / 2) / self.d              #
        # Hint: return scale * (self.S.T @ z)                     #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def estimate_inner_product(self, y, z):
        """Estimate <y, x> from query y and quantized representation z.

        Computes <y, Q_qjl^{-1}(z)> = sqrt(π/2)/d · y^T · S^T · z.

        Equivalently: scale * (S @ y)^T · z.
        This form is efficient for database lookups — precompute (S @ y)
        once per query, then evaluate against many sign vectors z.

        Parameters
        ----------
        y : np.ndarray, shape (d,)
            Query vector.  Does NOT need to be unit-norm.
        z : np.ndarray, shape (d,)
            Sign vector from quantize(), values in {-1, +1}.

        Returns
        -------
        float
            Scalar estimate of <y, x>.

        Hints
        -----
        - You can call self.dequantize(z) and then dot with y, OR
        - Compute it directly as scale * np.dot(self.S @ y, z)
        - Both give the same result (S^T z dotted with y = y^T S^T z = (Sy)^T z)
        """
        ###########################################################
        # YOUR CODE HERE - 3-5 lines                              #
        #                                                         #
        # Hint: x_tilde = self.dequantize(z)                      #
        # Hint: return float(np.dot(y, x_tilde))                  #
        #                                                         #
        # Or equivalently (slightly faster for many z vectors):   #
        # scale = np.sqrt(np.pi / 2) / self.d                     #
        # return float(scale * np.dot(self.S @ y, z))             #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################


# ---------------------------------------------------------------------------
# YOUR CODE: Variance measurement
# ---------------------------------------------------------------------------

def measure_qjl_variance(qjl_seed_base, x, y, n_trials=1000):
    """Measure the empirical variance of the QJL inner product estimator.

    To measure variance, we must run INDEPENDENT trials — each using a
    freshly generated S matrix (new QJL instance).  Using the same S
    repeatedly gives the same estimate, not variance.

    Parameters
    ----------
    qjl_seed_base : int
        Base seed; trial i uses seed (qjl_seed_base + i) for independence.
    x : np.ndarray, shape (d,)
        Unit-norm database vector to quantize.
    y : np.ndarray, shape (d,)
        Query vector.
    n_trials : int
        Number of independent estimates to collect.

    Returns
    -------
    float
        Empirical variance of the inner product estimates across trials.

    Notes
    -----
    Theoretical variance bound: π/(2d) · ||y||²  (Lemma 2 in the paper)
    Empirical variance should be below (and approximately equal to) this bound.

    Hints
    -----
    - Create n_trials fresh QJL instances with distinct seeds
    - For each: quantize x, then call estimate_inner_product(y, z)
    - Return np.var(estimates_array)
    """
    d = len(x)
    ###########################################################
    # YOUR CODE HERE - 6-10 lines                             #
    #                                                         #
    # Hint: estimates = []                                    #
    # Hint: for i in range(n_trials):                         #
    #           qjl = QJL(d, seed=qjl_seed_base + i)         #
    #           z = qjl.quantize(x)                           #
    #           estimates.append(qjl.estimate_inner_product(y, z))  #
    # Hint: return float(np.var(estimates))                   #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# PROVIDED: Formatters
# ---------------------------------------------------------------------------

def print_unbiasedness_test(true_ips, estimated_ips, method_name="QJL"):
    """Print a formatted unbiasedness report.

    Parameters
    ----------
    true_ips : np.ndarray, shape (n,)
        True inner products for each pair.
    estimated_ips : np.ndarray, shape (n,)
        Estimated inner products for each pair (mean over trials).
    method_name : str
        Name to display in the report.
    """
    errors = estimated_ips - true_ips
    abs_true = np.abs(true_ips)
    # Relative bias: mean(errors) / mean(|true|)
    relative_bias = np.abs(np.mean(errors)) / np.mean(abs_true)
    # Correlation between estimate and truth
    corr = np.corrcoef(true_ips, estimated_ips)[0, 1]
    print(f"  {method_name} unbiasedness test:")
    print(f"    Pairs tested:       {len(true_ips)}")
    print(f"    Mean error:         {np.mean(errors):.5f}")
    print(f"    Mean |true IP|:     {np.mean(abs_true):.5f}")
    print(f"    Relative bias:      {relative_bias:.4f}  (threshold: 0.02)")
    print(f"    Correlation r:      {corr:.4f}")
    if relative_bias < 0.02:
        print(f"    → PASSES: estimator is unbiased (relative bias < 2%)")
    else:
        print(f"    → WARNING: relative bias {relative_bias:.4f} exceeds 2% threshold")


def print_variance_test(empirical_var, theoretical_var):
    """Print variance comparison against theoretical bound.

    Parameters
    ----------
    empirical_var : float
        Variance measured from independent trials.
    theoretical_var : float
        Upper bound: π/(2d) · ||y||².
    """
    ratio = empirical_var / theoretical_var if theoretical_var > 0 else float('inf')
    print(f"  QJL variance test:")
    print(f"    Empirical variance:    {empirical_var:.6f}")
    print(f"    Theoretical bound:     {theoretical_var:.6f}  (π/(2d) · ||y||²)")
    print(f"    Ratio emp/theory:      {ratio:.3f}")
    if ratio <= 1.20:
        print(f"    → PASSES: empirical variance within 20% of theoretical bound")
    else:
        print(f"    → WARNING: ratio {ratio:.3f} exceeds 1.20")


# ---------------------------------------------------------------------------
# TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d = 128

    print("=" * 70)
    print(f"QJL: 1-Bit Inner Product Quantizer  (d={d})")
    print("=" * 70)

    # -----------------------------------------------------------------
    # TEST 1: Unbiasedness test over 1000 vector pairs
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 1: Unbiasedness  (1000 pairs, 200 QJL trials each)")
    print("─" * 50)

    n_pairs = 1000
    # Unit-norm database vectors
    X = generate_random_unit_vectors(n_pairs, d, seed=1)
    # Query vectors (not necessarily unit-norm)
    Y = generate_query_vectors(n_pairs, d, seed=2)

    true_ips = np.array([float(Y[i] @ X[i]) for i in range(n_pairs)])

    # For each pair, average 200 independent QJL estimates
    n_trials_bias = 200
    estimated_ips = np.zeros(n_pairs)
    for i in range(n_pairs):
        trials = []
        for t in range(n_trials_bias):
            qjl = QJL(d, seed=i * n_trials_bias + t)
            z = qjl.quantize(X[i])
            trials.append(qjl.estimate_inner_product(Y[i], z))
        estimated_ips[i] = np.mean(trials)

    print_unbiasedness_test(true_ips, estimated_ips)

    # -----------------------------------------------------------------
    # TEST 2: Variance test — one x, one y, 1000 independent QJL instances
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 2: Variance bound  (single pair, 1000 independent QJL trials)")
    print("─" * 50)

    x_test = generate_random_unit_vectors(1, d, seed=99)[0]
    y_test = generate_query_vectors(1, d, seed=100)[0]
    true_ip = float(x_test @ y_test)

    empirical_var = measure_qjl_variance(
        qjl_seed_base=5000, x=x_test, y=y_test, n_trials=1000
    )
    theoretical_var = (np.pi / (2 * d)) * float(np.dot(y_test, y_test))

    print(f"  True inner product:    {true_ip:.5f}")
    print_variance_test(empirical_var, theoretical_var)

    # -----------------------------------------------------------------
    # TEST 3: Consistency — same S, same x gives same z
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 3: Determinism  (same seed → same result)")
    print("─" * 50)

    qjl_a = QJL(d, seed=7)
    qjl_b = QJL(d, seed=7)
    x_det = generate_random_unit_vectors(1, d, seed=42)[0]

    z_a = qjl_a.quantize(x_det)
    z_b = qjl_b.quantize(x_det)
    print(f"  Same seed, same x → identical z: {np.all(z_a == z_b)}")
    unique_vals = set(z_a.tolist())
    print(f"  z values ∈ {{-1, +1}}: {unique_vals == {-1.0, 1.0}}")

    # -----------------------------------------------------------------
    # TEST 4: Scaling check — QJL dequantization scale
    # -----------------------------------------------------------------
    print()
    print("─" * 50)
    print("Test 4: Scaling — compare to manual computation")
    print("─" * 50)

    qjl_check = QJL(d, seed=13)
    x_sc = generate_random_unit_vectors(1, d, seed=55)[0]
    y_sc = generate_query_vectors(1, d, seed=66)[0]

    z_sc = qjl_check.quantize(x_sc)
    ip_via_method = qjl_check.estimate_inner_product(y_sc, z_sc)
    # Manual: scale * (S @ y)^T · z
    scale = np.sqrt(np.pi / 2) / d
    ip_manual = float(scale * (qjl_check.S @ y_sc) @ z_sc)
    print(f"  Via estimate_inner_product(): {ip_via_method:.6f}")
    print(f"  Via manual formula:           {ip_manual:.6f}")
    print(f"  Match: {abs(ip_via_method - ip_manual) < 1e-10}")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    # Compute relative bias from Test 1
    errors = estimated_ips - true_ips
    relative_bias = np.abs(np.mean(errors)) / np.mean(np.abs(true_ips))
    ratio_var = empirical_var / theoretical_var

    print(f"  Relative bias:    {relative_bias:.4f}  (target: < 0.02)")
    print(f"  Variance ratio:   {ratio_var:.3f}   (target: ≤ 1.20)")
    print()
    if relative_bias < 0.02 and ratio_var <= 1.20:
        print("  QJL is unbiased and its variance matches the theoretical bound.")
        print("  The sqrt(π/2)/d scaling in dequantization is exactly right.")
    else:
        print("  Check your implementation — one or more tests did not pass.")

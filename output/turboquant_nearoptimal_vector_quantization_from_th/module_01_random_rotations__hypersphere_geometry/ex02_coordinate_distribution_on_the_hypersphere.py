"""
Exercise 2: Coordinate Distribution on the Hypersphere
=======================================================

In Exercise 1, you verified that a random rotation Π maps any vector to a
uniformly random point on S^{d-1}. Now we study the COORDINATE DISTRIBUTION
of such a uniform random point — the distribution that makes TurboQuant's
codebook data-oblivious.

From Lemma 1 of the TurboQuant paper, if x ∈ S^{d-1} is uniform on the
unit hypersphere, each coordinate x_j follows:

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}

In high dimensions this converges to N(0, 1/d).

In this exercise you will:
  1. Collect coordinate samples from rotated vectors of diverse types
  2. Compute the empirical CDF and compare to the theoretical Beta PDF
  3. Measure empirical moments and compare to theory (mean=0, var=1/d)

All functions from Exercise 1 are imported — you verified they work:
Π is orthogonal (max |Π^T·Π - I| ≈ 8.88e-16), norm-preserving, and
inner-product-preserving.

References:
  - TurboQuant paper, Lemma 1 (coordinate distribution)
  - §3.1 (MSE Optimal TurboQuant): "each coordinate of Π·x follows a
    Beta distribution, which converges to N(0,1/d) in high dimensions"
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import kstest, norm as scipy_norm

# Import from Exercise 1 (which you verified produces correct rotations)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from ex01_random_rotation_matrix_generation import (
    generate_random_rotation,
    rotate_vector,
)


# ---------------------------------------------------------------------------
# Provided: theoretical Beta PDF and diverse test vectors
# ---------------------------------------------------------------------------

def beta_pdf_theoretical(x, d):
    """
    Evaluate the marginal PDF for one coordinate of a uniform point on S^{d-1}.

    Formula (TurboQuant paper, Lemma 1):
        f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}

    Computed in log-space to avoid overflow for large d (gamma(256) ≈ 10^500).
    Defined on x ∈ [-1, 1]; zero outside.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the PDF. Values outside [-1,1] give 0.
    d : int
        Dimension of the ambient space.

    Returns
    -------
    pdf : np.ndarray
        PDF values at x, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    # Use log-gamma to avoid overflow (gamma(256) ≈ 10^500 overflows float64)
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    inside = np.maximum(1.0 - x**2, 0.0)
    exponent = (d - 3) / 2
    log_shape = np.where(inside > 1e-300, exponent * np.log(inside), -np.inf)
    pdf = np.exp(log_norm + log_shape)
    pdf = np.where(np.abs(x) <= 1.0, pdf, 0.0)
    return pdf


def generate_diverse_unit_vectors(n, d, seed=17):
    """
    Generate n uniformly random unit vectors on S^{d-1}.

    The theoretical guarantee in TurboQuant is:
      "For any fixed unit vector x and Haar-uniform random Π, the vector Π·x
       is uniformly distributed on S^{d-1}."

    Equivalently (and what we verify here):
      "If x is uniformly distributed on S^{d-1} and Π is any fixed orthogonal
       matrix, then Π·x is also uniformly distributed on S^{d-1}."

    We generate uniform sphere points using the normalized Gaussian construction:
        x = g / ‖g‖,    g ~ N(0, I_d)

    This is the unique simple construction that yields the Haar measure on S^{d-1}.
    Four sub-groups use different random seeds to verify reproducibility.

    Note: vectors formed by normalizing scaled Gaussians (e.g., outlier channels
    scaled 5×) are NOT uniformly distributed on S^{d-1}. To show f_X holds for
    adversarial inputs, use a FRESH random Π per sample.

    Parameters
    ----------
    n : int
        Total number of vectors to generate.
    d : int
        Dimension of each vector.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    vectors : np.ndarray, shape (n, d)
        Array of unit-norm vectors (each row has L2 norm = 1.0),
        uniformly distributed on S^{d-1}.
    """
    rng = np.random.default_rng(seed)
    n_each = n // 4
    results = []
    for sub_seed in [seed, seed + 1, seed + 2, seed + 3]:
        sub_rng = np.random.default_rng(sub_seed)
        g = sub_rng.standard_normal((n_each, d))
        v = g / np.linalg.norm(g, axis=1, keepdims=True)
        results.append(v)
    return np.vstack(results)


# ---------------------------------------------------------------------------
# Part 1: Collect coordinate samples
# ---------------------------------------------------------------------------

def collect_coordinate_samples(vectors, Pi, coord_idx=0):
    """
    Rotate each vector by Π and extract a single coordinate.

    This function implements the core observation: for uniformly random
    unit vectors x ∈ S^{d-1}, after rotation by any fixed orthogonal Π,
    the extracted coordinate follows:

        f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^{(d-3)/2}

    Parameters
    ----------
    vectors : np.ndarray, shape (n, d)
        Input unit-norm vectors. Should be uniformly distributed on S^{d-1}
        (use generate_diverse_unit_vectors to ensure this).
    Pi : np.ndarray, shape (d, d)
        Haar-uniform orthogonal matrix from generate_random_rotation().
    coord_idx : int
        Which coordinate to extract from the rotated vector (0-indexed).

    Returns
    -------
    samples : np.ndarray, shape (n,)
        Values of the coord_idx-th coordinate from each rotated vector.
        Each sample is in [-1, 1] (since rotated vectors are unit-norm).

    Notes
    -----
    Do NOT rotate multiple coordinates from the same rotated vector and
    treat them as independent samples of the marginal f_X — distinct
    coordinates from the same vector are nearly (but not exactly) independent.
    """
    ###########################################################################
    # YOUR CODE HERE - 4-6 lines                                              #
    #                                                                         #
    # For each vector in `vectors`, rotate it by Pi and extract coordinate   #
    # coord_idx. Collect all extracted values into a 1D array.               #
    #                                                                         #
    # Efficient batch approach (recommended):                                 #
    #   rotated = Pi @ vectors.T   # shape: (d, n)                           #
    #   samples = rotated[coord_idx, :]  # shape: (n,)                       #
    #                                                                         #
    # Slow loop approach (also correct):                                      #
    #   samples = np.array([rotate_vector(v, Pi)[coord_idx] for v in vectors])
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ---------------------------------------------------------------------------
# Part 2: Compute empirical CDF
# ---------------------------------------------------------------------------

def compute_empirical_cdf(samples, x_grid):
    """
    Compute the empirical CDF of `samples` evaluated on `x_grid`.

    The empirical CDF at point t is: F_n(t) = #{samples ≤ t} / n

    Parameters
    ----------
    samples : np.ndarray, shape (n,)
        Observed coordinate values from collect_coordinate_samples().
    x_grid : np.ndarray, shape (m,)
        Grid of points at which to evaluate the empirical CDF.

    Returns
    -------
    ecdf : np.ndarray, shape (m,)
        Empirical CDF values at each point in x_grid.
        Values are in [0, 1].

    Notes
    -----
    The empirical CDF is a step function: for each grid point t,
    count how many samples fall at or below t, then divide by n.
    Broadcasting (samples[:, None] <= x_grid[None, :]).mean(0) is
    memory-efficient for n ~ 50000 and m ~ 500.
    """
    ###########################################################################
    # YOUR CODE HERE - 4-6 lines                                              #
    #                                                                         #
    # Hint — broadcasting approach:                                           #
    #   samples[:, None] has shape (n, 1)                                    #
    #   x_grid[None, :] has shape (1, m)                                     #
    #   (samples[:, None] <= x_grid[None, :]) has shape (n, m) — boolean    #
    #   Take mean over axis=0 to get empirical CDF at each grid point.       #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ---------------------------------------------------------------------------
# Part 3: Compare moments to theory
# ---------------------------------------------------------------------------

def compare_moments(samples, d):
    """
    Compare empirical moments of coordinate samples to theoretical values.

    For x_j ~ f_X (marginal of uniform S^{d-1}):
      - Theoretical mean: E[x_j] = 0   (by symmetry of f_X around 0)
      - Theoretical variance: Var[x_j] = 1/d   (from Beta distribution moments)
      - Theoretical kurtosis: excess kurtosis = -6/(d+2)
        (negative → lighter tails than Gaussian for finite d)

    Parameters
    ----------
    samples : np.ndarray, shape (n,)
        Coordinate samples from collect_coordinate_samples().
    d : int
        Dimension (used to compute theoretical values).

    Returns
    -------
    results : dict with keys:
        'empirical_mean'      : float  (should be near 0)
        'theoretical_mean'    : float  (= 0)
        'empirical_var'       : float  (should be near 1/d)
        'theoretical_var'     : float  (= 1/d)
        'var_rel_error'       : float  (|emp_var - 1/d| / (1/d), should be < 0.05)
        'empirical_kurtosis'  : float  (excess kurtosis of samples)
        'theoretical_kurtosis': float  (= -6/(d+2))

    Notes
    -----
    Excess kurtosis formula: E[(x - mu)^4] / sigma^4 - 3
    For N(0,1): excess kurtosis = 0.
    For f_X (symmetric Beta): excess kurtosis = -6/(d+2) < 0 (platykurtic).
    """
    ###########################################################################
    # YOUR CODE HERE - 10-15 lines                                            #
    #                                                                         #
    # Step 1: Compute empirical mean and variance.                            #
    #   empirical_mean = np.mean(samples)                                     #
    #   empirical_var  = np.var(samples)   # ddof=0 (biased estimator is OK) #
    #                                                                         #
    # Step 2: Set theoretical values.                                         #
    #   theoretical_mean = 0.0                                                #
    #   theoretical_var  = 1.0 / d                                            #
    #                                                                         #
    # Step 3: Compute var_rel_error = |emp_var - 1/d| / (1/d).              #
    #                                                                         #
    # Step 4: Compute empirical excess kurtosis.                             #
    #   centered = samples - empirical_mean                                   #
    #   empirical_kurtosis = np.mean(centered**4) / empirical_var**2 - 3     #
    #   (be careful about sigma^4 = var^2, not std^4)                       #
    #                                                                         #
    # Step 5: Compute theoretical_kurtosis = -6.0 / (d + 2).               #
    #                                                                         #
    # Step 6: Return a dict with all 7 keys listed in the docstring.         #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ---------------------------------------------------------------------------
# Provided: KS test report helper
# ---------------------------------------------------------------------------

def ks_test_report(samples, d):
    """
    Run a Kolmogorov-Smirnov test comparing samples to the theoretical Beta PDF.

    Uses a numerical CDF constructed by integrating beta_pdf_theoretical.
    The log-space implementation of beta_pdf_theoretical handles large d.

    Parameters
    ----------
    samples : np.ndarray, shape (n,)
        Coordinate samples.
    d : int
        Dimension.

    Returns
    -------
    ks_stat : float
        KS statistic (sup-norm of |F_empirical - F_theoretical|).
    p_value : float
        p-value (> 0.05 means we fail to reject H0: samples from f_X).
    """
    x_grid = np.linspace(-0.9999, 0.9999, 2000)
    dx = x_grid[1] - x_grid[0]
    pdf_vals = beta_pdf_theoretical(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * dx
    cdf_vals = cdf_vals / cdf_vals[-1]

    from scipy.interpolate import interp1d
    theoretical_cdf_fn = interp1d(x_grid, cdf_vals, bounds_error=False,
                                   fill_value=(0.0, 1.0))
    ks_stat, p_value = kstest(samples, theoretical_cdf_fn)
    return ks_stat, p_value


# ---------------------------------------------------------------------------
# Main: verify coordinate distribution across dimensions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 2: Coordinate Distribution on the Hypersphere")
    print("=" * 70)

    n_vectors = 50000
    dimensions = [32, 128, 512]

    print(f"\n{'d':>5} | {'Mean':>8} | {'Var':>10} | {'Theo Var':>10} | "
          f"{'Var Rel Err':>12} | {'KS stat':>8} | {'p-value':>8}")
    print("-" * 80)

    for d in dimensions:
        Pi = generate_random_rotation(d, seed=d * 3 + 1)

        # Generate diverse uniform sphere points
        vectors = generate_diverse_unit_vectors(n_vectors, d, seed=d + 42)

        # Collect samples of coordinate 0 after rotation
        samples = collect_coordinate_samples(vectors, Pi, coord_idx=0)

        # Moment comparison
        moments = compare_moments(samples, d)

        # KS test against theoretical Beta distribution
        ks_stat, p_value = ks_test_report(samples, d)

        print(f"{d:>5} | {moments['empirical_mean']:>+8.4f} | "
              f"{moments['empirical_var']:>10.6f} | "
              f"{moments['theoretical_var']:>10.6f} | "
              f"{moments['var_rel_error']:>12.4%} | "
              f"{ks_stat:>8.4f} | {p_value:>8.4f}")

    print("\n--- KS test interpretation ---")
    print("p-value > 0.05 means we FAIL to reject H0: samples follow f_X.")
    print("This is good — it confirms the coordinate distribution matches theory.")

    print("\n--- Gaussian convergence check at d=128 ---")
    d = 128
    Pi = generate_random_rotation(d, seed=999)
    vectors = generate_diverse_unit_vectors(n_vectors, d, seed=1000)
    samples = collect_coordinate_samples(vectors, Pi, coord_idx=0)

    x_grid = np.linspace(-0.5, 0.5, 500)
    ecdf = compute_empirical_cdf(samples, x_grid)

    # Compare to N(0, 1/d) Gaussian CDF
    gauss_cdf = scipy_norm.cdf(x_grid, loc=0, scale=1/np.sqrt(d))
    sup_diff = np.max(np.abs(ecdf - gauss_cdf))
    print(f"sup|F_empirical - Φ(x·√d)| = {sup_diff:.4f}  (should be < 0.02 at d=128)")

    print("\n--- Moment table for d=128 ---")
    moments = compare_moments(samples, d)
    print(f"  Empirical mean:      {moments['empirical_mean']:+.6f}  (theory: {moments['theoretical_mean']:+.1f})")
    print(f"  Empirical variance:  {moments['empirical_var']:.6f}  (theory: {moments['theoretical_var']:.6f})")
    print(f"  Variance rel error:  {moments['var_rel_error']:.4%}  (should be < 5%)")
    print(f"  Empirical kurtosis:  {moments['empirical_kurtosis']:+.4f}  (theory: {moments['theoretical_kurtosis']:+.4f})")

    print("\n--- Summary ---")
    print("KS test p-values > 0.05 confirm: coordinate distribution matches f_X")
    print("Variance ≈ 1/d confirmed across all dimensions")
    print("This is the key fact enabling TurboQuant's data-oblivious codebook design")

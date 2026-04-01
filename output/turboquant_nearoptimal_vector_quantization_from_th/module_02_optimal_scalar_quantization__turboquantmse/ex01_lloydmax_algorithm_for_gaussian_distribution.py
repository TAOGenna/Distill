"""
Exercise 1: Lloyd-Max Algorithm for Gaussian Distribution
==========================================================

The Lloyd-Max algorithm (Lloyd 1982, Max 1960) solves the continuous k-means
problem: given a known probability distribution f_X, find the optimal set of
2^b centroids that minimizes the expected squared quantization error.

    C(f_X, b) = min_{c_1,...,c_{2^b}} sum_i integral_{t_{i-1}}^{t_i} |x - c_i|^2 * f_X(x) dx

where t_i are the Voronoi boundaries (midpoints between adjacent centroids).

In this exercise you work with the Gaussian distribution N(0, sigma^2). In
Exercise 2 you will adapt the same algorithm to the Beta distribution that
actually arises in TurboQuant after random rotation.

The algorithm alternates two conditions until convergence:
  1. Boundaries: t_i = (c_i + c_{i+1}) / 2  (Voronoi / midpoint condition)
  2. Centroids: c_i = E[X | t_{i-1} ≤ X ≤ t_i]  (centroid / conditional mean)

Known optimal results for N(0,1):
  b=1: codebook = {-0.7979, +0.7979}
  b=2: codebook = {-1.5104, -0.4528, +0.4528, +1.5104}
"""

import numpy as np
from scipy import integrate
from scipy.stats import norm


# ---------------------------------------------------------------------------
# PROVIDED HELPERS
# ---------------------------------------------------------------------------

def gaussian_pdf(x, sigma=1.0):
    """Probability density of N(0, sigma^2) at x.

    Parameters
    ----------
    x : float or np.ndarray
        Point(s) at which to evaluate the PDF.
    sigma : float
        Standard deviation of the Gaussian. Default 1.0.

    Returns
    -------
    float or np.ndarray
        PDF value(s) at x.
    """
    return norm.pdf(x, loc=0, scale=sigma)


def gaussian_conditional_expectation(a, b_bound, sigma=1.0):
    """Conditional mean E[X | a <= X <= b] for X ~ N(0, sigma^2).

    Uses the closed-form formula for truncated Gaussian moments to avoid
    numerical integration for this specific case.

    Parameters
    ----------
    a : float
        Lower bound (may be -np.inf).
    b_bound : float
        Upper bound (may be +np.inf).
    sigma : float
        Standard deviation. Default 1.0.

    Returns
    -------
    float
        Conditional mean E[X | a <= X <= b].
    """
    # CDF and PDF values at bounds
    phi_a = norm.pdf(a / sigma)   # standard normal PDF at a/sigma
    phi_b = norm.pdf(b_bound / sigma)
    Phi_a = norm.cdf(a / sigma)
    Phi_b = norm.cdf(b_bound / sigma)
    denom = Phi_b - Phi_a
    if denom < 1e-300:
        # Extremely small cell — return midpoint as fallback
        return (a + b_bound) / 2.0
    return sigma * (phi_a - phi_b) / denom


def initial_centroids_uniform(b, support_range=(-4.0, 4.0)):
    """Seed 2^b centroids uniformly in [support_range[0], support_range[1]].

    Parameters
    ----------
    b : int
        Bit-width. Will create 2^b centroids.
    support_range : tuple of float
        (low, high) range for initial placement.

    Returns
    -------
    np.ndarray, shape (2^b,)
        Sorted initial centroids.
    """
    n = 2 ** b
    lo, hi = support_range
    # Uniformly spaced, shifted to avoid boundary effects
    return np.linspace(lo + (hi - lo) / (2 * n),
                       hi - (hi - lo) / (2 * n), n)


def convergence_check(old_centroids, new_centroids, tol=1e-10):
    """Check whether centroids have converged.

    Parameters
    ----------
    old_centroids : np.ndarray
        Centroids from the previous iteration.
    new_centroids : np.ndarray
        Centroids from the current iteration.
    tol : float
        Convergence tolerance on max absolute change.

    Returns
    -------
    bool
        True if max |new - old| < tol.
    """
    return np.max(np.abs(new_centroids - old_centroids)) < tol


# ---------------------------------------------------------------------------
# YOUR CODE — implement the four functions below
# ---------------------------------------------------------------------------

def compute_boundaries(centroids):
    """Compute Voronoi cell boundaries for a sorted centroid array.

    The boundaries are the midpoints between consecutive centroids, with
    -infinity and +infinity as the outermost boundaries:

        t_0 = -inf
        t_i = (c_i + c_{i+1}) / 2   for i = 1, ..., 2^b - 1
        t_{2^b} = +inf

    Parameters
    ----------
    centroids : np.ndarray, shape (k,)
        Sorted quantization centroids.

    Returns
    -------
    np.ndarray, shape (k+1,)
        Boundary array where boundaries[0] = -inf and boundaries[-1] = +inf.

    Examples
    --------
    >>> compute_boundaries(np.array([-1.0, 1.0]))
    array([-inf,  0. ,  inf])
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                              #
    #                                                         #
    # Hint: use np.concatenate to prepend -np.inf and append  #
    # np.inf to the midpoints array.  The midpoints are just  #
    # (centroids[:-1] + centroids[1:]) / 2.                   #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def update_centroids(boundaries, pdf_func, b):
    """Update centroids to conditional means within each Voronoi cell.

    For each cell [boundaries[i], boundaries[i+1]], compute:
        c_i = E[X | boundaries[i] <= X <= boundaries[i+1]]
            = integral_{boundaries[i]}^{boundaries[i+1]} x * f_X(x) dx
              / integral_{boundaries[i]}^{boundaries[i+1]} f_X(x) dx

    Uses scipy.integrate.quad which handles infinite limits natively.

    Parameters
    ----------
    boundaries : np.ndarray, shape (2^b + 1,)
        Cell boundaries including ±inf as the outer boundaries.
    pdf_func : callable
        Function pdf_func(x) -> float, the probability density.
    b : int
        Bit-width; there are 2^b cells.

    Returns
    -------
    np.ndarray, shape (2^b,)
        Updated centroids, one per cell.

    Notes
    -----
    If a cell has negligible probability mass (denominator < 1e-300),
    use the midpoint of the cell boundaries as a fallback to avoid
    division by zero.
    """
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint: loop over range(2**b). For cell i, integrate      #
    # x*pdf_func(x) and pdf_func(x) over                      #
    # [boundaries[i], boundaries[i+1]] using                  #
    # scipy.integrate.quad.  Return numerator/denominator.    #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def lloyd_max_iterate(b, pdf_func, initial_centroids, max_iter=200, tol=1e-10):
    """Run Lloyd-Max alternating optimization until convergence.

    Alternates between:
      (a) compute_boundaries — recompute cell boundaries as midpoints
      (b) update_centroids  — recompute centroids as conditional means

    until either centroids stop changing (convergence_check passes) or
    max_iter is reached.

    Parameters
    ----------
    b : int
        Bit-width; the codebook will have 2^b entries.
    pdf_func : callable
        Probability density function f_X(x) -> float.
    initial_centroids : np.ndarray, shape (2^b,)
        Starting centroid positions.
    max_iter : int
        Maximum number of Lloyd-Max iterations.
    tol : float
        Convergence tolerance passed to convergence_check.

    Returns
    -------
    centroids : np.ndarray, shape (2^b,)
        Converged centroid positions.
    boundaries : np.ndarray, shape (2^b + 1,)
        Final cell boundaries corresponding to returned centroids.
    n_iters : int
        Number of iterations performed.
    """
    ###########################################################
    # YOUR CODE HERE - 10-15 lines                            #
    #                                                         #
    # Hint:                                                    #
    # 1. Start with centroids = initial_centroids.copy()      #
    # 2. In each iteration, call compute_boundaries then      #
    #    update_centroids to get new_centroids.               #
    # 3. Check convergence with convergence_check.            #
    # 4. Update centroids <- new_centroids and continue.      #
    # 5. After the loop, compute final boundaries and return  #
    #    (centroids, boundaries, iteration_count).            #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_mse_cost(centroids, boundaries, pdf_func):
    """Compute the total MSE cost of the given quantizer under pdf_func.

    MSE = sum_i integral_{boundaries[i]}^{boundaries[i+1]} |x - c_i|^2 * f_X(x) dx

    This is C(f_X, b), the per-coordinate quantization cost.

    Parameters
    ----------
    centroids : np.ndarray, shape (k,)
        Centroid values.
    boundaries : np.ndarray, shape (k+1,)
        Cell boundaries (first = -inf, last = +inf).
    pdf_func : callable
        Probability density function f_X(x) -> float.

    Returns
    -------
    float
        Total MSE cost summed over all cells.
    """
    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Hint: loop over cells. For cell i with centroid c_i,    #
    # integrate  (x - c_i)**2 * pdf_func(x)  from            #
    # boundaries[i] to boundaries[i+1] using quad.           #
    # Sum up all cell contributions.                          #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# TEST HARNESS — do not modify below this line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sigma = 1.0
    pdf = lambda x: gaussian_pdf(x, sigma)

    # Known optimal codebooks for N(0,1)
    KNOWN_CODEBOOKS = {
        1: np.array([-0.7979, 0.7979]),
        2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
        3: np.array([-2.1520, -1.3439, -0.6568, -0.0000,
                      0.6568,  1.3439,  2.1520]),   # 3-bit has 8 centroids
        4: np.array([-2.7326, -2.0690, -1.5105, -1.0111,
                     -0.5224, -0.0685,  0.3995,  0.8723,
                      1.0111,  1.5105,  2.0690,  2.7326]),  # approx
    }

    # Correct 3-bit known codebook (8 centroids)
    KNOWN_3BIT = np.array([-2.1520, -1.3439, -0.6568, -0.0000,
                            0.0000,  0.6568,  1.3439,  2.1520])
    # For a symmetric distribution, b=3 has 8 centroids symmetric about 0
    KNOWN_CODEBOOKS[3] = np.array([-2.1520, -1.3439, -0.6568, -0.2138,
                                    0.2138,  0.6568,  1.3439,  2.1520])

    print("=" * 65)
    print("Lloyd-Max Algorithm — N(0,1) Codebook Verification")
    print("=" * 65)

    all_passed = True

    for b in [1, 2, 3, 4]:
        init_c = initial_centroids_uniform(b, support_range=(-4.0, 4.0))
        centroids, boundaries, n_iters = lloyd_max_iterate(
            b, pdf, init_c, max_iter=300, tol=1e-10
        )
        mse = compute_mse_cost(centroids, boundaries, pdf)

        print(f"\nb={b} ({2**b} centroids), converged in {n_iters} iterations")
        print(f"  codebook = {np.array2string(centroids, precision=4, separator=', ')}")
        print(f"  MSE cost = {mse:.6f}")

        # Verify b=1 matches known optimal values
        if b == 1:
            expected = np.array([-0.7979, 0.7979])
            max_err = np.max(np.abs(centroids - expected))
            status = "PASS" if max_err < 1e-3 else "FAIL"
            print(f"  vs known {np.array2string(expected, precision=4)}: "
                  f"max_err={max_err:.2e} [{status}]")
            if status == "FAIL":
                all_passed = False

        if b == 2:
            expected = np.array([-1.5104, -0.4528, 0.4528, 1.5104])
            max_err = np.max(np.abs(centroids - expected))
            status = "PASS" if max_err < 1e-3 else "FAIL"
            print(f"  vs known {np.array2string(expected, precision=4)}: "
                  f"max_err={max_err:.2e} [{status}]")
            if status == "FAIL":
                all_passed = False

    print("\n" + "=" * 65)
    print("MSE vs Theoretical Rate (each bit should cut MSE by ~4x)")
    print("=" * 65)
    print(f"{'b':>3} {'MSE':>12} {'Ratio to b-1':>16}")
    prev_mse = None
    for b in [1, 2, 3, 4]:
        init_c = initial_centroids_uniform(b, support_range=(-4.0, 4.0))
        centroids, boundaries, _ = lloyd_max_iterate(b, pdf, init_c)
        mse = compute_mse_cost(centroids, boundaries, pdf)
        ratio_str = f"{prev_mse / mse:.2f}x" if prev_mse else "  --"
        print(f"  {b:1d}   {mse:12.6f}   {ratio_str:>12}")
        prev_mse = mse

    print()
    if all_passed:
        print("All verification checks PASSED.")
    else:
        print("Some checks FAILED — review your implementation.")
        sys.exit(1)

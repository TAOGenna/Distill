"""
Exercise 2: Lloyd-Max for the Beta Distribution
================================================

In Exercise 1 you implemented Lloyd-Max for the Gaussian distribution and
confirmed that b=1 gives codebook {±0.7979} and b=2 gives
{±0.4528, ±1.5104}.

Now adapt the same algorithm to the *actual* distribution that arises inside
TurboQuant after random rotation.  If x is a uniformly random unit vector in
R^d, each coordinate follows the Beta-like distribution (Lemma 1, TurboQuant):

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)   for x ∈ [-1, 1]

This distribution is symmetric around 0, supported on [-1, 1], and converges
to N(0, 1/d) as d → ∞.

Key insight from the paper:
  The optimal codebooks for the Beta distribution are the Gaussian codebooks
  scaled by 1/√d.  Verify: at d=128, multiply centroids by √128 ≈ 11.31 and
  they should match the N(0,1) codebooks from Exercise 1.

Per-coordinate MSE targets (from paper, Theorem 1):
    C(f_X, 1) ≈ 0.36 / d
    C(f_X, 2) ≈ 0.117 / d
    C(f_X, 3) ≈ 0.030 / d
    C(f_X, 4) ≈ 0.009 / d

So for d=128: C(f_X, 1) ≈ 0.00281, etc.
"""

import numpy as np
from scipy import integrate, special


# ---------------------------------------------------------------------------
# PROVIDED: Lloyd-Max core (copied from Exercise 1 — working implementation)
# ---------------------------------------------------------------------------

def compute_boundaries(centroids):
    """Voronoi boundaries: midpoints of consecutive centroids + ±inf.

    Parameters
    ----------
    centroids : np.ndarray, shape (k,)
        Sorted quantization centroids.

    Returns
    -------
    np.ndarray, shape (k+1,)
        boundaries[0] = -inf, boundaries[-1] = +inf.
    """
    midpoints = (centroids[:-1] + centroids[1:]) / 2.0
    return np.concatenate([[-np.inf], midpoints, [np.inf]])


def lloyd_max_iterate(b, pdf_func, initial_centroids, update_fn,
                      max_iter=300, tol=1e-10):
    """Run Lloyd-Max alternating optimization until convergence.

    Generalized version that accepts a custom update function so it can
    work with any distribution (Gaussian, Beta, etc.).

    Parameters
    ----------
    b : int
        Bit-width; codebook will have 2^b entries.
    pdf_func : callable
        Probability density f_X(x) -> float.
    initial_centroids : np.ndarray, shape (2^b,)
        Starting centroid positions.
    update_fn : callable
        update_fn(boundaries, b) -> np.ndarray, shape (2^b,).
        Recomputes centroids given current boundaries.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    centroids : np.ndarray, shape (2^b,)
    boundaries : np.ndarray, shape (2^b + 1,)
    n_iters : int
    """
    centroids = initial_centroids.copy()
    boundaries = compute_boundaries(centroids)

    for iteration in range(1, max_iter + 1):
        new_centroids = update_fn(boundaries, b)
        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            boundaries = compute_boundaries(centroids)
            return centroids, boundaries, iteration
        centroids = new_centroids
        boundaries = compute_boundaries(centroids)

    return centroids, boundaries, max_iter


# ---------------------------------------------------------------------------
# PROVIDED: Beta distribution helpers
# ---------------------------------------------------------------------------

def beta_pdf(x, d):
    """Coordinate distribution of a uniform random point on S^{d-1}.

    From Lemma 1 of TurboQuant:

        f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)

    Defined only on [-1, 1]; returns 0 elsewhere.

    Parameters
    ----------
    x : float
        Point at which to evaluate the density.
    d : int
        Dimension of the ambient space.

    Returns
    -------
    float
        Density value at x; 0 if |x| > 1.
    """
    if np.abs(x) >= 1.0:
        return 0.0
    norm_const = special.gamma(d / 2) / (np.sqrt(np.pi) * special.gamma((d - 1) / 2))
    return norm_const * (1.0 - x ** 2) ** ((d - 3) / 2)


def initial_centroids_symmetric(b, d):
    """Seed 2^b centroids symmetrically within the bulk of f_X.

    Centers initial guess at ±k/(sqrt(d)) for k = 0.5, 1.5, ...,
    which covers the 1-sigma range of the limiting Gaussian N(0,1/d).

    Parameters
    ----------
    b : int
        Bit-width.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray, shape (2^b,)
        Sorted initial centroids.
    """
    n = 2 ** b
    sigma = 1.0 / np.sqrt(d)
    # Uniformly spaced in [-2*sigma, +2*sigma]
    lo, hi = -2.5 * sigma, 2.5 * sigma
    return np.linspace(lo + (hi - lo) / (2 * n),
                       hi - (hi - lo) / (2 * n), n)


# ---------------------------------------------------------------------------
# YOUR CODE — implement the four functions below
# ---------------------------------------------------------------------------

def beta_conditional_expectation(a, b_bound, d):
    """Compute E[X | a <= X <= b] for X ~ f_X (Beta / hypersphere dist.).

    Uses numerical integration since there is no closed form for the
    truncated Beta distribution on [-1, 1].

    Parameters
    ----------
    a : float
        Lower integration limit.  Will be clipped to [-1, 1].
    b_bound : float
        Upper integration limit.  Will be clipped to [-1, 1].
    d : int
        Dimension; determines the shape of f_X.

    Returns
    -------
    float
        Conditional mean E[X | a <= X <= b_bound].

    Notes
    -----
    - Clip a and b_bound to [-1, 1] because f_X is zero outside this range.
    - If the cell has negligible probability mass, return the midpoint.
    """
    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Hint: clip a and b_bound to [-1, 1]. Integrate          #
    # x*beta_pdf(x, d) and beta_pdf(x, d) over [a, b_bound]. #
    # Return numerator / denominator. Guard against near-zero #
    # denominator with a midpoint fallback.                   #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def update_centroids_beta(boundaries, d, n_buckets):
    """Update centroids to conditional means under the Beta distribution.

    For each cell [boundaries[i], boundaries[i+1]], compute the conditional
    mean of X ~ f_X (hypersphere coordinate distribution) by calling
    beta_conditional_expectation.

    Parameters
    ----------
    boundaries : np.ndarray, shape (n_buckets + 1,)
        Cell boundaries (may include ±inf; clips to ±1 internally).
    d : int
        Dimension; controls the Beta distribution shape.
    n_buckets : int
        Number of cells (= 2^b).

    Returns
    -------
    np.ndarray, shape (n_buckets,)
        Updated centroids.
    """
    ###########################################################
    # YOUR CODE HERE - 6-10 lines                             #
    #                                                         #
    # Hint: loop over range(n_buckets). For each cell i,      #
    # call beta_conditional_expectation(boundaries[i],        #
    # boundaries[i+1], d). Collect results in an array.       #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_per_coordinate_mse(centroids, boundaries, d):
    """Compute C(f_X, b) = per-coordinate MSE cost for the Beta distribution.

    C(f_X, b) = sum_i integral_{t_{i-1}}^{t_i} |x - c_i|^2 * f_X(x) dx

    where integration is clipped to [-1, 1] (support of f_X).

    Parameters
    ----------
    centroids : np.ndarray, shape (k,)
        Codebook centroid values.
    boundaries : np.ndarray, shape (k+1,)
        Cell boundaries.
    d : int
        Dimension.

    Returns
    -------
    float
        Per-coordinate MSE cost C(f_X, b).
    """
    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Hint: for each cell i, integrate (x - c_i)^2 *         #
    # beta_pdf(x, d) from max(boundaries[i], -1) to          #
    # min(boundaries[i+1], 1).  Sum up all contributions.    #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def scale_codebook_to_gaussian(centroids, d):
    """Scale the Beta codebook by sqrt(d) to compare with N(0,1) codebook.

    Since f_X converges to N(0, 1/d), the optimal codebook for f_X is
    approximately the N(0,1) codebook divided by sqrt(d).  Multiplying
    back by sqrt(d) should recover the N(0,1) codebook.

    Parameters
    ----------
    centroids : np.ndarray, shape (k,)
        Centroids for the Beta distribution at dimension d.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray, shape (k,)
        Centroids scaled by sqrt(d).
    """
    ###########################################################
    # YOUR CODE HERE - 2-3 lines                              #
    #                                                         #
    # Hint: multiply centroids by np.sqrt(d).                 #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# TEST HARNESS — do not modify below this line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Gaussian reference codebooks (from Exercise 1):
    GAUSSIAN_REF = {
        1: np.array([-0.7979, 0.7979]),
        2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    }

    # TurboQuant paper target per-coordinate MSE values
    TARGET_MSE = {1: 0.36, 2: 0.117, 3: 0.030, 4: 0.009}

    d = 128

    print("=" * 65)
    print(f"Lloyd-Max for Beta Distribution (d={d})")
    print("=" * 65)

    codebooks = {}
    all_passed = True

    for b in [1, 2, 3, 4]:
        pdf_func = lambda x, dim=d: beta_pdf(x, dim)
        init_c = initial_centroids_symmetric(b, d)
        update_fn = lambda bounds, bits, dim=d: update_centroids_beta(
            bounds, dim, 2 ** bits
        )
        centroids, boundaries, n_iters = lloyd_max_iterate(
            b, pdf_func, init_c, update_fn, max_iter=400, tol=1e-10
        )
        per_coord_mse = compute_per_coordinate_mse(centroids, boundaries, d)
        total_mse = d * per_coord_mse
        scaled = scale_codebook_to_gaussian(centroids, d)

        codebooks[b] = centroids

        print(f"\nb={b} ({2**b} centroids), converged in {n_iters} iters")
        print(f"  codebook (raw) = "
              f"{np.array2string(centroids, precision=4, separator=', ')}")
        print(f"  codebook (x√{d}) = "
              f"{np.array2string(scaled, precision=4, separator=', ')}")
        print(f"  per-coordinate MSE = {per_coord_mse:.6f}   (target {TARGET_MSE[b]/d:.6f})")
        print(f"  total MSE d*C = {total_mse:.4f}   (target {TARGET_MSE[b]:.4f})")

        # Check total MSE is within 5% of target
        target = TARGET_MSE[b]
        rel_err = abs(total_mse - target) / target
        status = "PASS" if rel_err < 0.05 else "FAIL"
        print(f"  relative error vs target: {rel_err*100:.1f}% [{status}]")
        if status == "FAIL":
            all_passed = False

        # Check scaled codebook matches Gaussian reference for b=1,2
        if b in GAUSSIAN_REF:
            max_err = np.max(np.abs(scaled - GAUSSIAN_REF[b]))
            g_status = "PASS" if max_err < 0.05 else "FAIL"
            print(f"  scaled vs Gaussian ref: max_err={max_err:.4f} [{g_status}]")
            if g_status == "FAIL":
                all_passed = False

    print("\n" + "=" * 65)
    print("Per-coordinate MSE Summary (should match 0.36/d, 0.117/d, ...)")
    print("=" * 65)
    print(f"{'b':>3}  {'C(f_X,b)':>12}  {'d*C':>10}  {'target':>10}")
    for b in [1, 2, 3, 4]:
        pdf_func = lambda x, dim=d: beta_pdf(x, dim)
        init_c = initial_centroids_symmetric(b, d)
        update_fn = lambda bounds, bits, dim=d: update_centroids_beta(
            bounds, dim, 2 ** bits
        )
        centroids, boundaries, _ = lloyd_max_iterate(
            b, pdf_func, init_c, update_fn
        )
        c_cost = compute_per_coordinate_mse(centroids, boundaries, d)
        print(f"  {b:1d}  {c_cost:12.6f}  {d*c_cost:10.4f}  "
              f"{TARGET_MSE[b]:10.4f}")

    print()
    if all_passed:
        print("All verification checks PASSED.")
    else:
        print("Some checks FAILED — review your implementation.")
        sys.exit(1)

"""
Exercise 1: Shannon Lower Bound Computation
============================================

We have built TurboQuant from scratch across Modules 0-3 and measured its
empirical distortion:

    b=1: D_mse ≈ 0.363    b=2: D_mse ≈ 0.117
    b=3: D_mse ≈ 0.034    b=4: D_mse ≈ 0.009

But how good can *any* quantizer be?  In this exercise you will derive the
information-theoretic lower bound from first principles using two tools:

  1. Shannon's Rate-Distortion Lower Bound (SLB):
       D(B) ≥ (d / 2πe) · 2^{(2/d)(h(X) - B)}
     where h(X) = log₂(A_d) is the entropy of the uniform distribution
     on S^{d-1}, and B = b·d is the total bit budget.

  2. Stirling's Approximation:
     A_d ≈ (2πe/d)^{d/2} · √(2π/d)  ⟹  A_d^{2/d} → 2πe/d
     This simplifies the SLB to D(B) ≥ 2^{-2b} = 1/4^b.

After Yao's minimax principle (which the README explains), this lower bound
applies to ALL quantizers — deterministic, randomized, data-dependent or not.

Your Tasks
----------
1. compute_sphere_entropy(d)       — h(X) = log₂(A_d) using log-space arithmetic
2. compute_exact_slb(d, b)        — exact SLB formula  (~4-6 lines)
3. compute_simplified_slb(b)      — simplified bound 1/4^b  (~2 lines)
4. verify_stirling_approximation(d) — check A_d^{2/d} ≈ 2πe/d  (~5-8 lines)

Key Insight
-----------
The lower bound D_mse ≥ 1/4^b holds for ANY quantization algorithm — it is
a consequence of Shannon's source coding theorem applied to the geometry of
the hypersphere.  TurboQuant achieves ≤ 1.45 × this bound at b=1, meaning
it is within 45% of the information-theoretic optimum.
"""

import numpy as np
import scipy.special as special


# ---------------------------------------------------------------------------
# PROVIDED: Sphere surface area computation
# ---------------------------------------------------------------------------

def compute_sphere_surface_area(d):
    """Compute the surface area of S^{d-1} (the unit sphere in R^d).

    Uses the formula A_d = 2 π^{d/2} / Γ(d/2).
    Works in log-space to handle large d without overflow.

    Parameters
    ----------
    d : int
        Ambient dimension.  The sphere is (d-1)-dimensional.

    Returns
    -------
    float
        A_d = 2 π^{d/2} / Γ(d/2).

    Examples
    --------
    >>> compute_sphere_surface_area(2)  # circumference of unit circle = 2π
    6.283185307179586
    >>> compute_sphere_surface_area(3)  # area of unit sphere in R^3 = 4π
    12.566370614359172
    """
    log_A_d = np.log(2) + (d / 2) * np.log(np.pi) - special.gammaln(d / 2)
    return float(np.exp(log_A_d))


def print_bound_table(d_values, b_range, exact_fn, simplified_fn):
    """Print a formatted comparison of exact vs simplified lower bounds.

    Parameters
    ----------
    d_values : list of int
        Dimensions to show.
    b_range : range
        Bit-widths to show.
    exact_fn : callable
        exact_fn(d, b) → float: exact SLB value.
    simplified_fn : callable
        simplified_fn(b) → float: simplified 1/4^b value.
    """
    print()
    print(f"{'b':>4}  {'simplified 1/4^b':>18}", end="")
    for d in d_values:
        print(f"  {'exact d='+str(d):>16}", end="")
    print()
    print("-" * (4 + 2 + 18 + len(d_values) * 18))

    for b in b_range:
        simp = simplified_fn(b)
        print(f"  {b:2d}  {simp:18.6f}", end="")
        for d in d_values:
            exact = exact_fn(d, b)
            print(f"  {exact:16.6f}", end="")
        print()


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION — complete the four functions below
# ---------------------------------------------------------------------------

def compute_sphere_entropy(d):
    """Compute the differential entropy h(X) = log₂(A_d) of the uniform
    distribution on S^{d-1}.

    The entropy equals the log (base 2) of the surface area because the
    distribution is uniform over the sphere.

    Parameters
    ----------
    d : int
        Ambient dimension.

    Returns
    -------
    float
        h(X) in bits (base-2 logarithm of the sphere's surface area).

    Notes
    -----
    Do NOT call compute_sphere_surface_area and then take log2 — that would
    first exponentiate and then take the log, losing precision for large d.
    Instead, compute log A_d directly (as done inside compute_sphere_surface_area)
    and divide by log(2) to convert from nats to bits.

    Example
    -------
    >>> h = compute_sphere_entropy(128)
    >>> # Should be log2 of a very large number ≈ several hundred bits
    >>> h > 100
    True
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                             #
    #                                                         #
    # Hint: compute log_A_d in nats (same formula as in     #
    #   compute_sphere_surface_area), then divide by log(2)  #
    #   to convert to bits.  Do NOT call exp() — stay in     #
    #   log space until the final result.                     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_exact_slb(d, b):
    """Compute the exact Shannon Lower Bound for uniform-hypersphere source.

    The SLB with B = b*d total bits:
        D(B) ≥ (d / (2πe)) · A_d^{2/d} · (1/4)^b

    Equivalently in log-space:
        log D = log(d/(2πe)) + (2/d) · log(A_d) - 2b · log(2)

    Parameters
    ----------
    d : int
        Vector dimension.
    b : int
        Bits per coordinate.

    Returns
    -------
    float
        Lower bound on D_mse for any b-bit quantizer.

    Notes
    -----
    Work entirely in log-space to avoid overflow for large d.
    log(A_d) is available via the formula in compute_sphere_surface_area.
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                             #
    #                                                         #
    # Step-by-step:                                           #
    #   1. Compute log_A_d  (same 1-line formula as before)  #
    #   2. Compute log of prefactor: log(d) - log(2πe)       #
    #   3. Compute log of (A_d^{2/d}): (2/d) * log_A_d       #
    #   4. Compute log of (1/4)^b: -2b * log(2)              #
    #   5. Sum and exponentiate                               #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_simplified_slb(b):
    """Compute the simplified Shannon Lower Bound 1/4^b = 2^{-2b}.

    This is the result after applying Stirling's approximation to the
    exact SLB.  It holds for large d and is tight for d ≥ 32.

    Parameters
    ----------
    b : int or float
        Bits per coordinate.

    Returns
    -------
    float
        Lower bound 1/4^b = 2^{-2b}.

    Notes
    -----
    This should be a single line of code.  Use the exact formula 4.0**(-b)
    or equivalently 2.0**(-2*b).  Both give the same result.
    """
    ###########################################################
    # YOUR CODE HERE - 2 lines                               #
    #                                                         #
    # Hint: 1/4^b = 4^{-b} = 2^{-2b}                        #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def verify_stirling_approximation(d):
    """Verify that A_d^{2/d} ≈ 2πe/d (the Stirling approximation).

    The exact SLB simplifies to 1/4^b only if A_d^{2/d} ≈ 2πe/d.
    This function computes both sides and reports the ratio and relative error.

    Parameters
    ----------
    d : int
        Dimension to check.

    Returns
    -------
    dict with keys:
        "A_d_power"    : float — computed A_d^{2/d}
        "stirling"     : float — the approximation 2πe/d
        "ratio"        : float — A_d^{2/d} / (2πe/d), should be ≈ 1.0
        "rel_error"    : float — |ratio - 1|, should be < 0.05 for d ≥ 32

    Notes
    -----
    A_d^{2/d} = exp((2/d) * log_A_d).  Compute log_A_d first, then
    compute (2/d)*log_A_d, then exponentiate.
    """
    ###########################################################
    # YOUR CODE HERE - 5-8 lines                             #
    #                                                         #
    # Hint:                                                   #
    #   log_A_d: same formula as in compute_sphere_surface_area#
    #   A_d_power = exp((2/d) * log_A_d)                     #
    #   stirling = 2 * pi * e / d                            #
    #   ratio = A_d_power / stirling                          #
    #   rel_error = abs(ratio - 1.0)                          #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# __main__ TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Shannon Lower Bound for Vector Quantization on S^{d-1}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Part 1: Stirling approximation verification
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 1: Stirling approximation  A_d^{2/d} ≈ 2πe/d")
    print("─" * 60)
    print()
    print(f"  {'d':>5}  {'A_d^{2/d}':>12}  {'2πe/d':>12}  {'ratio':>8}  {'rel err':>8}  Status")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}")

    for d in [8, 16, 32, 64, 128, 256, 512]:
        info = verify_stirling_approximation(d)
        status = "OK" if info["rel_error"] < 0.05 else "WARN"
        print(
            f"  {d:5d}  {info['A_d_power']:12.6f}  {info['stirling']:12.6f}"
            f"  {info['ratio']:8.4f}  {info['rel_error']:8.4f}  [{status}]"
        )

    # ------------------------------------------------------------------
    # Part 2: Exact SLB vs simplified lower bound
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 2: Exact SLB vs simplified lower bound 1/4^b")
    print("─" * 60)
    print_bound_table(
        d_values=[64, 128, 256, 512],
        b_range=range(1, 9),
        exact_fn=compute_exact_slb,
        simplified_fn=compute_simplified_slb,
    )

    # ------------------------------------------------------------------
    # Part 3: TurboQuant's ratio to lower bound
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 3: TurboQuant_mse distortion vs lower bound")
    print("─" * 60)
    print()

    # Theoretical TurboQuant_mse values from Module 2
    TURBOQUANT_MSE = {1: 0.3634, 2: 0.1175, 3: 0.0345, 4: 0.0095}
    # Upper bound from Panter-Dite: (sqrt(3)*pi/2) / 4^b
    PANTER_DITE_FACTOR = (np.sqrt(3) * np.pi / 2)

    print(f"  {'b':>3}  {'D_mse TurboQ':>14}  {'lower bound':>12}"
          f"  {'ratio':>8}  {'PD bound':>10}  {'opt gap':>8}")
    print(f"  {'─'*3}  {'─'*14}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*8}")

    for b in [1, 2, 3, 4]:
        dq = TURBOQUANT_MSE[b]
        lb = compute_simplified_slb(b)
        ratio = dq / lb
        pd_bound = PANTER_DITE_FACTOR * lb
        opt_gap = dq / lb
        print(
            f"  {b:3d}  {dq:14.5f}  {lb:12.6f}"
            f"  {ratio:8.3f}  {pd_bound:10.5f}  {opt_gap:8.3f}"
        )

    print()
    print("  Observation: TurboQuant is within 1.45×–2.43× of the")
    print("  Shannon lower bound. (Paper states max gap is √3π/2 ≈ 2.72)")

    # ------------------------------------------------------------------
    # Part 4: Show the lower bound applies to ALL quantizers
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 4: Lower bound interpretation")
    print("─" * 60)
    print()

    for b in [1, 2, 3, 4]:
        lb = compute_simplified_slb(b)
        print(f"  b={b}: NO quantizer can achieve D_mse < {lb:.6f}")
        print(f"       (TurboQuant achieves {TURBOQUANT_MSE[b]:.5f} = "
              f"{TURBOQUANT_MSE[b]/lb:.2f}× this lower bound)")
        print()

    print("The lower bound 1/4^b is tight: proven by Shannon's theorem + Yao's")
    print("minimax principle.  Any quantizer must have D_mse ≥ lower bound.")
    print()
    print("DONE — lower bound computation complete.")

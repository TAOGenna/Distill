"""
Exercise 4: MSE Distortion — Theory vs. Experiment
====================================================

In Exercise 3, the TurboQuant_mse pipeline produced empirical MSE values of
0.362 (b=1), 0.116 (b=2), 0.034 (b=3), 0.009 (b=4) on 10000 random unit
vectors in d=128.  These match the paper's theoretically derived values to
within ~1.5%.

Now you will run a systematic sweep across multiple dimensions
(d = 64, 128, 256, 512, 1024) and bit-widths (b = 1, 2, 3, 4) to:

  1. Verify that empirical MSE is approximately *independent of dimension*
     (since we quantize unit-norm vectors, D_mse ≈ C(f_X, b) · d which is
     dimension-independent for unit norms).

  2. Confirm that empirical MSE lies between the information-theoretic lower
     bound (1/4^b) and the theoretical upper bound (exact for b<=4,
     Panter-Dite for b>4).

  3. Compute the ratio of empirical MSE to the lower bound, verifying the
     paper's claim that TurboQuant is within 2.7× of optimal.

From the TurboQuant paper (Theorem 3 + Theorem 1):
  - Lower bound:  D_mse ≥ 1/4^b           (Shannon information-theoretic limit)
  - Upper bound:  D_mse ≤ (√3·π/2)/4^b   (Panter-Dite, for b>4)
  - For b=1:      D_mse ≈ 0.36  vs lower bound 0.25  → ratio ≈ 1.44×
  - For b=4:      D_mse ≈ 0.009 vs lower bound 0.0039 → ratio ≈ 2.3×
  - Asymptotic:   ratio → √3·π/2 ≈ 2.72×  as b → ∞
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# PROVIDED: TurboQuantMSE (from Exercise 3 — complete working implementation)
# ---------------------------------------------------------------------------

CODEBOOKS = {
    64: {
        1: np.array([-0.09999924,  0.09999924]),
        2: np.array([-0.18799703, -0.05657672,  0.05657672,  0.18799703]),
        3: np.array([-0.26628028, -0.16682244, -0.09409769, -0.03057153,
                      0.03057153,  0.09409769,  0.16682244,  0.26628028]),
        4: np.array([-0.33587432, -0.25549745, -0.20035437, -0.15582217,
                     -0.11699011, -0.08171736, -0.04832849, -0.01603186,
                      0.01603186,  0.04832849,  0.08171736,  0.11699011,
                      0.15582217,  0.20035437,  0.25549745,  0.33587432]),
    },
    128: {
        1: np.array([-0.07070054,  0.07070054]),
        2: np.array([-0.13295555, -0.04001303,  0.04001303,  0.13295555]),
        3: np.array([-0.18844028, -0.11807956, -0.06658946, -0.02163026,
                      0.02163026,  0.06658946,  0.11807956,  0.18844028]),
        4: np.array([-0.23772688, -0.18079513, -0.14181505, -0.11029551,
                     -0.08278782, -0.05778706, -0.03421483, -0.01134267,
                      0.01134267,  0.03421483,  0.05778706,  0.08278782,
                      0.11029551,  0.14181505,  0.18079513,  0.23772688]),
    },
    256: {
        1: np.array([-0.04998714,  0.04998714]),
        2: np.array([-0.09400636, -0.02829268,  0.02829268,  0.09400636]),
        3: np.array([-0.13322052, -0.08349022, -0.04709416, -0.01529638,
                      0.01529638,  0.04709416,  0.08349022,  0.13322052]),
        4: np.array([-0.16802001, -0.12782285, -0.10027162, -0.07797148,
                     -0.05853019, -0.04087222, -0.02418884, -0.00802162,
                      0.00802162,  0.02418884,  0.04087222,  0.05853019,
                      0.07797148,  0.10027162,  0.12782285,  0.16802001]),
    },
    512: {
        1: np.array([-0.03535534,  0.03535534]),
        2: np.array([-0.06646116, -0.02001019,  0.02001019,  0.06646116]),
        3: np.array([-0.09421073, -0.05904476, -0.03330082, -0.01081611,
                      0.01081611,  0.03330082,  0.05904476,  0.09421073]),
        4: np.array([-0.11886298, -0.09039720, -0.07090732, -0.05514751,
                     -0.04139370, -0.02890120, -0.01710717, -0.00567085,
                      0.00567085,  0.01710717,  0.02890120,  0.04139370,
                      0.05514751,  0.07090732,  0.09039720,  0.11886298]),
    },
    1024: {
        1: np.array([-0.02500006,  0.02500006]),
        2: np.array([-0.04700197, -0.01414588,  0.01414588,  0.04700197]),
        3: np.array([-0.06660781, -0.04174723, -0.02354818, -0.00765094,
                      0.00765094,  0.02354818,  0.04174723,  0.06660781]),
        4: np.array([-0.08403145, -0.06391143, -0.05013741, -0.03898491,
                     -0.02927228, -0.02044804, -0.01209748, -0.00400849,
                      0.00400849,  0.01209748,  0.02044804,  0.02927228,
                      0.03898491,  0.05013741,  0.06391143,  0.08403145]),
    },
}


def generate_random_rotation(d, seed=None):
    """Generate a uniformly random orthogonal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Pi, _ = np.linalg.qr(A)
    return Pi


class TurboQuantMSE:
    """TurboQuant_mse: random rotation + Lloyd-Max scalar quantization."""

    def __init__(self, d, b, seed=42):
        self.d = d
        self.b = b
        self.Pi = generate_random_rotation(d, seed=seed)
        self.codebook = CODEBOOKS[d][b]

    def quantize_batch(self, X):
        Y = X @ self.Pi.T
        diffs = np.abs(Y[:, :, None] - self.codebook[None, None, :])
        return np.argmin(diffs, axis=2)

    def dequantize_batch(self, indices):
        return self.codebook[indices] @ self.Pi


# ---------------------------------------------------------------------------
# YOUR CODE — implement the four functions below
# ---------------------------------------------------------------------------

def compute_theoretical_upper_bound(b):
    """Compute the theoretical upper bound on TurboQuant_mse distortion.

    Uses exact numerically computed values for b = 1, 2, 3, 4 (where the
    Panter-Dite asymptotic formula is loose) and the Panter-Dite formula
    for b > 4:

        D_upper = (√3 · π / 2) / 4^b

    For b = 1, 2, 3, 4 the paper derives tighter values from the actual
    Lloyd-Max cost: 0.3634, 0.1175, 0.0345, 0.0095 respectively.

    Parameters
    ----------
    b : int
        Bit-width (bits per coordinate, b ≥ 1).

    Returns
    -------
    float
        Theoretical upper bound on D_mse.
    """
    ###########################################################
    # YOUR CODE HERE - 5-8 lines                              #
    #                                                         #
    # Hint: define a dict with exact values for b=1,2,3,4.    #
    # For b > 4, return (math.sqrt(3) * math.pi / 2) / 4**b. #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_lower_bound(b):
    """Compute the information-theoretic lower bound on MSE distortion.

    Any randomized b-bit quantizer applied to a worst-case unit-norm vector
    must have D_mse ≥ 1/4^b.  This follows from Shannon's distortion-rate
    function applied to the uniform distribution on S^{d-1} (Theorem 3 of
    the TurboQuant paper; derived via Yao's minimax principle + Shannon LB).

    Parameters
    ----------
    b : int
        Bit-width.

    Returns
    -------
    float
        Lower bound 1/4^b.
    """
    ###########################################################
    # YOUR CODE HERE - 2 lines                                #
    #                                                         #
    # Hint: 1.0 / (4 ** b)                                    #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def run_distortion_experiment(d, b, n_vectors=8000, seed=0):
    """Measure empirical TurboQuant_mse distortion on random unit vectors.

    Parameters
    ----------
    d : int
        Vector dimension (must be in CODEBOOKS: 64, 128, 256, 512, 1024).
    b : int
        Bit-width (1, 2, 3, or 4).
    n_vectors : int
        Number of random unit vectors to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Mean MSE = (1/n) · sum_i ||x_i - x̃_i||^2.

    Notes
    -----
    Use a different seed for the data and for the quantizer to avoid
    accidental correlations.  Recommended: seed for data = seed,
    seed for quantizer = seed + 1.
    """
    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Hint:                                                    #
    # 1. rng = np.random.default_rng(seed)                    #
    # 2. Sample X ~ N(0,I), shape (n_vectors, d)              #
    # 3. Normalize each row to unit norm                       #
    # 4. Create TurboQuantMSE(d, b, seed=seed+1)              #
    # 5. quantize_batch -> dequantize_batch -> compute MSE     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_ratio_to_optimal(empirical_mse, lower_bound):
    """Compute how many times larger empirical MSE is vs the lower bound.

    A ratio of 1.0 means the empirical distortion exactly meets the
    information-theoretic limit.  TurboQuant achieves ratios of ~1.44 at
    b=1 (nearly optimal) and up to ~2.72 at high bit-widths.

    Parameters
    ----------
    empirical_mse : float
        Observed MSE from run_distortion_experiment.
    lower_bound : float
        Information-theoretic lower bound from compute_lower_bound.

    Returns
    -------
    float
        empirical_mse / lower_bound.
    """
    ###########################################################
    # YOUR CODE HERE - 2-3 lines                              #
    #                                                         #
    # Hint: plain division.  Guard against lower_bound == 0.  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# TEST HARNESS — do not modify below this line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    dims = [64, 128, 256, 512, 1024]
    bits = [1, 2, 3, 4]
    n_vecs = 8000

    print("=" * 80)
    print("TurboQuant_mse: Theory vs Experiment across Dimensions and Bit-Widths")
    print("=" * 80)
    print(f"  n_vectors = {n_vecs} per experiment")
    print()

    # ---- Main table ----
    header = (f"{'d':>6}  {'b':>2}  {'empirical':>10}  {'upper_bound':>12}  "
              f"{'lower_bound':>12}  {'ratio':>8}  {'sandwiched':>11}")
    print(header)
    print("-" * 80)

    for d in dims:
        for b in bits:
            emp = run_distortion_experiment(d, b, n_vectors=n_vecs)
            ub = compute_theoretical_upper_bound(b)
            lb = compute_lower_bound(b)
            ratio = compute_ratio_to_optimal(emp, lb)
            sandwiched = "YES" if lb <= emp <= ub * 1.05 else "NO"
            print(f"  {d:>6}  {b:>2}  {emp:>10.5f}  {ub:>12.5f}  "
                  f"{lb:>12.6f}  {ratio:>8.3f}  {sandwiched:>11s}")
        print()

    # ---- Summary: ratio by bit-width (averaged over dimensions) ----
    print("=" * 80)
    print("Ratio empirical/lower_bound by bit-width (averaged over all dimensions)")
    print("Expected: ~1.44 at b=1, increasing toward ~2.7 at high b")
    print("=" * 80)
    print(f"{'b':>4}  {'mean ratio':>12}  {'min ratio':>12}  {'max ratio':>12}  "
          f"{'asymptotic':>12}")
    for b in bits:
        ratios = []
        for d in dims:
            emp = run_distortion_experiment(d, b, n_vectors=n_vecs)
            lb = compute_lower_bound(b)
            ratios.append(compute_ratio_to_optimal(emp, lb))
        asymptotic = compute_theoretical_upper_bound(b) / compute_lower_bound(b)
        print(f"  {b:2d}  {np.mean(ratios):>12.3f}  {np.min(ratios):>12.3f}  "
              f"{np.max(ratios):>12.3f}  {asymptotic:>12.3f}")

    print()
    print("Note: ratio approaches (√3·π/2) ≈ 2.72 as b → ∞")
    print("      At b=1: TurboQuant is only ~1.44× from optimal (almost tight!)")

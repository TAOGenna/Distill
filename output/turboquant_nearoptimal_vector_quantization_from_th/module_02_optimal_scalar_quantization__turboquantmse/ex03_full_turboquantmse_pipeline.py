"""
Exercise 3: Full TurboQuant_mse Pipeline
=========================================

In Exercise 2 you computed the optimal codebooks for the Beta distribution
(hypersphere coordinate distribution) using Lloyd-Max.  You confirmed that for
d=128, the total MSE matches the paper's values: 0.36 (b=1), 0.117 (b=2),
0.034 (b=3), 0.009 (b=4).

Now you will assemble the *complete* TurboQuant_mse quantizer (Algorithm 1
from the TurboQuant paper) which has three components:

  Setup (once per quantizer instance):
    - Generate a random orthogonal rotation matrix Π ∈ R^{d×d}
    - Load the precomputed Lloyd-Max codebook for the given (d, b)

  Quant (per vector x ∈ S^{d-1}):
    1.  y   = Π · x                            (random rotation)
    2.  idx_j = argmin_k |y_j - c_k|           (nearest centroid per coordinate)

  DeQuant (per index vector idx):
    1.  ỹ_j = c_{idx_j}                        (table lookup)
    2.  x̃   = Π^T · ỹ                          (inverse rotation)

The resulting MSE should match the paper: D_mse = d · C(f_X, b).

Key insight: the full pipeline is O(d²) dominated by the rotation, with O(d·2^b)
for the per-coordinate nearest-centroid lookup.  Everything else is negligible.
"""

import numpy as np
from scipy import integrate, special


# ---------------------------------------------------------------------------
# PROVIDED: Precomputed codebooks for d=128, b=1,2,3,4
# (These were computed by running Lloyd-Max from Exercise 2 at d=128)
# ---------------------------------------------------------------------------

CODEBOOKS = {
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
}

# Theoretical MSE targets from the TurboQuant paper (Theorem 1)
THEORETICAL_MSE = {1: 0.3634, 2: 0.1175, 3: 0.0345, 4: 0.0095}


# ---------------------------------------------------------------------------
# PROVIDED: Random rotation generator
# ---------------------------------------------------------------------------

def generate_random_rotation(d, seed=None):
    """Generate a uniformly random orthogonal matrix via QR decomposition.

    Applies QR decomposition to a random Gaussian matrix to produce a
    Haar-distributed random orthogonal matrix.

    Parameters
    ----------
    d : int
        Dimension. Output is d×d orthogonal.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (d, d)
        Orthogonal matrix Π satisfying Π^T · Π = I.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Pi, _ = np.linalg.qr(A)
    return Pi


# ---------------------------------------------------------------------------
# YOUR CODE — implement the TurboQuantMSE class methods below
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """TurboQuant_mse: random rotation + Lloyd-Max scalar quantization.

    Implements Algorithm 1 from the TurboQuant paper.

    Parameters
    ----------
    d : int
        Vector dimension.  Must be in CODEBOOKS.
    b : int
        Bits per coordinate (1, 2, 3, or 4).
    seed : int
        Random seed for the rotation matrix.
    """

    def __init__(self, d, b, seed=42):
        if d not in CODEBOOKS:
            raise ValueError(f"d={d} not in precomputed codebooks. "
                             f"Use one of: {list(CODEBOOKS.keys())}")
        if b not in CODEBOOKS[d]:
            raise ValueError(f"b={b} not in codebooks for d={d}.")
        self.d = d
        self.b = b
        self.Pi = generate_random_rotation(d, seed=seed)   # shape (d, d)
        self.codebook = CODEBOOKS[d][b]                    # shape (2^b,)

    def quantize(self, x):
        """Rotate x and assign each coordinate to the nearest codebook entry.

        Implements lines 5-6 of Algorithm 1 (Quant_mse):
          y = Π · x
          idx_j = argmin_k |y_j - c_k|

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Unit-norm input vector.

        Returns
        -------
        np.ndarray of int, shape (d,)
            Index array where indices[j] ∈ {0, ..., 2^b - 1}.
        """
        ###########################################################
        # YOUR CODE HERE - 6-8 lines                              #
        #                                                         #
        # Hint:                                                    #
        # 1. Compute y = self.Pi @ x                              #
        # 2. Compute absolute differences between each y_j and    #
        #    each codebook entry.  Shape: (d, 2^b).               #
        #    Use broadcasting: y[:, None] - self.codebook[None,:] #
        # 3. Take argmin along axis=1 to get indices of shape (d,)#
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def dequantize(self, indices):
        """Look up codebook entries and rotate back to the original basis.

        Implements lines 8-9 of Algorithm 1 (DeQuant_mse):
          ỹ_j = c_{idx_j}
          x̃   = Π^T · ỹ

        Parameters
        ----------
        indices : np.ndarray of int, shape (d,)
            Quantization indices from quantize().

        Returns
        -------
        np.ndarray, shape (d,)
            Reconstructed vector x̃ ≈ x.
        """
        ###########################################################
        # YOUR CODE HERE - 4-6 lines                              #
        #                                                         #
        # Hint:                                                    #
        # 1. y_tilde = self.codebook[indices]  (simple indexing)  #
        # 2. x_tilde = self.Pi.T @ y_tilde                        #
        # 3. Return x_tilde                                        #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def quantize_batch(self, X):
        """Vectorized quantization for a batch of vectors.

        Rotates all vectors in one matrix multiply, then assigns
        each coordinate to the nearest centroid.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Matrix of n unit-norm input vectors (one per row).

        Returns
        -------
        np.ndarray of int, shape (n, d)
            Quantization indices, one row per input vector.
        """
        ###########################################################
        # YOUR CODE HERE - 5-8 lines                              #
        #                                                         #
        # Hint:                                                    #
        # 1. Y = X @ self.Pi.T  (rotate all rows at once, shape   #
        #    (n, d)).  Note: (Pi @ x^T)^T = x @ Pi^T             #
        # 2. Compute absolute differences: Y[:,:,None] vs         #
        #    self.codebook[None,None,:] — shape (n, d, 2^b).      #
        # 3. argmin along axis=2.                                  #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def dequantize_batch(self, indices):
        """Vectorized dequantization for a batch of index arrays.

        Parameters
        ----------
        indices : np.ndarray of int, shape (n, d)
            Quantization indices from quantize_batch().

        Returns
        -------
        np.ndarray, shape (n, d)
            Reconstructed vectors, one per row.
        """
        Y_tilde = self.codebook[indices]          # (n, d) centroid lookup
        X_tilde = Y_tilde @ self.Pi               # (n, d) inverse rotation
        return X_tilde


def measure_empirical_mse(quantizer, vectors):
    """Quantize and dequantize a batch of vectors; return average MSE.

    MSE = (1/n) * sum_i ||x_i - x̃_i||^2

    Parameters
    ----------
    quantizer : TurboQuantMSE
        An initialized quantizer instance.
    vectors : np.ndarray, shape (n, d)
        Unit-norm input vectors (each row has unit L2 norm).

    Returns
    -------
    float
        Average MSE over all n vectors.
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                              #
    #                                                         #
    # Hint: use quantize_batch and dequantize_batch for       #
    # efficiency.  Compute the squared L2 norm of each        #
    # residual x_i - x̃_i and average.                        #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# TEST HARNESS — do not modify below this line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0)
    n_vectors = 10000
    d = 128

    print("=" * 70)
    print("TurboQuant_mse: Empirical vs Theoretical MSE (d=128, n=10000 vectors)")
    print("=" * 70)
    print(f"{'b':>3}  {'empirical MSE':>14}  {'theoretical':>13}  "
          f"{'lower bound':>13}  {'ratio emp/theory':>17}")

    all_ok = True
    for b in [1, 2, 3, 4]:
        # Generate random unit vectors
        X = rng.standard_normal((n_vectors, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        qmse = TurboQuantMSE(d=d, b=b, seed=42)
        emp_mse = measure_empirical_mse(qmse, X)
        theory = THEORETICAL_MSE[b]
        lower = 1.0 / (4 ** b)
        ratio = emp_mse / theory

        status = "OK" if abs(ratio - 1.0) < 0.05 else "CHECK"
        print(f"  {b:1d}  {emp_mse:14.5f}  {theory:13.5f}  "
              f"{lower:13.6f}  {ratio:17.4f}  [{status}]")
        if status != "OK":
            all_ok = False

    print()
    print("=" * 70)
    print("Data-Oblivious Property: MSE should be ~constant across input types")
    print("=" * 70)
    qmse = TurboQuantMSE(d=d, b=1, seed=7)

    # Three structurally different vectors
    e1 = np.zeros(d); e1[0] = 1.0                           # axis-aligned
    x_rand = rng.standard_normal(d); x_rand /= np.linalg.norm(x_rand)   # random
    x_sparse = np.zeros(d); x_sparse[:5] = 1.0/np.sqrt(5)   # 5-sparse

    test_vecs = [("axis-aligned e1", e1), ("random unit vec", x_rand),
                 ("5-sparse unit", x_sparse)]

    for name, x in test_vecs:
        mse_estimates = []
        for trial in range(200):
            q = TurboQuantMSE(d=d, b=1, seed=trial)
            idx = q.quantize(x)
            x_hat = q.dequantize(idx)
            mse_estimates.append(np.sum((x - x_hat)**2))
        avg_mse = np.mean(mse_estimates)
        print(f"  {name:>20s}: average MSE over 200 seeds = {avg_mse:.4f}")

    print()
    print("Expected: all three ≈ 0.36 (theoretical b=1 distortion)")
    print()

    print("=" * 70)
    print("Compression Stats")
    print("=" * 70)
    float_bytes = d * 4  # float32
    for b in [1, 2, 3, 4]:
        quant_bytes = (d * b + 7) // 8
        compression = float_bytes / quant_bytes
        print(f"  b={b}: {float_bytes} bytes → {quant_bytes} bytes  "
              f"({compression:.1f}× compression)")

    print()
    if all_ok:
        print("All theoretical bound checks PASSED.")
    else:
        print("Some checks may need review (empirical noise at n=10000).")

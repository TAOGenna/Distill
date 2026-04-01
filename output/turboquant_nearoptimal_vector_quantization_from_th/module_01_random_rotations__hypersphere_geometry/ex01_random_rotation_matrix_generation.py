"""
Exercise 1: Random Rotation Matrix Generation
=============================================

TurboQuant's first step is multiplying an input vector by a Haar-uniform
random orthogonal matrix Π. This transforms any worst-case vector into a
uniformly random point on the unit hypersphere S^{d-1}, making the
quantizer data-oblivious.

In this exercise you will implement:
  1. generate_random_rotation(d, seed) — the QR-based Haar-measure rotation
  2. rotate_vector(x, Pi)             — apply Π to a vector
  3. rotate_back(y, Pi)               — apply Π^T to invert the rotation

The sign correction in (1) is subtle but necessary for exact Haar measure.
See the lesson README for the theoretical explanation.

References:
  - TurboQuant paper §3.1 (MSE Optimal TurboQuant)
  - Lemma 1 (coordinate distribution of random point on hypersphere)
"""

import numpy as np
import numpy.linalg as la


# ---------------------------------------------------------------------------
# Part 1: Generate a Haar-uniform random orthogonal matrix
# ---------------------------------------------------------------------------

def generate_random_rotation(d, seed=None):
    """
    Generate a d×d Haar-uniform random orthogonal matrix via QR decomposition
    with sign correction.

    The algorithm:
      1. Draw G ~ N(0,1)^{d×d}  (i.i.d. Gaussian entries)
      2. Compute QR decomposition: G = Q · R
      3. Apply sign correction: Q[:,j] *= sign(R[j,j]) for each column j
         (this forces diag(R) > 0, making the decomposition unique and Q
          exactly Haar-distributed)
      4. Return Q

    Without step 3, numpy's QR may return a Q that is NOT Haar-uniform
    because LAPACK does not guarantee positive diagonal entries of R.

    Parameters
    ----------
    d : int
        Dimension. The returned matrix is shape (d, d).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Pi : np.ndarray, shape (d, d), dtype float64
        Orthogonal matrix drawn from the Haar measure on O(d).
        Satisfies Pi.T @ Pi == I_d to machine precision (~1e-14).

    Examples
    --------
    >>> Pi = generate_random_rotation(4, seed=0)
    >>> np.allclose(Pi.T @ Pi, np.eye(4), atol=1e-12)
    True
    """
    ###########################################################################
    # YOUR CODE HERE - 8-10 lines                                             #
    #                                                                         #
    # Hint 1: Use np.random.default_rng(seed) for a seeded random generator. #
    # Hint 2: rng.standard_normal((d, d)) gives the d×d Gaussian matrix G.   #
    # Hint 3: Q, R = np.linalg.qr(G) gives the QR decomposition.             #
    # Hint 4: np.diag(R) extracts the diagonal of R.                         #
    # Hint 5: np.sign(np.diag(R)) gives +1 or -1 for each column.            #
    # Hint 6: Multiply Q by the signs array — think about broadcasting:      #
    #         signs has shape (d,), Q has shape (d, d). You want to multiply  #
    #         each COLUMN j of Q by signs[j]. Use signs[np.newaxis, :].       #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ---------------------------------------------------------------------------
# Part 2: Apply and invert the rotation
# ---------------------------------------------------------------------------

def rotate_vector(x, Pi):
    """
    Apply rotation Π to vector x, computing y = Π · x.

    Since Π is orthogonal, this preserves:
      - L2 norm:      ‖Π·x‖ = ‖x‖
      - Inner product: ⟨Π·x, Π·y⟩ = ⟨x, y⟩

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        Input vector (any norm, any content).
    Pi : np.ndarray, shape (d, d)
        Orthogonal rotation matrix from generate_random_rotation().

    Returns
    -------
    y : np.ndarray, shape (d,)
        Rotated vector Π · x.
    """
    ###########################################################################
    # YOUR CODE HERE - 2-3 lines                                              #
    #                                                                         #
    # Hint: This is just a matrix-vector multiplication.                      #
    # Use Pi @ x  or  np.dot(Pi, x).                                         #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


def rotate_back(y, Pi):
    """
    Apply the INVERSE rotation Π^T to vector y, computing x̃ = Π^T · y.

    For an orthogonal matrix, the inverse is the transpose: Π^{-1} = Π^T.
    This is how TurboQuant dequantizes: it rotates the reconstructed
    coordinates back to the original basis.

    Parameters
    ----------
    y : np.ndarray, shape (d,)
        Rotated vector (e.g., a reconstructed coordinate vector).
    Pi : np.ndarray, shape (d, d)
        Orthogonal rotation matrix from generate_random_rotation().

    Returns
    -------
    x_hat : np.ndarray, shape (d,)
        Unrotated vector Π^T · y.

    Examples
    --------
    >>> Pi = generate_random_rotation(8, seed=1)
    >>> x = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.allclose(rotate_back(rotate_vector(x, Pi), Pi), x, atol=1e-12)
    True
    """
    ###########################################################################
    # YOUR CODE HERE - 2-3 lines                                              #
    #                                                                         #
    # Hint: The inverse of an orthogonal matrix is its transpose.             #
    # Use Pi.T @ y.                                                           #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ---------------------------------------------------------------------------
# Provided helper functions (do not modify)
# ---------------------------------------------------------------------------

def verify_orthogonality(Pi):
    """
    Check that Π^T · Π ≈ I (orthogonality condition).

    Parameters
    ----------
    Pi : np.ndarray, shape (d, d)

    Returns
    -------
    max_err : float
        Maximum absolute entry of |Π^T·Π - I|. Should be < 1e-12.
    is_orthogonal : bool
        True if max_err < 1e-12.
    """
    d = Pi.shape[0]
    residual = Pi.T @ Pi - np.eye(d)
    max_err = np.abs(residual).max()
    return max_err, max_err < 1e-12


def verify_norm_preservation(x, Pi):
    """
    Check that rotation preserves L2 norm: ‖Π·x‖ ≈ ‖x‖.

    Parameters
    ----------
    x : np.ndarray, shape (d,)
    Pi : np.ndarray, shape (d, d)

    Returns
    -------
    rel_err : float
        Relative error |‖Π·x‖ - ‖x‖| / ‖x‖. Should be < 1e-12.
    is_preserved : bool
    """
    y = rotate_vector(x, Pi)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    rel_err = abs(norm_y - norm_x) / (norm_x + 1e-300)
    return rel_err, rel_err < 1e-12


def verify_inner_product_preservation(x1, x2, Pi):
    """
    Check that rotation preserves inner products: ⟨Π·x1, Π·x2⟩ ≈ ⟨x1, x2⟩.

    Parameters
    ----------
    x1, x2 : np.ndarray, shape (d,)
    Pi : np.ndarray, shape (d, d)

    Returns
    -------
    rel_err : float
        Relative error |⟨Π·x1,Π·x2⟩ - ⟨x1,x2⟩| / |⟨x1,x2⟩|. Should be < 1e-12.
    is_preserved : bool
    """
    ip_original = np.dot(x1, x2)
    ip_rotated = np.dot(rotate_vector(x1, Pi), rotate_vector(x2, Pi))
    rel_err = abs(ip_rotated - ip_original) / (abs(ip_original) + 1e-300)
    return rel_err, rel_err < 1e-10


# ---------------------------------------------------------------------------
# Main: full verification across dimensions and adversarial test vectors
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Random Rotation Matrix Generation")
    print("=" * 70)

    # Test dimensions representing realistic transformer head sizes
    dimensions = [64, 128, 256]

    print(f"\n{'Dim':>5} | {'Orthogonal?':^12} | {'Norm Pres?':^12} | {'IP Pres?':^12} | {'Err Ortho':>10}")
    print("-" * 70)

    all_passed = True
    for d in dimensions:
        Pi = generate_random_rotation(d, seed=d * 7)

        # Adversarial test vectors that would cause problems for naive quantizers
        x_one_hot = np.zeros(d); x_one_hot[0] = 1.0          # worst-case: all energy in one coord
        x_linear  = np.linspace(-1, 1, d); x_linear /= np.linalg.norm(x_linear)
        x_cluster = np.zeros(d); x_cluster[:d//8] = 1.0; x_cluster /= np.linalg.norm(x_cluster)

        ortho_err, ortho_ok  = verify_orthogonality(Pi)
        norm_err,  norm_ok   = verify_norm_preservation(x_one_hot, Pi)
        ip_err,    ip_ok     = verify_inner_product_preservation(x_one_hot, x_linear, Pi)

        status = "PASS" if (ortho_ok and norm_ok and ip_ok) else "FAIL"
        if not (ortho_ok and norm_ok and ip_ok):
            all_passed = False

        print(f"{d:>5} | {'YES' if ortho_ok else 'NO':^12} | {'YES' if norm_ok else 'NO':^12} | "
              f"{'YES' if ip_ok else 'NO':^12} | {ortho_err:10.2e}  [{status}]")

    print("\n--- Adversarial vector: all energy in one coordinate ---")
    d = 128
    Pi = generate_random_rotation(d, seed=42)

    x_worst = np.zeros(d); x_worst[0] = 1.0
    y = rotate_vector(x_worst, Pi)

    print(f"Before rotation: max coord = {x_worst.max():.4f}, "
          f"std = {x_worst.std():.4f}, coords with |x|>0.1: {(np.abs(x_worst)>0.1).sum()}")
    print(f"After  rotation: max coord = {np.abs(y).max():.4f}, "
          f"std = {y.std():.4f}, coords with |y|>0.1: {(np.abs(y)>0.1).sum()}")
    print(f"Theoretical std after rotation: 1/sqrt({d}) = {1/np.sqrt(d):.4f}")
    print(f"Empirical   std after rotation: {y.std():.4f}")

    print("\n--- Round-trip test: rotate then rotate_back ---")
    x_test = np.random.default_rng(99).standard_normal(d)
    x_test /= np.linalg.norm(x_test)
    x_reconstructed = rotate_back(rotate_vector(x_test, Pi), Pi)
    round_trip_err = np.linalg.norm(x_reconstructed - x_test)
    print(f"Round-trip error ‖rotate_back(rotate(x)) - x‖ = {round_trip_err:.2e}")
    print(f"Round-trip passed: {round_trip_err < 1e-12}")

    print("\n--- Summary ---")
    if all_passed:
        print("✓ All verification checks PASSED")
        print("✓ Π is orthogonal (max |Π^T·Π - I| < 1e-12)")
        print("✓ Norms are preserved (relative error < 1e-12)")
        print("✓ Inner products are preserved (relative error < 1e-10)")
        print("✓ Adversarial one-hot vector becomes 'random-looking' after rotation")
    else:
        print("✗ Some checks FAILED — review your implementation")

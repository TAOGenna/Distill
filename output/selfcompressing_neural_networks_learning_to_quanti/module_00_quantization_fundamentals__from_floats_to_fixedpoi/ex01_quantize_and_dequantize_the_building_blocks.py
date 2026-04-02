"""
Exercise 1: Quantize and Dequantize — The Building Blocks
==========================================================
Module 0: Quantization Fundamentals

Implement the core quantization and dequantization functions that underpin the
Self-Compressing Neural Networks paper (arXiv 2301.13142).

The paper's quantization formula is:
    q(x, b, e) = 2^e * round( clamp( 2^(-e) * x, -2^(b-1), 2^(b-1)-1 ) )

Your task: implement `quantize` (scaling + clamping + rounding) and
`dequantize` (rescaling back to float space).

Run this file to see a table of reconstruction errors across bit-widths and
verify that more bits means less error.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Helper utilities (provided)
# ---------------------------------------------------------------------------

def print_quantization_table(results: dict) -> None:
    """
    Print a formatted table of reconstruction MSE vs (exponent, bit-width).

    Parameters
    ----------
    results : dict
        Nested dict: results[exponent][num_bits] = mse_value (float)
    """
    bit_widths = sorted({b for exp_dict in results.values() for b in exp_dict})
    print(f"\n{'Reconstruction MSE by exponent and bit-width':^72}")
    print(f"{'(lower is better — 8-bit should be near zero)':^72}")
    print("-" * 72)
    header = f"{'exponent':>10}" + "".join(f"  {b:>5}-bit" for b in bit_widths)
    print(header)
    print("-" * 72)
    for exp in sorted(results.keys()):
        row = f"{exp:>10.1f}"
        for b in bit_widths:
            mse = results[exp].get(b, float("nan"))
            if mse < 1e-6:
                row += f"  {mse:>8.2e}"
            else:
                row += f"  {mse:>8.5f}"
        print(row)
    print("-" * 72)


def make_kaiming_weights(out_channels: int, in_channels: int,
                         kernel_size: int, seed: int = 42) -> torch.Tensor:
    """
    Generate realistic convolutional weights using Kaiming uniform init.

    Parameters
    ----------
    out_channels : int
        Number of output feature channels.
    in_channels : int
        Number of input feature channels.
    kernel_size : int
        Spatial size of the square kernel.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size).
    """
    torch.manual_seed(seed)
    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    return layer.weight.data.clone()


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION — quantize and dequantize
# ---------------------------------------------------------------------------

def quantize(x: torch.Tensor, num_bits: int, exponent: float) -> torch.Tensor:
    """
    Quantize a floating-point tensor to a signed b-bit integer grid.

    Maps real-valued weights onto the discrete set of signed b-bit integers
    {-2^(b-1), ..., 2^(b-1)-1}, scaled by 2^exponent. The three steps are:
      1. Scale: multiply by 2^(-exponent) to express x in "integer units"
      2. Clamp: clip to the representable signed integer range
      3. Round: snap to the nearest integer

    Parameters
    ----------
    x : torch.Tensor
        Floating-point weights to quantize (any shape).
    num_bits : int
        Number of bits b. The representable integer range is
        [-2^(b-1), 2^(b-1)-1]. E.g. b=8 gives [-128, 127].
    exponent : float
        Scale exponent e. The step size between adjacent quantized
        values is 2^e. Smaller e = finer grid = more precision.

    Returns
    -------
    torch.Tensor
        Integer-valued tensor (stored as float32), same shape as x.
        Values are in the signed b-bit range.

    Examples
    --------
    >>> w = torch.tensor([0.047, -0.12, 0.31])
    >>> q = quantize(w, num_bits=3, exponent=-3)
    >>> q   # integers in [-4, 3]
    tensor([ 0., -1.,  2.])
    """
    ###########################################################
    # YOUR CODE HERE — 8-12 lines                             #
    #                                                         #
    # Step 1: Compute the signed integer bounds for b bits.   #
    #   q_min = -(2^(num_bits-1))     e.g. -128 for 8-bit    #
    #   q_max =  2^(num_bits-1) - 1   e.g.  127 for 8-bit    #
    #   NOTE: upper bound is 2^(b-1)-1, NOT 2^(b-1)!         #
    #                                                         #
    # Step 2: Scale x into "integer space":                   #
    #   x_scaled = 2^(-exponent) * x                         #
    #   (divides by step size so one step = 1.0)              #
    #                                                         #
    # Step 3: Clamp to [q_min, q_max].                        #
    #   Use x_scaled.clamp(q_min, q_max)                      #
    #                                                         #
    # Step 4: Round to nearest integer.                       #
    #   Use .round() — this is the discretization step        #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def dequantize(qx: torch.Tensor, exponent: float) -> torch.Tensor:
    """
    Restore floating-point values from a quantized integer tensor.

    Multiplies integer values by the step size 2^exponent to map back to
    the original floating-point scale. This is the inverse of the scaling
    step in quantize().

    Parameters
    ----------
    qx : torch.Tensor
        Integer-valued tensor (the output of quantize()), any shape.
    exponent : float
        Scale exponent e. Must match the exponent used in quantize().
        The dequantized value is qx * 2^exponent.

    Returns
    -------
    torch.Tensor
        Floating-point reconstruction of the original tensor, same shape.

    Examples
    --------
    >>> q = torch.tensor([0., -1., 2.])
    >>> dequantize(q, exponent=-3)
    tensor([ 0.0000, -0.1250,  0.2500])
    """
    ###########################################################
    # YOUR CODE HERE — 3-5 lines                              #
    #                                                         #
    # Multiply qx by 2^exponent to restore the original scale.#
    # Hint: 2 ** exponent gives the step size as a Python float#
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_reconstruction_mse(original: torch.Tensor, num_bits: int,
                                exponent: float) -> float:
    """
    Quantize then dequantize, return mean squared reconstruction error.

    Parameters
    ----------
    original : torch.Tensor
        Original floating-point weights.
    num_bits : int
        Number of quantization bits.
    exponent : float
        Scale exponent.

    Returns
    -------
    float
        Mean squared error between original and reconstructed weights.
    """
    q = quantize(original, num_bits, exponent)
    reconstructed = dequantize(q, exponent)
    return ((original - reconstructed) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Main test harness (provided — do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("Exercise 1: Quantize and Dequantize — The Building Blocks")
    print("=" * 72)

    # Generate realistic conv weights: shape (32, 1, 5, 5) = 800 weights
    weights = make_kaiming_weights(out_channels=32, in_channels=1, kernel_size=5)
    print(f"\nWeight tensor shape: {weights.shape}")
    print(f"Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, "
          f"std={weights.std():.4f}")

    # --- Test 1: Sanity check on a small example ---
    print("\n--- Sanity check (b=3, e=-3) ---")
    w_example = torch.tensor([0.0, 0.125, 0.25, 0.375, -0.125, -0.25, -0.5])
    q_example = quantize(w_example, num_bits=3, exponent=-3)
    w_hat = dequantize(q_example, exponent=-3)
    print(f"  original:     {w_example.tolist()}")
    print(f"  quantized:    {q_example.tolist()}  (integers in [-4, 3])")
    print(f"  reconstructed:{[round(v, 4) for v in w_hat.tolist()]}")
    # These exact multiples of 0.125 should reconstruct perfectly
    perfect = torch.allclose(w_example, w_hat, atol=1e-5)
    print(f"  Perfect reconstruction of grid-aligned values: {perfect}")
    assert perfect, "Grid-aligned values should reconstruct exactly!"

    # --- Test 2: Sweep bit-widths and exponents ---
    print("\n--- Sweeping bit-widths [1, 2, 3, 4, 8] and exponents [-6, -4, -2] ---")
    bit_widths = [1, 2, 3, 4, 8]
    exponents = [-6.0, -4.0, -2.0]

    results = {}
    for e in exponents:
        results[e] = {}
        for b in bit_widths:
            mse = compute_reconstruction_mse(weights, b, e)
            results[e][b] = mse

    print_quantization_table(results)

    # --- Test 3: Sweep exponents to find the best for each bit-width ---
    print("\n--- reconstruction_mse with near-optimal exponents ---")
    print("  (we sweep e in steps of 0.5 and pick the best for each b)")
    e_candidates = [e * 0.5 for e in range(-20, 0)]  # -10.0 to -0.5
    for b in bit_widths:
        best_e, best_mse = None, float("inf")
        for e in e_candidates:
            mse = compute_reconstruction_mse(weights, b, e)
            if mse < best_mse:
                best_mse, best_e = mse, e
        q = quantize(weights, b, best_e)
        n_unique = len(torch.unique(q))
        print(f"  b={b:2d} bits | best_e={best_e:5.1f} | {n_unique:4d} unique levels | "
              f"reconstruction_mse={best_mse:.6f}")

    # --- Test 4: Verify monotone improvement with more bits (at e=-7) ---
    print("\n--- Verifying MSE decreases monotonically with more bits (e=-7) ---")
    # e=-7 gives step=2^(-7)=0.0078; weights in [-0.2, 0.2] span ~51 integer values
    # so bits 1..8 genuinely use more and more of the integer range
    e_test = -7.0
    prev_mse = float("inf")
    for b in bit_widths:
        mse = compute_reconstruction_mse(weights, b, e_test)
        print(f"  b={b:2d} bits | MSE={mse:.2e} | "
              f"{'✓ improvement' if mse < prev_mse else '✗ regression!'}")
        assert mse <= prev_mse, f"MSE should decrease with more bits! b={b}"
        prev_mse = mse

    # --- Final assertion: 8-bit with e=-7 should be near-perfect ---
    mse_8bit = compute_reconstruction_mse(weights, 8, e_test)
    print(f"\n  8-bit reconstruction_mse = {mse_8bit:.2e} "
          f"(should be < 1e-4)")
    assert mse_8bit < 1e-4, (
        f"8-bit MSE {mse_8bit:.2e} exceeds 1e-4 — check your implementation!"
    )

    print("\n" + "=" * 72)
    print("All assertions passed! Your quantize/dequantize implementation is correct.")
    print("\nKey observations:")
    print("  1. Each additional bit roughly halves the MSE (~6 dB improvement)")
    print("  2. The exponent shifts MSE — e=-6 is better for small weights")
    print("  3. 8-bit is near-lossless; 1-bit is extremely lossy")
    print("  4. This is the core building block of QConv2d (Module 2)")
    print("=" * 72)

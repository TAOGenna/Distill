"""
Exercise 2: Bit-Width Explorer — Precision vs. Compression
===========================================================
Module 0: Quantization Fundamentals

Building on Exercise 1's quantize/dequantize, explore how reconstruction
quality varies across the 2D space of (exponent, bit-width) choices.

Your tasks:
  1. compute_quantization_stats(weights, exponents, bit_widths) — build a
     comprehensive statistics table for all (e, b) combinations.
  2. find_optimal_exponent(weights, num_bits) — find the exponent minimizing
     reconstruction MSE for a given bit-width.

Key question: Why does the paper initialize e=-8.0 and b=2.0?
Run this file to find out empirically.
"""

import torch
import math
import numpy as np


# ---------------------------------------------------------------------------
# Provided: quantize and dequantize from Exercise 1
# ---------------------------------------------------------------------------

def quantize(x: torch.Tensor, num_bits: int, exponent: float) -> torch.Tensor:
    """Quantize to signed b-bit integers with scale 2^exponent."""
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1
    x_scaled = (2 ** (-exponent)) * x
    return x_scaled.clamp(q_min, q_max).round()


def dequantize(qx: torch.Tensor, exponent: float) -> torch.Tensor:
    """Dequantize: multiply integers by step size 2^exponent."""
    return (2 ** exponent) * qx


def make_kaiming_weights(out_channels: int, in_channels: int,
                         kernel_size: int, seed: int = 42) -> torch.Tensor:
    """Generate convolutional weights using Kaiming uniform initialization."""
    torch.manual_seed(seed)
    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    return layer.weight.data.clone()


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION
# ---------------------------------------------------------------------------

def compute_quantization_stats(
    weights: torch.Tensor,
    exponents: list,
    bit_widths: list,
) -> dict:
    """
    Compute quantization quality metrics for all (exponent, bit_width) pairs.

    For each combination, quantize the weights, dequantize, and measure:
      - mse: mean squared error between original and reconstruction
      - max_abs_error: maximum absolute error (worst-case weight error)
      - num_unique_values: number of distinct integer values used
      - snr_db: signal-to-noise ratio in decibels
            SNR_dB = 10 * log10(signal_power / noise_power)
            where signal_power = mean(weights^2),
                  noise_power  = mean((weights - reconstructed)^2)
            Return +inf if noise_power < 1e-15 (perfect reconstruction).

    Parameters
    ----------
    weights : torch.Tensor
        1D or multi-dimensional weight tensor to analyze.
    exponents : list of float
        Exponent values e to sweep (e.g., [-8, -6, -4, -2, 0]).
    bit_widths : list of int
        Bit-widths b to sweep (e.g., [1, 2, 3, 4, 8]).

    Returns
    -------
    dict
        Nested dict: stats[exponent][bit_width] = {
            "mse": float,
            "max_abs_error": float,
            "num_unique_values": int,
            "snr_db": float,
        }

    Notes
    -----
    - SNR formula: 10 * log10(mean(w^2) / MSE)
    - Use math.log10 for the dB calculation (or torch equivalent)
    - Handle divide-by-zero: return float('inf') when MSE < 1e-15
    """
    ###########################################################
    # YOUR CODE HERE — 15-20 lines                            #
    #                                                         #
    # Outer loop: for e in exponents                          #
    #   Inner loop: for b in bit_widths                       #
    #     1. Call quantize(weights, b, e) to get integer repr #
    #     2. Call dequantize to get floating-point approx     #
    #     3. Compute mse = mean((weights - reconstructed)^2)  #
    #     4. Compute max_abs_error = max(|weights - recon|)   #
    #     5. Compute num_unique = len(torch.unique(q))        #
    #     6. Compute SNR in dB:                               #
    #        signal_power = mean(weights^2)                   #
    #        if mse < 1e-15: snr_db = inf                     #
    #        else: snr_db = 10 * log10(signal_power / mse)    #
    #     7. Store in nested dict stats[e][b]                 #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def find_optimal_exponent(
    weights: torch.Tensor,
    num_bits: int,
    e_min: float = -12.0,
    e_max: float = 4.0,
    e_step: float = 0.25,
) -> float:
    """
    Find the exponent that minimizes reconstruction MSE for a given bit-width.

    Performs a grid search over exponents from e_min to e_max (exclusive)
    in steps of e_step, returning the exponent achieving lowest MSE.

    Parameters
    ----------
    weights : torch.Tensor
        Weight tensor to quantize.
    num_bits : int
        Number of quantization bits.
    e_min : float
        Start of exponent search range (inclusive).
    e_max : float
        End of exponent search range (exclusive).
    e_step : float
        Step size for the exponent grid search.

    Returns
    -------
    float
        The exponent value (from the search grid) that minimizes MSE.

    Notes
    -----
    - Use a simple loop over np.arange(e_min, e_max, e_step)
    - Track best_e and best_mse, updating when mse < best_mse
    - The optimal exponent balances clipping error vs. rounding error:
        too small e → saturation of large weights (clipping)
        too large e → coarse grid (rounding error dominates)
    """
    ###########################################################
    # YOUR CODE HERE — 8-10 lines                             #
    #                                                         #
    # Initialize best_e = e_min, best_mse = float('inf')     #
    # For each e in np.arange(e_min, e_max, e_step):         #
    #   q = quantize(weights, num_bits, e)                    #
    #   reconstructed = dequantize(q, e)                      #
    #   mse = ((weights - reconstructed)**2).mean().item()    #
    #   if mse < best_mse: update best_mse and best_e        #
    # Return best_e                                           #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Main analysis harness (provided — do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 76)
    print("Exercise 2: Bit-Width Explorer — Precision vs. Compression")
    print("=" * 76)

    # Generate realistic weights for a 5x5 conv layer
    weights = make_kaiming_weights(out_channels=32, in_channels=1, kernel_size=5)
    w_flat = weights.flatten()
    print(f"\nWeight tensor: {weights.shape} = {w_flat.numel()} weights")
    print(f"Stats: mean={w_flat.mean():.4f}, std={w_flat.std():.4f}, "
          f"max_abs={w_flat.abs().max():.4f}")

    # --- Part A: 2D stats table ---
    exponents = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    bit_widths = [1, 2, 3, 4, 8]

    print("\n--- Part A: compute_quantization_stats() ---")
    stats = compute_quantization_stats(w_flat, exponents, bit_widths)

    # Print MSE table
    print(f"\n{'MSE table (lower is better)':^70}")
    print(f"{'exponent':>10}" + "".join(f"  {b:>4}-bit" for b in bit_widths))
    print("-" * 60)
    for e in exponents:
        row = f"{e:>10.1f}"
        for b in bit_widths:
            mse = stats[e][b]["mse"]
            row += f"  {mse:>8.2e}"
        print(row)

    # Print SNR table
    print(f"\n{'SNR (dB) table (higher is better)':^70}")
    print(f"{'exponent':>10}" + "".join(f"  {b:>4}-bit" for b in bit_widths))
    print("-" * 60)
    for e in exponents:
        row = f"{e:>10.1f}"
        for b in bit_widths:
            snr = stats[e][b]["snr_db"]
            if snr == float("inf"):
                row += f"  {'  +inf':>8}"
            else:
                row += f"  {snr:>8.1f}"
        print(row)

    # --- Part B: find_optimal_exponent ---
    print("\n--- Part B: find_optimal_exponent() ---")
    print(f"{'bit-width':>12} | {'optimal_exponent':>16} | "
          f"{'best_MSE':>10} | {'SNR (dB)':>10} | {'num_unique':>10}")
    print("-" * 70)
    for b in bit_widths:
        e_opt = find_optimal_exponent(w_flat, b)
        q_opt = quantize(w_flat, b, e_opt)
        w_hat = dequantize(q_opt, e_opt)
        mse = ((w_flat - w_hat) ** 2).mean().item()
        signal_power = (w_flat ** 2).mean().item()
        snr = 10 * math.log10(signal_power / mse) if mse > 1e-15 else float("inf")
        n_unique = len(torch.unique(q_opt))
        print(f"{b:>12} | {e_opt:>16.2f} | {mse:>10.2e} | {snr:>10.1f} | "
              f"{n_unique:>10}")

    # --- Key observations ---
    print("\n--- Key observations ---")
    # 1. Compare 2-bit performance at different exponents
    print("\n1. Impact of exponent on 2-bit quantization:")
    for e in [-8.0, -3.0, 0.0]:
        mse = stats[e][2]["mse"]
        snr = stats[e][2]["snr_db"]
        n_unique = stats[e][2]["num_unique_values"]
        print(f"   e={e:5.1f}: MSE={mse:.4f}, SNR={snr:.1f}dB, "
              f"unique_vals={n_unique}")

    # 2. Show why paper initializes e=-8
    print("\n2. Paper initializes e=-8.0, b=2.0. At init, the step size is:")
    step = 2 ** (-8.0)
    max_range = (2 ** (2 - 1) - 1) * step
    print(f"   Step size = 2^(-8) = {step:.6f}")
    print(f"   2-bit range = [{-2*step:.6f}, {max_range:.6f}]")
    print(f"   Weight range = [{w_flat.min():.4f}, {w_flat.max():.4f}]")
    pct_clipped = (w_flat.abs() > max_range).float().mean().item()
    print(f"   Fraction of weights outside 2-bit range at init: {pct_clipped:.1%}")
    print("   (Most weights are CLIPPED initially — the model learns e upward!)")

    # 3. The 6 dB/bit rule
    print("\n3. Verify ~6 dB per additional bit (at optimal exponent):")
    snrs = []
    for b in [1, 2, 3, 4]:
        e_opt = find_optimal_exponent(w_flat, b)
        q = quantize(w_flat, b, e_opt)
        w_hat = dequantize(q, e_opt)
        mse = ((w_flat - w_hat) ** 2).mean().item()
        sp = (w_flat ** 2).mean().item()
        snr = 10 * math.log10(sp / mse) if mse > 1e-15 else float("inf")
        snrs.append(snr)
        gain = snr - snrs[-2] if len(snrs) > 1 else 0.0
        print(f"   b={b}: SNR={snr:.1f} dB  (gain={gain:+.1f} dB vs prev)")

    # --- Assertion ---
    e_opt_2bit = find_optimal_exponent(w_flat, 2)
    print(f"\noptimal_exponent for 2-bit: {e_opt_2bit:.2f}")
    assert -6.0 <= e_opt_2bit <= -1.0, (
        f"optimal_exponent={e_opt_2bit} seems wrong for 2-bit Kaiming weights"
    )

    e_opt_8bit = find_optimal_exponent(w_flat, 8)
    q8 = quantize(w_flat, 8, e_opt_8bit)
    w_hat8 = dequantize(q8, e_opt_8bit)
    mse_8bit = ((w_flat - w_hat8) ** 2).mean().item()
    assert mse_8bit < 1e-4, f"8-bit optimal MSE={mse_8bit:.2e} should be < 1e-4"

    print("\n" + "=" * 76)
    print("All assertions passed!")
    print("\nConclusion: The optimal exponent shifts with bit-width.")
    print("Finer grids (smaller e) are better for more bits; coarser grids")
    print("(larger e) avoid clipping at the cost of rounding precision.")
    print("The paper's learned e finds this sweet spot automatically!")
    print("=" * 76)

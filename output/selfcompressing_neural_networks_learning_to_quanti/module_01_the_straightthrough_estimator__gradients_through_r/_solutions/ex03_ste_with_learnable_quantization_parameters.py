"""
Exercise 3: STE with Learnable Quantization Parameters — SOLUTION
=================================================================
Course: Self-Compressing Neural Networks
Module: 1 — The Straight-Through Estimator
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# STE round (from Exercise 2 — provided)
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator rounding.

    Forward:  x.round()
    Backward: identity (gradient passes through unchanged)
    """
    return (x.round() - x).detach() + x


# ---------------------------------------------------------------------------
# Part 1: Quantization with learnable exponent
# ---------------------------------------------------------------------------

def quantize_with_learnable_e(
    weights: torch.Tensor,
    e: nn.Parameter,
    num_bits: int,
) -> torch.Tensor:
    """Quantize weights using the paper's formula with a learnable exponent.

    Formula:
        qw = clip(2^(-e) * w, -2^(b-1), 2^(b-1) - 1)
        w_quantized = 2^e * ste_round(qw)

    Parameters
    ----------
    weights : torch.Tensor
        Weight tensor to quantize.
    e : nn.Parameter
        Learnable scalar exponent (grid spacing = 2^e).
    num_bits : int
        Fixed bit-width for quantization range.

    Returns
    -------
    torch.Tensor
        Dequantized weights. Gradient flows back to 'e' via STE.
    """
    # Integer range bounds
    lo = -(2.0 ** (num_bits - 1))
    hi =   2.0 ** (num_bits - 1) - 1.0

    # Scale down to integer-grid space
    scaled = (2 ** (-e)) * weights

    # Clamp to valid integer range
    clamped = scaled.clamp(lo, hi)

    # STE rounding: forward=round, backward=identity
    rounded = ste_round(clamped)

    # Dequantize back to float space
    return (2 ** e) * rounded


# ---------------------------------------------------------------------------
# Part 2: Training loop to learn 'e'
# ---------------------------------------------------------------------------

def training_loop(
    weights: torch.Tensor,
    target: torch.Tensor,
    num_bits: int,
    steps: int = 500,
    lr: float = 0.1,
    log_every: int = 50,
) -> tuple[nn.Parameter, list[float], list[float]]:
    """Learn the optimal quantization exponent 'e' via gradient descent.

    Parameters
    ----------
    weights : torch.Tensor
        Fixed weight tensor (NOT updated — only 'e' is a parameter).
    target : torch.Tensor
        Target quantized weights at the known-optimal exponent.
    num_bits : int
        Fixed bit-width for quantization range.
    steps : int
        Number of Adam gradient steps.
    lr : float
        Learning rate for Adam optimizer.
    log_every : int
        Print progress every log_every steps.

    Returns
    -------
    e : nn.Parameter
        The learned exponent after training.
    e_history : list[float]
        e value at each step.
    loss_history : list[float]
        MSE loss at each step.
    """
    # Initialize e at 0.0 (neutral starting point)
    e = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([e], lr=lr)

    e_history = []
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()

        # Quantize with current e
        w_q = quantize_with_learnable_e(weights, e, num_bits)

        # MSE loss vs target (optimal quantization)
        loss = F.mse_loss(w_q, target)

        # Record before step
        e_history.append(e.item())
        loss_history.append(loss.item())

        # Backward — STE enables gradient to flow to e
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:4d}: e={e.item():.4f},  loss={loss.item():.8f}")

    return e, e_history, loss_history


# ---------------------------------------------------------------------------
# Main (DO NOT MODIFY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Exercise 3: STE with Learnable Quantization Parameters")
    print("=" * 60)
    print()

    in_channels, out_channels, k = 32, 64, 3
    fan_in = in_channels * k * k
    scale = 1.0 / math.sqrt(fan_in)
    weights = torch.empty(out_channels, in_channels, k, k).uniform_(-scale, scale)
    weights = weights.flatten()

    num_bits = 4

    max_abs = weights.abs().max().item()
    max_int = 2 ** (num_bits - 1) - 1
    target_e = math.log2(max_abs / max_int)

    print(f"Weight stats:  mean={weights.mean():.4f}, std={weights.std():.4f}")
    print(f"               max|w|={max_abs:.6f}")
    print(f"Num bits:      {num_bits}  (range: {-2**(num_bits-1)} to {2**(num_bits-1)-1})")
    print(f"Target e:      {target_e:.4f}  (optimal grid for these weights)")
    print(f"Initial e:     0.0000  (neutral starting point)")
    print()

    e_fixed = torch.tensor(target_e)
    with torch.no_grad():
        lo = -(2 ** (num_bits - 1))
        hi = 2 ** (num_bits - 1) - 1
        scaled_opt = (2 ** (-e_fixed)) * weights
        clamped_opt = scaled_opt.clamp(lo, hi)
        target = (2 ** e_fixed) * clamped_opt.round()

    print(f"Target quantization error (optimal): {F.mse_loss(target, weights).item():.8f}")
    print()
    print("Training: learning e via gradient descent (STE enables gradients)")
    print("-" * 60)

    e_learned, e_history, loss_history = training_loop(
        weights=weights,
        target=target,
        num_bits=num_bits,
        steps=500,
        lr=0.1,
        log_every=50,
    )

    print("-" * 60)
    print()

    final_e = e_learned.item()
    final_loss = loss_history[-1]
    converged = abs(final_e - target_e) < 0.5 and final_loss < 1e-4

    print(f"Learned e  = {final_e:.4f}")
    print(f"Target e   = {target_e:.4f}")
    print(f"Difference = {abs(final_e - target_e):.4f}")
    print(f"Final MSE  = {final_loss:.2e}")
    print(f"converged: {converged}")
    print()

    e_range = max(e_history) - min(e_history)
    print(f"e trajectory range: {min(e_history):.4f} → {max(e_history):.4f}  (range={e_range:.4f})")
    assert e_range > 1.0, f"e barely moved — gradient not flowing!"
    assert final_loss < 1e-3, f"MSE too high ({final_loss:.2e})"
    assert converged, f"e={final_e:.4f} did not converge near target {target_e:.4f}"

    print()
    print("KEY INSIGHT:")
    print("  The STE enabled gradient flow through ste_round() to 'e'.")
    print("  Without STE, gradient of e would be 0 and it would never move.")
    print("  This same mechanism enables learning bit-widths 'b' in Module 2.")

"""
Exercise 3: STE with Learnable Quantization Parameters
=======================================================
Course: Self-Compressing Neural Networks
Module: 1 — The Straight-Through Estimator

GOAL
----
Use the STE to learn the quantization exponent 'e' via gradient descent.

The quantization formula from the self-compressing paper:
    qw = clip(2^(-e) * w, -2^(b-1), 2^(b-1) - 1)
    w_quantized = 2^e * round(qw)          ← STE applied to round()

The exponent 'e' controls the quantization grid spacing (step size = 2^e).
If 'e' is too large, the grid is coarse and quantization error is high.
If 'e' is too small, the grid is fine but may waste bits.

You will:
1. Implement quantize_with_learnable_e() — the per-step quantization formula
2. Implement training_loop() — learn e via gradient descent to minimize
   MSE between quantized weights and a pre-computed target

The STE makes this possible: even though round() is non-differentiable,
gradient flows through the STE back to 'e', driving it toward the optimal value.

After training for 500 steps, 'e' should converge near the known optimal
exponent for Kaiming-initialized weights.

DEPENDENCIES
------------
  pip install torch  # PyTorch >= 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# STE round (from Exercise 2 — provided for you)
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator rounding.

    Forward:  x.round()   (exact rounding)
    Backward: identity    (gradient passes through unchanged)

    This is the one-liner detach trick from the paper:
        (x.round() - x).detach() + x
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

    The formula (from the self-compressing paper):
        qw = clip(2^(-e) * w, -2^(b-1), 2^(b-1) - 1)
        w_quantized = 2^e * ste_round(qw)

    Where:
        e     = learnable exponent (scalar nn.Parameter)
        b     = num_bits (fixed integer for this exercise)
        2^e   = the quantization step size (grid spacing)
        2^(-e) = the inverse, used to map weights to integer-grid units

    Parameters
    ----------
    weights : torch.Tensor
        Weight tensor to quantize. Shape: (N,) or any shape.
        These are the actual neural network weight values.
    e : nn.Parameter
        Learnable scalar exponent. Controls the grid spacing.
        Initialized externally (typically at 0.0).
        Gradient will flow through the STE to update e.
    num_bits : int
        Fixed bit-width for quantization range.
        Range of integer values: [-2^(num_bits-1), 2^(num_bits-1) - 1]
        E.g., num_bits=2 → integers in {-2, -1, 0, 1}
             num_bits=4 → integers in {-8, ..., 7}

    Returns
    -------
    torch.Tensor
        Dequantized weights: same shape as input weights.
        Values are multiples of 2^e (on the quantization grid in float space).
        Gradient flows back through to 'e' via the STE.

    Example
    -------
    With e=0 (step=1.0), num_bits=2, weights=[0.7, -0.4, 1.8]:
        scaled = 2^0 * [0.7, -0.4, 1.8] = [0.7, -0.4, 1.8]
        clamped to [-2, 1]: [0.7, -0.4, 1.0]  (1.8 gets clamped to 1)
        rounded (STE): [1, 0, 1]
        dequantized: 2^0 * [1, 0, 1] = [1.0, 0.0, 1.0]
    """
    ###########################################################
    # YOUR CODE HERE - 6-8 lines                              #
    #                                                         #
    # Step 1: Compute the b-bit integer range bounds          #
    #   lo = -2^(num_bits - 1)                                #
    #   hi =  2^(num_bits - 1) - 1                            #
    #   Hint: use 2.0 ** (num_bits - 1) for float arithmetic  #
    #                                                         #
    # Step 2: Scale weights to integer-grid space             #
    #   scaled = 2^(-e) * weights                             #
    #   Hint: 2**(-e) works when e is a tensor                #
    #                                                         #
    # Step 3: Clamp to valid integer range                    #
    #   clamped = scaled.clamp(lo, hi)                        #
    #                                                         #
    # Step 4: Apply STE rounding                              #
    #   rounded = ste_round(clamped)                          #
    #                                                         #
    # Step 5: Dequantize back to float space                  #
    #   return 2^e * rounded                                  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


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

    Minimizes MSE(quantize(weights, e, num_bits), target) over 'e'.

    The STE enables gradients to flow through ste_round() back to 'e',
    even though rounding is non-differentiable. This same mechanism is what
    makes learning bit-widths 'b' possible in the full self-compression setup.

    Parameters
    ----------
    weights : torch.Tensor
        Fixed weight tensor (NOT updated — only 'e' is a parameter).
        Realistic Conv2d weights (Kaiming-initialized).
    target : torch.Tensor
        Target: weights quantized at the known-optimal exponent.
        Same shape as weights. The optimizer should drive e to match this.
    num_bits : int
        Fixed bit-width for the quantization range.
    steps : int
        Number of Adam gradient steps to take.
    lr : float
        Learning rate for Adam optimizer.
    log_every : int
        Print e value and loss every log_every steps.

    Returns
    -------
    e : nn.Parameter
        The learned exponent after training.
    e_history : list[float]
        Value of e at each training step (length = steps).
    loss_history : list[float]
        MSE loss at each training step (length = steps).

    Notes
    -----
    Initialize e at 0.0 (a neutral starting point, far from the typical
    optimal value around -8.0 for Kaiming-initialized weights).
    Use Adam optimizer with the given lr.
    At each step: zero_grad → quantize → MSE loss vs target → backward → step.
    Record e.item() and loss.item() at each step BEFORE the optimizer step.
    """
    ###########################################################
    # YOUR CODE HERE - 15-20 lines                            #
    #                                                         #
    # Step 1: Initialize e as nn.Parameter at 0.0             #
    #   e = nn.Parameter(torch.tensor(0.0))                   #
    #                                                         #
    # Step 2: Create Adam optimizer for [e]                   #
    #   optimizer = torch.optim.Adam([e], lr=lr)              #
    #                                                         #
    # Step 3: Training loop for 'steps' iterations:           #
    #   a. optimizer.zero_grad()                              #
    #   b. w_q = quantize_with_learnable_e(weights, e,        #
    #                                      num_bits)          #
    #   c. loss = F.mse_loss(w_q, target)                     #
    #   d. Record e.item() and loss.item() to history lists   #
    #   e. loss.backward()                                    #
    #   f. optimizer.step()                                   #
    #   g. If step % log_every == 0 or step == steps-1:       #
    #      print(f"  step {step:4d}: e={e.item():.4f},        #
    #              loss={loss.item():.8f}")                    #
    #                                                         #
    # Step 4: Return e, e_history, loss_history               #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Main: learn exponent for realistic Conv2d weights (DO NOT MODIFY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Exercise 3: STE with Learnable Quantization Parameters")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Set up: realistic Kaiming-initialized Conv2d weights
    # A 3×3 conv with 32 input channels → 64 output channels
    # Uses Kaiming uniform init (std ≈ 1/sqrt(in_channels * kH * kW))
    # -----------------------------------------------------------------------
    in_channels, out_channels, k = 32, 64, 3
    fan_in = in_channels * k * k
    scale = 1.0 / math.sqrt(fan_in)
    weights = torch.empty(out_channels, in_channels, k, k).uniform_(-scale, scale)
    weights = weights.flatten()  # work with 1D for simplicity

    # The known-optimal exponent for these weights
    # For Kaiming uniform weights with fan_in=288: weight range ≈ [-0.059, 0.059]
    # To quantize at 2 bits (integers in {-2,-1,0,1}), the optimal scale maps
    # max(|w|) ≈ 0.059 to ≈ 1.0, so 2^e ≈ 0.059, meaning e ≈ log2(0.059) ≈ -4.08
    # But with 4 bits we get more precision: target_e ≈ log2(0.059/7) ≈ -6.9
    # We use 4 bits and compute target_e empirically below.
    num_bits = 4

    # Compute the target: weights quantized at the exact optimal exponent
    # The optimal e makes max(|weights|) map to 2^(b-1)-1
    max_abs = weights.abs().max().item()
    max_int = 2 ** (num_bits - 1) - 1   # = 7 for 4-bit
    target_e = math.log2(max_abs / max_int)   # e such that 2^e * max_int ≈ max_abs

    print(f"Weight stats:  mean={weights.mean():.4f}, std={weights.std():.4f}")
    print(f"               max|w|={max_abs:.6f}")
    print(f"Num bits:      {num_bits}  (range: {-2**(num_bits-1)} to {2**(num_bits-1)-1})")
    print(f"Target e:      {target_e:.4f}  (optimal grid for these weights)")
    print(f"Initial e:     0.0000  (neutral starting point)")
    print()

    # Pre-compute the target quantized weights at optimal exponent
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

    # Run the training loop
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

    # Evaluate convergence
    final_e = e_learned.item()
    final_loss = loss_history[-1]
    converged = abs(final_e - target_e) < 0.5 and final_loss < 1e-4

    print(f"Learned e  = {final_e:.4f}")
    print(f"Target e   = {target_e:.4f}")
    print(f"Difference = {abs(final_e - target_e):.4f}")
    print(f"Final MSE  = {final_loss:.2e}")
    print(f"converged: {converged}")
    print()

    # Verify that e moved meaningfully from its starting point
    e_range = max(e_history) - min(e_history)
    print(f"e trajectory range: {min(e_history):.4f} → {max(e_history):.4f}  (range={e_range:.4f})")
    assert e_range > 1.0, \
        f"e barely moved ({e_range:.4f}) — gradient not flowing through STE!"
    assert final_loss < 1e-3, \
        f"MSE too high ({final_loss:.2e}) — e did not converge!"
    assert converged, \
        f"e={final_e:.4f} did not converge near target {target_e:.4f}"

    print()
    print("KEY INSIGHT:")
    print("  The STE enabled gradient flow through ste_round() to 'e'.")
    print("  Without STE, gradient of e would be 0 and it would never move.")
    print("  This same mechanism enables learning bit-widths 'b' in Module 2.")

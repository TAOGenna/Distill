"""
Exercise 1: Why Rounding Kills Gradients
=========================================
Course: Self-Compressing Neural Networks
Module: 1 — The Straight-Through Estimator

GOAL
----
Demonstrate the core problem: torch.round() produces zero gradients because
it is a piecewise constant function (staircase). Then show that the
Straight-Through Estimator (STE) fixes this by replacing the zero gradient
with an identity pass in the backward pass.

You will implement:
  1. naive_round_forward  — straightforward quantization with torch.round()
  2. ste_round_forward    — same forward value but with gradient bypass
  3. train_and_log        — training loop that records loss and gradient magnitudes

After running both approaches for 200 steps each you should observe:
  - Naive:  loss stuck near initial value, gradients zero after step 0
  - STE:    loss decreasing smoothly, gradients nonzero throughout

DEPENDENCIES
------------
  pip install torch  # PyTorch >= 2.0
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Part 1: Forward functions
# ---------------------------------------------------------------------------

def naive_round_forward(x: torch.Tensor, scale: nn.Parameter) -> torch.Tensor:
    """Quantize x by multiplying by scale and rounding. No gradient tricks.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be quantized (shape: any).
    scale : nn.Parameter
        Learnable scale factor. The quantization grid spacing is 1/scale,
        so larger scale = finer grid.

    Returns
    -------
    torch.Tensor
        Rounded (quantized) tensor. Same shape as x.
        Gradient of scale through this operation is ZERO (piecewise constant).

    Example
    -------
    >>> x = torch.tensor([0.73])
    >>> scale = nn.Parameter(torch.tensor([1.0]))
    >>> naive_round_forward(x, scale)
    tensor([1.])
    """
    ###########################################################
    # YOUR CODE HERE - 3-5 lines                              #
    #                                                         #
    # Hint: multiply x by scale, then apply torch.round().   #
    # Do NOT use any detach() or special gradient tricks.     #
    # This is intentionally broken for learning purposes.     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def ste_round_forward(x: torch.Tensor, scale: nn.Parameter) -> torch.Tensor:
    """Quantize x using the Straight-Through Estimator for rounding.

    Forward pass:  returns round(scale * x)  — exact rounding, no approximation
    Backward pass: gradient flows as if rounding never happened (identity)

    The STE trick uses the identity:
        (expr.round() - expr).detach() + expr
    which equals expr.round() in the forward pass, but has gradient equal to
    the gradient of expr in the backward pass (detach blocks the subtracted term).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be quantized (shape: any).
    scale : nn.Parameter
        Learnable scale factor. The quantization grid spacing is 1/scale.

    Returns
    -------
    torch.Tensor
        Rounded (quantized) tensor. Same shape as x.
        Gradient of scale through this operation is NONZERO (STE identity pass).

    Example
    -------
    >>> x = torch.tensor([0.73])
    >>> scale = nn.Parameter(torch.tensor([1.0]))
    >>> y = ste_round_forward(x, scale)
    >>> y          # same as round(scale * x) = 1.0
    tensor([1.])
    >>> loss = y.sum(); loss.backward()
    >>> scale.grad  # nonzero! STE passed gradient through
    tensor([0.73])
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                              #
    #                                                         #
    # Step 1: compute scaled = scale * x                      #
    # Step 2: apply STE: (scaled.round() - scaled).detach()  #
    #         + scaled                                        #
    # Step 3: return the result                               #
    #                                                         #
    # Hint: the result must equal scaled.round() in value,   #
    # but must have nonzero gradient with respect to scale.   #
    # The key is that .detach() removes the gradient-killing  #
    # round() from the backward graph, leaving only the +     #
    # scaled term which contributes gradient = x.             #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Part 2: Training loop
# ---------------------------------------------------------------------------

def train_and_log(
    forward_fn,
    steps: int = 200,
    lr: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Run a training loop using the given forward function, log loss and grad.

    The setup:
      - Input x: fixed tensor of 8 values uniformly spaced in [-1, 1]
      - Target: x quantized at scale=4.0 (optimal answer: scale converges to 4.0)
      - Learnable: scale (nn.Parameter), initialized to 1.0
      - Loss: MSE between forward_fn(x, scale) and target
      - Optimizer: SGD with given lr

    The goal is for 'scale' to learn the value 4.0 (which produces the target
    quantization). With naive rounding, scale never moves. With STE, it converges.

    Parameters
    ----------
    forward_fn : callable
        Either naive_round_forward or ste_round_forward. Signature:
        forward_fn(x: Tensor, scale: nn.Parameter) -> Tensor
    steps : int
        Number of gradient descent steps to take.
    lr : float
        Learning rate for SGD optimizer.

    Returns
    -------
    losses : list[float]
        MSE loss value at each step.
    grad_mags : list[float]
        Absolute value of scale.grad at each step (0.0 if grad is None).

    Notes
    -----
    The input x and target are fixed throughout training. Only scale is updated.
    Log the loss and gradient magnitude at EVERY step (not just every N steps).
    """
    # Fixed input: 8 values in [-1, 1]
    x = torch.linspace(-1.0, 1.0, 8)

    # Target: what we want scale to produce (quantized at scale=4.0)
    true_scale = 4.0
    with torch.no_grad():
        target = torch.round(true_scale * x)

    # Learnable scale, initialized far from the true value
    scale = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([scale], lr=lr)

    losses = []
    grad_mags = []

    ###########################################################
    # YOUR CODE HERE - 12-15 lines                            #
    #                                                         #
    # For each step:                                          #
    #   1. optimizer.zero_grad()                              #
    #   2. Call forward_fn(x, scale) to get quantized output  #
    #   3. Compute MSE loss vs target                         #
    #   4. loss.backward()                                    #
    #   5. Record loss.item() and abs(scale.grad.item())      #
    #      (if scale.grad is None, record 0.0)                #
    #   6. optimizer.step()                                   #
    #                                                         #
    # Hint: use F.mse_loss(output, target) for the loss, or  #
    # compute ((output - target)**2).mean() directly.         #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################

    return losses, grad_mags


# ---------------------------------------------------------------------------
# Main: compare naive vs STE (DO NOT MODIFY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(42)

    print("=" * 60)
    print("Exercise 1: Why Rounding Kills Gradients")
    print("=" * 60)
    print()

    # Run both approaches
    naive_losses, naive_grads = train_and_log(naive_round_forward, steps=200, lr=0.05)
    ste_losses, ste_grads = train_and_log(ste_round_forward, steps=200, lr=0.05)

    # Print comparison table
    print(f"{'Step':>6}  {'Naive Loss':>12}  {'Naive |grad|':>14}  {'STE Loss':>10}  {'STE |grad|':>12}")
    print("-" * 62)
    for step in [0, 1, 5, 10, 25, 50, 100, 150, 199]:
        print(
            f"{step:>6}  "
            f"{naive_losses[step]:>12.6f}  "
            f"{naive_grads[step]:>14.6f}  "
            f"{ste_losses[step]:>10.6f}  "
            f"{ste_grads[step]:>12.6f}"
        )

    print()
    print("SUMMARY")
    print("-" * 40)
    final_naive = naive_losses[-1]
    final_ste   = ste_losses[-1]
    print(f"Naive final loss:  {final_naive:.6f}  (should be ~0.3 or higher, stuck)")
    print(f"STE   final loss:  {final_ste:.6f}  (should be < 0.01, converged)")
    print()

    naive_nonzero = sum(1 for g in naive_grads if g > 1e-8)
    ste_nonzero   = sum(1 for g in ste_grads   if g > 1e-8)
    print(f"Naive steps with nonzero gradient: {naive_nonzero}/200")
    print(f"STE   steps with nonzero gradient: {ste_nonzero}/200")
    print()

    # Sanity checks
    assert final_ste < final_naive, \
        f"STE loss ({final_ste:.4f}) should be < naive loss ({final_naive:.4f})"
    assert final_ste < 0.05, \
        f"STE should converge below 0.05, got {final_ste:.4f}"
    assert ste_nonzero > 15, \
        f"STE should have nonzero gradients while learning, got {ste_nonzero}"
    assert naive_nonzero < 5, \
        f"Naive should have zero gradients almost always, got {naive_nonzero}"

    print("STE loss:", f"{final_ste:.6f}")
    print("Naive loss:", f"{final_naive:.6f}")
    print()
    print("All assertions passed! STE enables gradient flow, naive does not.")

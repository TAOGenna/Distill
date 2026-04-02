"""
Exercise 1: Why Rounding Kills Gradients — SOLUTION
=====================================================
Course: Self-Compressing Neural Networks
Module: 1 — The Straight-Through Estimator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Learnable scale factor.

    Returns
    -------
    torch.Tensor
        Rounded (quantized) tensor. Gradient of scale is ZERO.
    """
    scaled = scale * x
    return torch.round(scaled)


def ste_round_forward(x: torch.Tensor, scale: nn.Parameter) -> torch.Tensor:
    """Quantize x using the Straight-Through Estimator for rounding.

    Forward pass:  returns round(scale * x)  — exact rounding
    Backward pass: gradient flows as if rounding never happened (identity)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be quantized (shape: any).
    scale : nn.Parameter
        Learnable scale factor.

    Returns
    -------
    torch.Tensor
        Rounded tensor with STE gradient. Gradient of scale is NONZERO.
    """
    scaled = scale * x
    # STE trick: forward value = scaled.round()
    # backward gradient = gradient of scaled (identity through the round)
    return (scaled.round() - scaled).detach() + scaled


# ---------------------------------------------------------------------------
# Part 2: Training loop
# ---------------------------------------------------------------------------

def train_and_log(
    forward_fn,
    steps: int = 200,
    lr: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Run a training loop, log loss and gradient magnitudes.

    Parameters
    ----------
    forward_fn : callable
        Either naive_round_forward or ste_round_forward.
    steps : int
        Number of gradient descent steps.
    lr : float
        Learning rate for SGD.

    Returns
    -------
    losses : list[float]
        MSE loss at each step.
    grad_mags : list[float]
        |scale.grad| at each step.
    """
    # Fixed input and target
    x = torch.linspace(-1.0, 1.0, 8)
    true_scale = 4.0
    with torch.no_grad():
        target = torch.round(true_scale * x)

    # Learnable parameter starting far from truth
    scale = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([scale], lr=lr)

    losses = []
    grad_mags = []

    for step in range(steps):
        optimizer.zero_grad()

        output = forward_fn(x, scale)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Record before step (so we see the gradient that drove the update)
        losses.append(loss.item())
        grad_mag = abs(scale.grad.item()) if scale.grad is not None else 0.0
        grad_mags.append(grad_mag)

        optimizer.step()

    return losses, grad_mags


# ---------------------------------------------------------------------------
# Main (DO NOT MODIFY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Exercise 1: Why Rounding Kills Gradients")
    print("=" * 60)
    print()

    naive_losses, naive_grads = train_and_log(naive_round_forward, steps=200, lr=0.05)
    ste_losses, ste_grads = train_and_log(ste_round_forward, steps=200, lr=0.05)

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
    print(f"Naive final loss:  {final_naive:.6f}  (stuck)")
    print(f"STE   final loss:  {final_ste:.6f}  (converged)")
    print()

    naive_nonzero = sum(1 for g in naive_grads if g > 1e-8)
    ste_nonzero   = sum(1 for g in ste_grads   if g > 1e-8)
    print(f"Naive steps with nonzero gradient: {naive_nonzero}/200")
    print(f"STE   steps with nonzero gradient: {ste_nonzero}/200")
    print()

    assert final_ste < final_naive, f"STE ({final_ste:.4f}) should be < naive ({final_naive:.4f})"
    assert final_ste < 0.05, f"STE should converge, got {final_ste:.4f}"
    assert ste_nonzero > 15, f"STE should have nonzero grads while learning, got {ste_nonzero}"
    assert naive_nonzero < 5, f"Naive should always have zero grads, got {naive_nonzero}"

    print("STE loss:", f"{final_ste:.6f}")
    print("Naive loss:", f"{final_naive:.6f}")
    print()
    print("All assertions passed! STE enables gradient flow, naive does not.")

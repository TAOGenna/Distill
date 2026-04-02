"""
Exercise 1: The Compression Loss
=================================
Module 3 — Self-Compression Training on MNIST

The core training objective of Self-Compressing Neural Networks:

    L = L_task + lambda * Q

where Q = avg bits per weight (differentiable w.r.t. each channel's b parameter).

The compression term Q is computed by summing qbits() across all QConv2d layers
and dividing by the total weight count. Because relu(b) appears in the computation,
Q is differentiable: d(Q)/d(b_c) = fan_in / N_weights when b_c > 0.

This creates NEGATIVE pressure on b during minimization:
  * At lambda=0  : b only moves if task loss gradient drives it
  * At lambda>0  : b is pushed toward 0 (cheaper to store)
  * At lambda=1.0: strong compression pressure, most channels pruned

Your tasks:
  1. compute_compression_term(model) — compute Q = total_bits / total_weights
  2. self_compression_loss(model, logits, targets, lam) — combine task + Q
  3. verify_compression_gradient(model, lam_values) — test gradient directions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# QConv2d — from Module 2 (provided, do NOT modify)
# ──────────────────────────────────────────────────────────────────────────────

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for rounding.

    Forward:  x.round()
    Backward: gradient flows as if round() was identity
    """
    return (x.round() - x).detach() + x


class QConv2d(nn.Module):
    """Quantization-Aware Conv2d with per-channel learnable bit-widths.

    Parameters per output channel:
      e  (float): log-scale exponent, controls quantization resolution
      b  (float): bit-width, controls range (number of discrete values)

    Both are learnable nn.Parameters updated by gradient descent.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        scale = 1.0 / math.sqrt(fan_in)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size).uniform_(-scale, scale)
        )
        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), -8.0))
        self.b = nn.Parameter(torch.full((out_channels, 1, 1, 1),  2.0))

    def qweight(self) -> torch.Tensor:
        """Quantized weight (pre-STE, continuous)."""
        eff_b = torch.relu(self.b)
        lower = -(2 ** (eff_b - 1))
        upper =  (2 ** (eff_b - 1)) - 1
        return torch.clamp(2 ** (-self.e) * self.weight, lower, upper)

    def qbits(self) -> torch.Tensor:
        """Total bits needed to store this layer's weights.

        Returns
        -------
        torch.Tensor
            Scalar, differentiable w.r.t. self.b.
            = relu(b).sum() * fan_in
        """
        fan_in = math.prod(self.weight.shape[1:])
        return torch.relu(self.b).sum() * fan_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self.qweight()
        w = ste_round(qw)
        dw = (2 ** self.e) * w
        return F.conv2d(x, dw, stride=self.stride, padding=self.padding)


# ──────────────────────────────────────────────────────────────────────────────
# Small test model (2 QConv2d layers)
# ──────────────────────────────────────────────────────────────────────────────

class TinyQModel(nn.Module):
    """Two-layer QConv2d model for testing the compression loss.

    Architecture: conv(1→8, k=3) → relu → conv(8→4, k=1) → global avg → logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = QConv2d(1, 8, 3, padding=1)
        self.conv2 = QConv2d(8, 4, 1)
        self.fc    = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=[2, 3])   # global average pool
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# YOUR CODE — implement the three functions below
# ──────────────────────────────────────────────────────────────────────────────

def compute_compression_term(model: nn.Module) -> torch.Tensor:
    """Compute the average bits per weight Q across all QConv2d layers.

    Q = (sum of qbits() for each QConv2d layer) / (total weight parameters)

    This is the compression metric from the paper. It is differentiable
    with respect to each layer's b parameter (via relu in qbits()).

    Parameters
    ----------
    model : nn.Module
        A model containing one or more QConv2d layers.

    Returns
    -------
    torch.Tensor
        Scalar tensor Q (avg bits/weight). Gradient flows back to all b params.

    Notes
    -----
    - Use model.modules() to iterate over all submodules recursively.
    - Divide total bits by weight_count (not channel count!) to normalize.
    - weight_count = sum of layer.weight.numel() for each QConv2d layer.
    - Do NOT call .item() — this breaks the gradient graph.
    """
    ###########################################################################
    # YOUR CODE HERE — 8-12 lines                                             #
    #                                                                         #
    # Hint: collect two sums:                                                 #
    #   1. total_bits = sum of layer.qbits() for each QConv2d                #
    #   2. weight_count = sum of layer.weight.numel() for each QConv2d       #
    # Then return total_bits / weight_count                                   #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


def self_compression_loss(
    model: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, float, float]:
    """Compute the combined self-compression training loss.

    L = L_task + lambda * Q

    where:
      L_task = F.cross_entropy(logits, targets)   (classification loss)
      Q      = compute_compression_term(model)     (avg bits per weight)
      lambda = lam                                 (compression strength)

    Parameters
    ----------
    model : nn.Module
        The model (for computing Q via its QConv2d layers).
    logits : torch.Tensor
        Shape (B, num_classes). Raw classification scores.
    targets : torch.Tensor
        Shape (B,). Integer class labels.
    lam : float
        Compression regularization strength. Paper uses 0.05.

    Returns
    -------
    tuple[torch.Tensor, float, float]
        (total_loss, task_loss_value, Q_value)
        total_loss : differentiable Tensor for .backward()
        task_loss_value : float (for logging)
        Q_value : float (avg bits per weight, for logging)

    Notes
    -----
    - Return Q as a float in the tuple (Q.item()), but the total_loss tensor
      must remain live (no .item() on it!) for backward() to work.
    - lam * Q must be added BEFORE converting anything to float.
    """
    ###########################################################################
    # YOUR CODE HERE — 6-8 lines                                              #
    #                                                                         #
    # 1. task_loss = F.cross_entropy(logits, targets)                         #
    # 2. Q = compute_compression_term(model)                                  #
    # 3. total_loss = task_loss + lam * Q                                     #
    # 4. return (total_loss, task_loss.item(), Q.item())                      #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


def verify_compression_gradient(
    model: nn.Module,
    lam_values: List[float],
) -> None:
    """For each lambda, compute the full loss and inspect b.grad.

    This function demonstrates that:
      - At lam=0: b.grad direction is controlled by task loss only
      - At lam>0: b.grad has a NEGATIVE component (compression pushes b down)
      - Higher lam → more negative b.grad → stronger compression pressure

    Parameters
    ----------
    model : nn.Module
        A TinyQModel (or any model with QConv2d layers).
    lam_values : list[float]
        List of lambda values to test. E.g. [0.0, 0.01, 0.05, 0.1, 1.0].

    Returns
    -------
    None
        Prints a table: lambda | Q | task_loss | total_loss | mean(b.grad)
        Ends with: 'Compression gradient verified: higher lambda -> stronger b reduction pressure'

    Notes
    -----
    - You need fresh random data each iteration (or reuse — doesn't matter).
    - IMPORTANT: zero all parameter gradients before each backward() call
      using model.zero_grad(). Otherwise gradients accumulate.
    - Access b.grad from QConv2d layers via model.modules().
    - Collect all b.grad tensors from all QConv2d layers into one list,
      then compute the mean of their concatenated values.
    - The 'mean b.grad' should become more negative as lam increases.
    """
    ###########################################################################
    # YOUR CODE HERE — 10-15 lines                                            #
    #                                                                         #
    # Setup: create random input (B=16, C=1, H=8, W=8) and targets (B=16)   #
    # For each lam in lam_values:                                             #
    #   1. model.zero_grad()                                                  #
    #   2. logits = model(x)                                                  #
    #   3. loss, task_val, Q_val = self_compression_loss(model, logits, y, l) #
    #   4. loss.backward()                                                    #
    #   5. Collect b.grad from each QConv2d in model.modules()               #
    #   6. Compute mean_b_grad = mean of all b gradients                     #
    #   7. Print: f"lam={lam:.3f} | Q={Q_val:.3f} | task={task_val:.3f} | " #
    #             f"total={loss.item():.3f} | mean_b.grad={mean_b_grad:.4f}" #
    # After the loop, print the verification message                          #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ──────────────────────────────────────────────────────────────────────────────
# Main test harness — DO NOT MODIFY
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 65)
    print("Exercise 1: The Compression Loss")
    print("=" * 65)

    model = TinyQModel(num_classes=10)

    # ── Sanity check: Q at initialization ───────────────────────────────────
    print("\n[1] Compression term at initialization (b=2.0 everywhere):")
    Q_init = compute_compression_term(model)
    print(f"    Q = {Q_init.item():.4f} bits/weight  (expected ~2.0)")
    assert abs(Q_init.item() - 2.0) < 0.1, \
        f"Expected Q ≈ 2.0, got {Q_init.item():.4f}"

    # ── Sanity check: Q is differentiable ───────────────────────────────────
    print("\n[2] Verifying Q is differentiable w.r.t. b:")
    Q = compute_compression_term(model)
    Q.backward()
    for name, layer in model.named_modules():
        if isinstance(layer, QConv2d):
            assert layer.b.grad is not None, f"b.grad is None for {name}!"
            print(f"    {name}.b.grad shape: {layer.b.grad.shape}, "
                  f"mean: {layer.b.grad.mean().item():.4f}  ✓")
    model.zero_grad()

    # ── Test self_compression_loss ───────────────────────────────────────────
    print("\n[3] Testing self_compression_loss with lambda=0.05:")
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 10, (16,))
    logits = model(x)
    loss, task_val, Q_val = self_compression_loss(model, logits, y, lam=0.05)
    print(f"    task_loss={task_val:.4f}, Q={Q_val:.4f}, total_loss={loss.item():.4f}")
    assert loss.requires_grad, "total_loss must be a live tensor (requires_grad=True)"
    assert abs(loss.item() - (task_val + 0.05 * Q_val)) < 0.01, \
        "total_loss != task_loss + 0.05 * Q"
    print("    ✓ loss composition verified")

    # ── Main gradient table ──────────────────────────────────────────────────
    print("\n[4] Gradient direction vs lambda:")
    print(f"    {'lambda':>7} | {'Q':>6} | {'task':>8} | {'total':>8} | {'mean b.grad':>12}")
    print("    " + "-" * 55)
    verify_compression_gradient(model, lam_values=[0.0, 0.01, 0.05, 0.1, 1.0])

    print("\n✓ All checks passed!")

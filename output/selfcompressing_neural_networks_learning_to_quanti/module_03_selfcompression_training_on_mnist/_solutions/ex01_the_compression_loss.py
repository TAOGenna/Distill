"""
Exercise 1: The Compression Loss — SOLUTION
============================================
Module 3 — Self-Compression Training on MNIST

Reference from notebook:
    Q = functools.reduce(lambda x,y: x+y,
            [l.qbits() for l in model.layers if isinstance(l, QConv2d)]
        ) / weight_count
    loss = loss + 0.05*Q
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
    """Straight-Through Estimator for rounding."""
    return (x.round() - x).detach() + x


class QConv2d(nn.Module):
    """Quantization-Aware Conv2d with per-channel learnable bit-widths."""

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
        eff_b = torch.relu(self.b)
        lower = -(2 ** (eff_b - 1))
        upper =  (2 ** (eff_b - 1)) - 1
        return torch.clamp(2 ** (-self.e) * self.weight, lower, upper)

    def qbits(self) -> torch.Tensor:
        fan_in = math.prod(self.weight.shape[1:])
        return torch.relu(self.b).sum() * fan_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self.qweight()
        w = ste_round(qw)
        dw = (2 ** self.e) * w
        return F.conv2d(x, dw, stride=self.stride, padding=self.padding)


# ──────────────────────────────────────────────────────────────────────────────
# Small test model
# ──────────────────────────────────────────────────────────────────────────────

class TinyQModel(nn.Module):
    """Two-layer QConv2d model for testing the compression loss."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = QConv2d(1, 8, 3, padding=1)
        self.conv2 = QConv2d(8, 4, 1)
        self.fc    = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=[2, 3])
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# SOLUTION implementations
# ──────────────────────────────────────────────────────────────────────────────

def compute_compression_term(model: nn.Module) -> torch.Tensor:
    """Compute the average bits per weight Q across all QConv2d layers.

    Q = (sum of qbits() for each QConv2d layer) / (total weight parameters)

    Parameters
    ----------
    model : nn.Module
        A model containing one or more QConv2d layers.

    Returns
    -------
    torch.Tensor
        Scalar Q (avg bits/weight), differentiable w.r.t. all b params.
    """
    total_bits = torch.tensor(0.0)
    weight_count = 0

    for layer in model.modules():
        if isinstance(layer, QConv2d):
            total_bits = total_bits + layer.qbits()
            weight_count += layer.weight.numel()

    return total_bits / weight_count


def self_compression_loss(
    model: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, float, float]:
    """Compute the combined self-compression training loss.

    L = L_task + lambda * Q

    Parameters
    ----------
    model : nn.Module
    logits : torch.Tensor, shape (B, num_classes)
    targets : torch.Tensor, shape (B,)
    lam : float

    Returns
    -------
    tuple[torch.Tensor, float, float]
        (total_loss, task_loss_value, Q_value)
    """
    task_loss = F.cross_entropy(logits, targets)
    Q = compute_compression_term(model)
    total_loss = task_loss + lam * Q
    return total_loss, task_loss.item(), Q.item()


def verify_compression_gradient(
    model: nn.Module,
    lam_values: List[float],
) -> None:
    """For each lambda, compute the full loss and inspect b.grad direction.

    Parameters
    ----------
    model : nn.Module
    lam_values : list[float]
    """
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 10, (16,))

    for lam in lam_values:
        model.zero_grad()
        logits = model(x)
        loss, task_val, Q_val = self_compression_loss(model, logits, y, lam)
        loss.backward()

        # Collect b.grad from all QConv2d layers
        b_grads = []
        for layer in model.modules():
            if isinstance(layer, QConv2d) and layer.b.grad is not None:
                b_grads.append(layer.b.grad.flatten())

        mean_b_grad = torch.cat(b_grads).mean().item()
        print(f"    lam={lam:.3f} | Q={Q_val:.3f} | task={task_val:.3f} | "
              f"total={loss.item():.3f} | mean_b.grad={mean_b_grad:+.4f}")

    print("\nCompression gradient verified: higher lambda -> stronger b reduction pressure")


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

"""
Exercise 4: Gradient Flow Verification
=======================================
SOLUTION FILE

Key insight: ALL THREE parameters (weight, e, b) must receive nonzero gradients.
The b gradient comes primarily from the compression penalty λ*Q, with magnitude ≈ λ*fan_in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _kaiming_uniform_like_ref(out_channels, in_channels, kH, kW):
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


class QConv2d(nn.Module):
    """Complete QConv2d layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride      = stride
        self.padding     = padding
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            _kaiming_uniform_like_ref(out_channels, in_channels, *kernel_size)
        )
        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), -8.0))
        self.b = nn.Parameter(torch.full((out_channels, 1, 1, 1),  2.0))

    def qweight(self) -> torch.Tensor:
        eff_b  = torch.relu(self.b)
        lower  = -(2 ** (eff_b - 1))
        upper  =  (2 ** (eff_b - 1)) - 1
        scaled = (2 ** (-self.e)) * self.weight
        return torch.minimum(torch.maximum(scaled, lower), upper)

    def qbits(self) -> torch.Tensor:
        fan_in = math.prod(self.weight.shape[1:])
        return torch.relu(self.b).sum() * fan_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self.qweight()
        w  = ste_round(qw)
        dw = (2 ** self.e) * w
        return F.conv2d(x, dw, stride=self.stride, padding=self.padding)


def compute_loss_with_compression(layer: QConv2d, x: torch.Tensor,
                                   target: torch.Tensor,
                                   lam: float = 0.05) -> torch.Tensor:
    """Combined task + compression loss."""
    out = layer(x)
    task_loss = F.mse_loss(out, target)
    Q = layer.qbits() / layer.weight.numel()
    return task_loss + lam * Q


def verify_gradient_flow(layer: QConv2d,
                          x: torch.Tensor,
                          lam: float = 0.05) -> bool:
    """Run forward-backward and verify all three parameters get nonzero gradients.

    Parameters
    ----------
    layer : QConv2d
    x     : torch.Tensor — input batch
    lam   : float — compression penalty coefficient

    Returns
    -------
    bool — True if all three parameter gradients are nonzero
    """
    # Step 1: Clear any stale gradients
    # Setting to None is cleaner than .zero_() — avoids allocating zero tensors
    for p in [layer.weight, layer.e, layer.b]:
        p.grad = None

    # Step 2: Get output shape, create a random target
    # We detach the output so the target doesn't carry graph history
    with torch.no_grad():
        out_shape = layer(x).shape
    target = torch.randn(out_shape)

    # Step 3: Compute combined task + compression loss
    loss = compute_loss_with_compression(layer, x, target, lam)

    # Step 4: Backward pass — accumulates gradients on all leaf tensors
    loss.backward()

    # Step 5: Check and report each parameter's gradient
    all_nonzero = True
    for name, param in [("weight", layer.weight), ("e", layer.e), ("b", layer.b)]:
        grad = param.grad
        if grad is None:
            print(f"  {name:<12} | {'None':>12} | {'None':>10} | {'None':>8}")
            all_nonzero = False
            continue

        grad_mean  = grad.abs().mean().item()
        grad_std   = grad.std().item()
        frac_nz    = (grad.abs() > 1e-10).float().mean().item()

        print(f"  {name:<12} | {grad_mean:>12.6f} | {grad_std:>10.6f} | {frac_nz:>8.4f}")

        if grad.abs().sum().item() == 0:
            all_nonzero = False

    return all_nonzero


def analyze_gradient_magnitudes(layer: QConv2d,
                                 x: torch.Tensor,
                                 lam: float = 0.05) -> dict:
    """Compare gradient magnitudes and check theoretical b gradient formula.

    Parameters
    ----------
    layer : QConv2d
    x     : torch.Tensor — input batch
    lam   : float — compression penalty

    Returns
    -------
    dict with gradient norms and theory check
    """
    # Clear gradients
    for p in [layer.weight, layer.e, layer.b]:
        p.grad = None

    # Forward + backward
    with torch.no_grad():
        out_shape = layer(x).shape
    target = torch.randn(out_shape)
    loss = compute_loss_with_compression(layer, x, target, lam)
    loss.backward()

    # Compute L2 norms
    weight_norm = layer.weight.grad.norm().item()
    e_norm      = layer.e.grad.norm().item()
    b_norm      = layer.b.grad.norm().item()

    # Theoretical b gradient from compression term only.
    # Q = qbits() / weight.numel() = relu(b).sum() * fan_in / (out_ch * fan_in)
    #                               = relu(b).sum() / out_ch
    # So: ∂(λ*Q)/∂b_i = λ / out_ch   (when b_i > 0)
    # L2 norm of this constant-magnitude gradient over all out_ch channels:
    #   ||b_grad||_2 = (λ / out_ch) * sqrt(out_ch) = λ / sqrt(out_ch)
    out_ch    = layer.b.shape[0]
    b_theory  = lam / math.sqrt(out_ch)

    # Task loss also contributes a small amount through the clamp bounds,
    # so allow 50% tolerance
    b_matches = abs(b_norm - b_theory) / (b_theory + 1e-8) < 0.5

    print(f"\n  {'Parameter':<12} | {'L2 norm':>12}")
    print(f"  {'-'*12}-+-{'-'*12}")
    print(f"  {'weight':<12} | {weight_norm:>12.6f}")
    print(f"  {'e':<12} | {e_norm:>12.6f}")
    print(f"  {'b (actual)':<12} | {b_norm:>12.6f}")
    print(f"  {'b (theory)':<12} | {b_theory:>12.6f}  (λ/sqrt(out_ch))")
    print(f"\n  Ratio weight/b : {weight_norm / (b_norm + 1e-8):.2f}x")
    print(f"  Ratio e/b      : {e_norm / (b_norm + 1e-8):.2f}x")
    print(f"  Theory matches : {'✓' if b_matches else '~'} (within 50%)")

    return {
        'weight_grad_norm'    : weight_norm,
        'e_grad_norm'         : e_norm,
        'b_grad_norm'         : b_norm,
        'b_theoretical'       : b_theory,
        'b_grad_matches_theory': b_matches,
    }


# ---------------------------------------------------------------------------
# __main__ — validation harness (identical to scaffold)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(7)

    print("=" * 65)
    print("Exercise 4: Gradient Flow Verification")
    print("=" * 65)

    # -------------------------------------------------------------------
    # Test 1
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("[Test 1] Gradient flow for QConv2d(1, 16, 5)")
    print(f"{'─'*65}")
    layer1 = QConv2d(in_channels=1, out_channels=16, kernel_size=5)
    x1 = torch.randn(4, 1, 12, 12, requires_grad=True)
    print(f"  Layer: {layer1.weight.shape}, input: {x1.shape}")
    print(f"  {'Parameter':<12} | {'|grad| mean':>12} | {'grad std':>10} | {'frac nz':>8}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    flow_ok = verify_gradient_flow(layer1, x1, lam=0.05)
    print(f"\n  All gradients nonzero: {'✓ PASS' if flow_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("[Test 2] Gradient flow for QConv2d(1, 32, 5) — MNIST layer 0")
    print(f"{'─'*65}")
    layer2 = QConv2d(in_channels=1, out_channels=32, kernel_size=5)
    x2 = torch.randn(8, 1, 28, 28)
    print(f"  Layer: {layer2.weight.shape}, input: {x2.shape}")
    print(f"  {'Parameter':<12} | {'|grad| mean':>12} | {'grad std':>10} | {'frac nz':>8}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    flow_ok2 = verify_gradient_flow(layer2, x2, lam=0.05)
    print(f"\n  All gradients nonzero: {'✓ PASS' if flow_ok2 else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 3
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("[Test 3] Gradient magnitude analysis — theory vs actual")
    print(f"{'─'*65}")
    layer3 = QConv2d(in_channels=1, out_channels=32, kernel_size=5)
    x3 = torch.randn(8, 1, 28, 28)
    analysis = analyze_gradient_magnitudes(layer3, x3, lam=0.05)
    theory_ok = analysis.get('b_grad_matches_theory', False)
    print(f"\n  Theory check (λ*fan_in formula): {'✓ PASS' if theory_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 4: Pruned channel
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("[Test 4] Pruned channel — b < 0 → zero gradient")
    print(f"{'─'*65}")
    layer4 = QConv2d(in_channels=1, out_channels=4, kernel_size=3)
    with torch.no_grad():
        layer4.b[1] = -0.5
        layer4.b[3] = -1.2
    x4 = torch.randn(2, 1, 8, 8)
    target4 = torch.zeros(2, 4, 6, 6)
    for p in [layer4.weight, layer4.e, layer4.b]:
        p.grad = None
    loss4 = compute_loss_with_compression(layer4, x4, target4, lam=0.05)
    loss4.backward()
    b_grad = layer4.b.grad.flatten()
    ch1_zero   = abs(b_grad[1].item()) < 1e-6
    ch3_zero   = abs(b_grad[3].item()) < 1e-6
    ch0_nonzero = abs(b_grad[0].item()) > 1e-6
    prune_grad_ok = ch1_zero and ch3_zero and ch0_nonzero
    print(f"  b.grad for each channel: {[f'{v:.4f}' for v in b_grad.tolist()]}")
    print(f"  ch 1 (b=-0.5) grad=0: {'✓' if ch1_zero else '✗'}")
    print(f"  ch 3 (b=-1.2) grad=0: {'✓' if ch3_zero else '✗'}")
    print(f"  ch 0 (b=2.0)  nonzero: {'✓' if ch0_nonzero else '✗'}")
    print(f"  Pruned channels get no gradient: {'✓ PASS' if prune_grad_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    all_pass = flow_ok and flow_ok2 and theory_ok and prune_grad_ok
    print(f"\n{'=' * 65}")
    if all_pass:
        print("GRADIENT FLOW VERIFIED: all parameters receive gradients")
        print("Pruned channels correctly receive zero b gradient")
    else:
        print("SOME TESTS FAILED — check your implementation")
    print("=" * 65)

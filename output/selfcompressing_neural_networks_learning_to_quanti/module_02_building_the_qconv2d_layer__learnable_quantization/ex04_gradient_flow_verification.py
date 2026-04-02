"""
Exercise 4: Gradient Flow Verification
=======================================
Module 2: Building the QConv2d Layer — Learnable Quantization

Self-compression only works if ALL THREE parameter groups (weight, e, b)
receive meaningful gradients during training:

    weight.grad → drives task performance (cross-entropy minimization)
    e.grad      → adapts quantization scale (finds right resolution per channel)
    b.grad      → drives compression (compression penalty pushes b toward 0)

If any one of these is zero or None, a crucial feedback loop is broken.
This exercise is about verifying — and understanding — these gradient flows.

The training loss is:
    L = L_task + λ * Q
where Q = compute_avg_bits(model) uses relu(b).sum() * fan_in per layer.

    ∂L/∂weight_i ≈ 2^(-e_i) * conv_gradient    (task signal)
    ∂L/∂e_i      = ln(2) * 2^(e_i) * Σ_j round(q_ij) * conv_grad_ij
    ∂L/∂b_i      = λ * fan_in * 𝟙[b_i > 0]     (pure compression pressure)

A key insight: ∂L/∂b comes primarily from the λ*Q term, NOT from the
task loss (the task loss's gradient through the clamp bounds is usually
small). This means b's gradient is approximately λ * fan_in — constant,
always pointing in the compression direction.

In this exercise you analyze these gradients concretely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Complete QConv2d (from Exercises 1-3 — provided)
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    return (x.round() - x).detach() + x


def _kaiming_uniform_like_ref(out_channels, in_channels, kH, kW):
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


class QConv2d(nn.Module):
    """Complete QConv2d layer.

    Attributes
    ----------
    weight : nn.Parameter, shape (out_ch, in_ch, kH, kW)
    e      : nn.Parameter, shape (out_ch, 1, 1, 1), init=-8.0
    b      : nn.Parameter, shape (out_ch, 1, 1, 1), init=2.0
    """

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


# ---------------------------------------------------------------------------
# Compression-aware loss helper (provided)
# ---------------------------------------------------------------------------

def compute_loss_with_compression(layer: QConv2d, x: torch.Tensor,
                                   target: torch.Tensor,
                                   lam: float = 0.05) -> torch.Tensor:
    """Compute combined task + compression loss for a single layer.

    L = MSE(output, target) + lam * Q
    where Q = layer.qbits() / layer.weight.numel()

    Parameters
    ----------
    layer  : QConv2d
    x      : torch.Tensor — input, shape (N, in_ch, H, W)
    target : torch.Tensor — target output, same shape as layer(x)
    lam    : float — compression penalty strength (default 0.05)

    Returns
    -------
    torch.Tensor — scalar loss
    """
    out = layer(x)
    task_loss = F.mse_loss(out, target)
    Q = layer.qbits() / layer.weight.numel()
    return task_loss + lam * Q


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def verify_gradient_flow(layer: QConv2d,
                          x: torch.Tensor,
                          lam: float = 0.05) -> bool:
    """Run a forward-backward pass and verify all three parameters get gradients.

    Computes the combined task + compression loss, calls .backward(), then
    inspects the .grad attribute of each parameter. Reports statistics for
    each gradient tensor and checks that all are nonzero.

    Parameters
    ----------
    layer : QConv2d — the layer to inspect (must have requires_grad=True for all params)
    x     : torch.Tensor — a single batch of random input, shape (N, in_ch, H, W)
    lam   : float — compression penalty coefficient (default 0.05)

    Returns
    -------
    bool — True if all three parameters have nonzero gradients, False otherwise

    Side effects
    ------------
    Prints a gradient report table:
        "param    | mean      | std       | frac_nonzero"
        For weight, e, and b.
    Zeroes out any existing .grad before the backward pass.
    """
    ###########################################################
    # YOUR CODE HERE — 18-25 lines                             #
    #                                                          #
    # Step 1: Zero any existing gradients on weight, e, b      #
    #         Use param.grad = None (cleaner than zero_grad)   #
    #                                                          #
    # Step 2: Create a target tensor with same shape as output  #
    #         output = layer(x), but we need target shape first #
    #         Use torch.no_grad() to get the shape, then make  #
    #         a random target of that shape.                   #
    #         OR: compute output, detach() it, add small noise  #
    #                                                          #
    # Step 3: Compute combined loss using compute_loss_with_compression() #
    #         loss = compute_loss_with_compression(layer, x, target, lam) #
    #                                                          #
    # Step 4: loss.backward()                                  #
    #                                                          #
    # Step 5: For each of [layer.weight, layer.e, layer.b]:    #
    #   - Check .grad is not None                              #
    #   - Compute mean abs value: grad.abs().mean().item()     #
    #   - Compute std: grad.std().item()                       #
    #   - Compute fraction nonzero: (grad != 0).float().mean() #
    #   - Print a formatted row                                #
    #                                                          #
    # Step 6: Return True if ALL three gradients are nonzero   #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def analyze_gradient_magnitudes(layer: QConv2d,
                                 x: torch.Tensor,
                                 lam: float = 0.05) -> dict:
    """Compare relative gradient magnitudes of weight, e, and b.

    Runs the same forward-backward as verify_gradient_flow but focuses
    on comparing the RELATIVE magnitude of each parameter's gradient.
    This is relevant for learning rate tuning: if b.grad is 1000x smaller
    than weight.grad, we might need a larger learning rate for b.

    Also verifies the theoretical gradient magnitude for b:
        ∂(λ*Q)/∂b_i ≈ λ * fan_in  (when b_i > 0)

    Parameters
    ----------
    layer : QConv2d
    x     : torch.Tensor — input batch
    lam   : float — compression penalty (default 0.05)

    Returns
    -------
    dict with keys:
        'weight_grad_norm' : float — L2 norm of weight.grad
        'e_grad_norm'      : float — L2 norm of e.grad
        'b_grad_norm'      : float — L2 norm of b.grad
        'b_theoretical'    : float — expected L2 norm from λ*fan_in formula
        'b_grad_matches_theory' : bool — actual vs theoretical within 20%
    """
    ###########################################################
    # YOUR CODE HERE — 8-12 lines                              #
    #                                                          #
    # Step 1: Zero grads, compute loss, call backward()        #
    #         (same setup as verify_gradient_flow)             #
    #                                                          #
    # Step 2: Compute L2 norms for each gradient              #
    #         norm = grad.norm().item()                        #
    #                                                          #
    # Step 3: Compute theoretical b gradient norm              #
    #         The compression term ∂(λ*Q)/∂b_i ≈ λ * fan_in  #
    #         for each of out_channels channels where b_i > 0  #
    #         b_theoretical = λ * fan_in * sqrt(out_channels)  #
    #         (sqrt because L2 norm of a constant vector)      #
    #                                                          #
    # Step 4: Check if actual b_grad_norm ≈ theoretical        #
    #         Within 50% is fine (task loss also contributes)  #
    #                                                          #
    # Step 5: Print a report and return the dict               #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# __main__ — validation harness (do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(7)

    print("=" * 65)
    print("Exercise 4: Gradient Flow Verification")
    print("=" * 65)

    # -------------------------------------------------------------------
    # Test 1: verify_gradient_flow — small layer, single batch
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("[Test 1] Gradient flow for QConv2d(1, 16, 5)")
    print(f"{'─'*65}")
    layer1 = QConv2d(in_channels=1, out_channels=16, kernel_size=5)
    x1 = torch.randn(4, 1, 12, 12, requires_grad=True)   # small input
    print(f"  Layer: {layer1.weight.shape}, input: {x1.shape}")
    print(f"  {'Parameter':<12} | {'|grad| mean':>12} | {'grad std':>10} | {'frac nz':>8}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    flow_ok = verify_gradient_flow(layer1, x1, lam=0.05)
    print(f"\n  All gradients nonzero: {'✓ PASS' if flow_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2: verify_gradient_flow — MNIST first layer
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
    # Test 3: analyze_gradient_magnitudes — verify theoretical b grad
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
    # Test 4: Pruned channel gets zero b gradient (b < 0 → relu'=0)
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("[Test 4] Pruned channel — b < 0 → zero gradient")
    print(f"{'─'*65}")
    layer4 = QConv2d(in_channels=1, out_channels=4, kernel_size=3)
    with torch.no_grad():
        layer4.b[1] = -0.5   # channel 1: pruned
        layer4.b[3] = -1.2   # channel 3: pruned
    x4 = torch.randn(2, 1, 8, 8)
    target4 = torch.zeros(2, 4, 6, 6)
    for p in [layer4.weight, layer4.e, layer4.b]:
        p.grad = None
    loss4 = compute_loss_with_compression(layer4, x4, target4, lam=0.05)
    loss4.backward()
    b_grad = layer4.b.grad.flatten()
    ch1_zero = abs(b_grad[1].item()) < 1e-6
    ch3_zero = abs(b_grad[3].item()) < 1e-6
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

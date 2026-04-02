"""
Exercise 3: Counting Bits — The qbits() Method and Model Size
=============================================================
SOLUTION FILE

Reference (tinygrad):
    def qbits(self):
        return self.b.relu().sum() * prod(self.weight.shape[1:])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _kaiming_uniform_like_ref(out_channels, in_channels, kH, kW):
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


class QConv2d(nn.Module):
    """Quantization-aware Conv2d (complete)."""

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
        """Total bits required to store this layer's weights at current bit-widths.

        Returns
        -------
        torch.Tensor — scalar, differentiable w.r.t. self.b
        """
        # fan_in = number of weights per output channel (in_ch * kH * kW)
        fan_in = math.prod(self.weight.shape[1:])

        # Sum effective bit-widths over all output channels
        # relu(b) ensures pruned channels (b < 0) contribute 0, not negative bits
        # Gradient: d(relu(b_i).sum() * fan_in) / d(b_i) = fan_in when b_i > 0
        return torch.relu(self.b).sum() * fan_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self.qweight()
        w  = ste_round(qw)
        dw = (2 ** self.e) * w
        return F.conv2d(x, dw, stride=self.stride, padding=self.padding)


def compute_avg_bits(model: nn.Module) -> torch.Tensor:
    """Compute average bits per parameter across all QConv2d layers.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    torch.Tensor — scalar Q, average bits per parameter
    """
    # Find all QConv2d layers (including nested ones via modules())
    qconv_layers = [m for m in model.modules() if isinstance(m, QConv2d)]

    if not qconv_layers:
        raise ValueError("No QConv2d layers found in model")

    # Sum qbits() across all layers — total bits for all weights
    total_bits = sum(layer.qbits() for layer in qconv_layers)

    # Total parameter count — includes weight + e + b for each layer
    # This matches the reference: 87,860 total params (not just weight params)
    total_params = sum(p.numel() for p in model.parameters())

    return total_bits / total_params


def compute_model_bytes(Q: float, weight_count: int) -> float:
    """Convert average bits per parameter to total model size in bytes.

    Parameters
    ----------
    Q            : float — average bits per parameter
    weight_count : int   — total number of parameters

    Returns
    -------
    float — model_bytes = Q / 8 * weight_count
    """
    return (Q / 8) * weight_count


class MNISTModel(nn.Module):
    """Self-compressing CNN for MNIST — reference architecture from the paper."""

    def __init__(self):
        super().__init__()
        self.conv1 = QConv2d(1,   32, 5)
        self.conv2 = QConv2d(32,  32, 5)
        self.bn1   = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.conv3 = QConv2d(32,  64, 3)
        self.conv4 = QConv2d(64,  64, 3)
        self.bn2   = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.conv5 = QConv2d(576, 10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(1).reshape(-1, 576, 1, 1)
        x = self.conv5(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# __main__ — validation harness (identical to scaffold)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 65)
    print("Exercise 3: Counting Bits — qbits() and Model Size")
    print("=" * 65)

    # -------------------------------------------------------------------
    # Test 1: qbits() for a single QConv2d(1, 32, 5)
    # -------------------------------------------------------------------
    layer = QConv2d(in_channels=1, out_channels=32, kernel_size=5)
    qb = layer.qbits()
    expected_qbits = 1600.0
    qbits_ok = abs(qb.item() - expected_qbits) < 1.0
    print(f"\n[Test 1] Single layer qbits() — QConv2d(1, 32, 5), b=2.0")
    print(f"  fan_in = 1*5*5 = 25, sum(relu(b)) = 32*2.0 = 64.0")
    print(f"  Expected qbits: {expected_qbits:.1f}")
    print(f"  Got     qbits: {qb.item():.1f}")
    print(f"  Result        : {'✓ PASS' if qbits_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2: qbits() for QConv2d(32, 64, 3)
    # -------------------------------------------------------------------
    layer2 = QConv2d(in_channels=32, out_channels=64, kernel_size=3)
    qb2 = layer2.qbits()
    expected_qbits2 = 36864.0
    qbits2_ok = abs(qb2.item() - expected_qbits2) < 1.0
    print(f"\n[Test 2] Layer qbits() — QConv2d(32, 64, 3), b=2.0")
    print(f"  fan_in = 32*3*3 = 288, sum(relu(b)) = 64*2.0 = 128.0")
    print(f"  Expected qbits: {expected_qbits2:.1f}")
    print(f"  Got     qbits: {qb2.item():.1f}")
    print(f"  Result        : {'✓ PASS' if qbits2_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 3: qbits() with one pruned channel
    # -------------------------------------------------------------------
    layer3 = QConv2d(in_channels=1, out_channels=4, kernel_size=3)
    with torch.no_grad():
        layer3.b[2] = -0.5
    qb3 = layer3.qbits()
    expected_qbits3 = 54.0
    qbits3_ok = abs(qb3.item() - expected_qbits3) < 0.1
    print(f"\n[Test 3] qbits() with pruned channel (b=-0.5 for ch 2)")
    print(f"  Expected: 3 active * 2.0 * 9 = {expected_qbits3:.1f}")
    print(f"  Got     : {qb3.item():.4f}")
    print(f"  Result  : {'✓ PASS' if qbits3_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 4: Full MNIST model metrics
    # -------------------------------------------------------------------
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Test 4] Full MNIST model metrics")
    print(f"  Total parameters: {total_params:,}")

    Q = compute_avg_bits(model)
    Q_val = Q.item() if hasattr(Q, 'item') else float(Q)
    q_ok = abs(Q_val - 2.0) < 0.05

    model_bytes = compute_model_bytes(Q_val, total_params)
    # Note: Q is slightly below 2.0 because total_params includes e and b
    # (which don't count toward qbits but do count toward the denominator).
    # So model_bytes will be slightly below total_params * 2 / 8.
    bytes_ok = abs(model_bytes - Q_val * total_params / 8) < 1.0

    print(f"  Q (bits/weight) : {Q_val:.4f}  (expected ~2.0 at init)")
    print(f"  Q correct       : {'✓ PASS' if q_ok else '✗ FAIL'}")
    print(f"  Model bytes     : {model_bytes:.1f}  ({model_bytes/1024:.2f} KB)")
    print(f"  Bytes correct   : {'✓ PASS' if bytes_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 5: Per-layer breakdown table
    # -------------------------------------------------------------------
    print(f"\n[Test 5] Per-layer breakdown")
    print(f"  {'Layer':<18}  {'out_ch':>6}  {'fan_in':>7}  {'qbits':>8}  {'bits/w':>7}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*7}")
    for name, m in model.named_modules():
        if isinstance(m, QConv2d):
            fan_in = math.prod(m.weight.shape[1:])
            out_ch = m.weight.shape[0]
            bits   = m.qbits().item()
            bpw    = bits / (out_ch * fan_in)
            print(f"  {name:<18}  {out_ch:>6}  {fan_in:>7}  {bits:>8.0f}  {bpw:>7.4f}")

    # -------------------------------------------------------------------
    # Test 6: qbits() is differentiable
    # -------------------------------------------------------------------
    layer4 = QConv2d(in_channels=3, out_channels=8, kernel_size=3)
    qb4 = layer4.qbits()
    qb4.backward()
    b_grad = layer4.b.grad
    expected_grad = float(math.prod(layer4.weight.shape[1:]))
    grad_ok = (b_grad.abs() - expected_grad).abs().max().item() < 0.01
    print(f"\n[Test 6] qbits() is differentiable w.r.t. b")
    print(f"  Expected grad magnitude: {expected_grad}")
    print(f"  Actual grad (sample)   : {b_grad.flatten()[0].item():.4f}")
    print(f"  Gradient correct       : {'✓ PASS' if grad_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    all_pass = qbits_ok and qbits2_ok and qbits3_ok and q_ok and bytes_ok and grad_ok
    print(f"\n{'=' * 65}")
    if all_pass:
        print(f"Model size: ~{model_bytes/1024:.1f} KB at {Q_val:.1f} bits/weight")
        print(f"bits/weight: {Q_val:.4f}")
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — check your implementation")
    print("=" * 65)

"""
Exercise 2: QConv2d Forward Pass with Straight-Through Estimator
=================================================================
SOLUTION FILE

Reference (tinygrad):
    qw = self.qweight()
    w  = (qw.round() - qw).detach() + qw   # STE
    return x.conv2d(2**self.e * w)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for integer rounding.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor — x.round() in forward, gradient=1 in backward
    """
    return (x.round() - x).detach() + x


def _kaiming_uniform_like_ref(out_channels: int, in_channels: int,
                               kH: int, kW: int) -> torch.Tensor:
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


class QConv2d(nn.Module):
    """Quantization-aware Conv2d with per-channel learnable bit-widths.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int or tuple
    stride       : int (default 1)
    padding      : int (default 0)
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
        """Compute per-channel clamped quantized weights.

        Returns
        -------
        torch.Tensor — shape (out_ch, in_ch, kH, kW), values in integer range
        """
        eff_b  = torch.relu(self.b)
        lower  = -(2 ** (eff_b - 1))
        upper  =  (2 ** (eff_b - 1)) - 1
        scaled = (2 ** (-self.e)) * self.weight
        return torch.minimum(torch.maximum(scaled, lower), upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized forward pass with straight-through estimator.

        Pipeline: qweight() → ste_round → 2^e * w → F.conv2d

        Parameters
        ----------
        x : torch.Tensor — shape (N, in_ch, H, W)

        Returns
        -------
        torch.Tensor — shape (N, out_ch, H', W')
        """
        # Step 1: get clamped integer-range weights (float32, not yet rounded)
        qw = self.qweight()

        # Step 2: STE rounding
        # Forward: qw.round() — discrete integer values
        # Backward: gradient flows as if this were the identity
        w = ste_round(qw)

        # Step 3: dequantize — scale back to the weight domain
        # 2^e * round(qw) maps integer back to the quantization grid in float space
        # Grid spacing is 2^e; representable values are {..., -2^e, 0, 2^e, 2*2^e, ...}
        dw = (2 ** self.e) * w

        # Step 4: standard convolution — no modification here
        return F.conv2d(x, dw, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# __main__ — validation harness (identical to scaffold)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("Exercise 2: QConv2d Forward Pass with STE")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Test 1: Output shape for MNIST-like input
    # -------------------------------------------------------------------
    layer = QConv2d(in_channels=1, out_channels=32, kernel_size=5)
    x = torch.randn(8, 1, 28, 28, requires_grad=True)
    out = layer(x)

    expected_shape = (8, 32, 24, 24)
    shape_ok = tuple(out.shape) == expected_shape
    print(f"\n[Test 1] Output shape")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Expected: {expected_shape}")
    print(f"  Shape  : {'✓ PASS' if shape_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2: Output requires_grad
    # -------------------------------------------------------------------
    grad_ok = out.requires_grad
    print(f"\n[Test 2] Output requires_grad (gradient computation enabled)")
    print(f"  out.requires_grad: {out.requires_grad}")
    print(f"  Grad flows      : {'✓ PASS' if grad_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 3: Output statistics
    # -------------------------------------------------------------------
    out_mean = out.mean().item()
    out_std  = out.std().item()
    out_finite = torch.isfinite(out).all().item()
    out_nonzero = (out.abs() > 1e-8).any().item()
    stats_ok = out_finite and out_nonzero
    print(f"\n[Test 3] Output statistics")
    print(f"  mean  : {out_mean:.4f}")
    print(f"  std   : {out_std:.4f}")
    print(f"  finite: {out_finite}")
    print(f"  nonzero: {out_nonzero}")
    print(f"  Stats : {'✓ PASS' if stats_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 4: Quantization grid
    # -------------------------------------------------------------------
    with torch.no_grad():
        qw = layer.qweight()
        rounded_qw = qw.round()
        unique_vals = rounded_qw.unique()
        grid_ok = len(unique_vals) <= 4
    print(f"\n[Test 4] Quantization grid")
    print(f"  Unique integer values in qw.round(): {sorted(unique_vals.tolist())}")
    print(f"  Count (<= 4 for 2-bit)              : {'✓ PASS' if grid_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 5: Backward pass
    # -------------------------------------------------------------------
    loss = out.mean()
    loss.backward()
    w_grad  = layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    e_grad  = layer.e.grad is not None and layer.e.grad.abs().sum() > 0
    b_grad  = layer.b.grad is not None and layer.b.grad.abs().sum() > 0
    backprop_ok = w_grad and e_grad and b_grad
    print(f"\n[Test 5] Backward pass (all parameters receive gradients)")
    print(f"  weight.grad nonzero: {w_grad}")
    print(f"  e.grad nonzero     : {e_grad}")
    print(f"  b.grad nonzero     : {b_grad}")
    print(f"  All grads flow     : {'✓ PASS' if backprop_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 6: stride and padding
    # -------------------------------------------------------------------
    layer_padded = QConv2d(in_channels=1, out_channels=8, kernel_size=3,
                            stride=1, padding=1)
    x2 = torch.randn(2, 1, 28, 28)
    out2 = layer_padded(x2)
    pad_ok = tuple(out2.shape) == (2, 8, 28, 28)
    print(f"\n[Test 6] Stride/padding (kernel=3, stride=1, padding=1 → same size)")
    print(f"  Output shape: {tuple(out2.shape)}")
    print(f"  Same spatial: {'✓ PASS' if pad_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    all_pass = shape_ok and grad_ok and stats_ok and backprop_ok and pad_ok
    print(f"\n{'=' * 60}")
    if all_pass:
        print("QConv2d forward pass: PASSED")
        print(f"Output shape {tuple(out.shape)}, mean={out_mean:.4f}, std={out_std:.4f}")
    else:
        print("SOME TESTS FAILED — check your implementation")
    print("=" * 60)

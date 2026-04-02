"""
Exercise 2: QConv2d Forward Pass with Straight-Through Estimator
=================================================================
Module 2: Building the QConv2d Layer — Learnable Quantization

Building on Exercise 1 (qweight is already implemented), you now wire up
the complete forward pass. The key challenge: rounding is non-differentiable,
but we need gradients to flow through it so all three parameters (weight, e, b)
can be updated via backprop.

The Straight-Through Estimator (STE) from Module 1 solves this:

    Forward:  ste_round(qw) == qw.round()      (discrete integer)
    Backward: d/d(qw) ste_round(qw) == 1       (as if no rounding)

Implemented as:  (qw.round() - qw).detach() + qw

After rounding, the dequantized weight is: dw = 2^e * round(qw)
This maps the discrete integer back to the continuous weight space.
The convolution then runs on dw — a standard F.conv2d call.

Important: the order matters.
    CORRECT:   dw = 2^e * ste_round(qw)   → round first, then scale
    INCORRECT: dw = ste_round(2^e * qw)   → changes the quantization grid

Reference (tinygrad):
    qw = self.qweight()
    w  = (qw.round() - qw).detach() + qw   # STE
    return x.conv2d(2**self.e * w)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# STE round — provided (from Module 1)
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for integer rounding.

    Forward pass: equivalent to x.round() — returns discrete integer values.
    Backward pass: gradient of output w.r.t. input is 1 everywhere — the
    rounding operation is treated as an identity for gradient computation.

    The trick: (x.round() - x).detach() has gradient 0 (detached from graph),
    so the gradient of the whole expression comes entirely from the final +x.

    Parameters
    ----------
    x : torch.Tensor — any shape, float dtype

    Returns
    -------
    torch.Tensor — same shape as x, rounded to nearest integer (forward pass)
    """
    return (x.round() - x).detach() + x


# ---------------------------------------------------------------------------
# Helper (from Exercise 1 — provided)
# ---------------------------------------------------------------------------

def _kaiming_uniform_like_ref(out_channels: int, in_channels: int,
                               kH: int, kW: int) -> torch.Tensor:
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


# ---------------------------------------------------------------------------
# QConv2d — qweight() provided from Exercise 1, you implement forward()
# ---------------------------------------------------------------------------

class QConv2d(nn.Module):
    """Quantization-aware Conv2d with per-channel learnable bit-widths.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int or tuple
    stride       : int (default 1)
    padding      : int (default 0)

    Attributes
    ----------
    weight : nn.Parameter, shape (out_ch, in_ch, kH, kW) — raw float weights
    e      : nn.Parameter, shape (out_ch, 1, 1, 1), init=-8.0 — exponent
    b      : nn.Parameter, shape (out_ch, 1, 1, 1), init=2.0  — bit-width
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
        """Compute per-channel clamped quantized weights (from Exercise 1).

        Returns
        -------
        torch.Tensor — shape (out_ch, in_ch, kH, kW)
            Values in signed integer range [−2^(eff_b−1), 2^(eff_b−1)−1],
            stored as float32 (NOT yet rounded).
        """
        eff_b  = torch.relu(self.b)
        lower  = -(2 ** (eff_b - 1))
        upper  =  (2 ** (eff_b - 1)) - 1
        scaled = (2 ** (-self.e)) * self.weight
        return torch.minimum(torch.maximum(scaled, lower), upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized forward pass with straight-through estimator.

        Steps:
            1. Compute quantized (integer-range, float) weights via qweight()
            2. Apply STE rounding: ste_round(qw) — discrete in forward,
               gradient=1 in backward
            3. Dequantize: multiply rounded integers by 2^e to get weights
               back in the original float scale
            4. Run standard F.conv2d with the dequantized weights

        Why dequantize AFTER rounding (not before)?
            Applying 2^e before rounding would change the quantization grid.
            The grid spacing is 2^e. We want to round to multiples of 2^e,
            which means scaling by 2^(-e) first (in qweight), rounding, then
            scaling back by 2^e.

        Parameters
        ----------
        x : torch.Tensor — shape (N, in_ch, H, W), float32, requires_grad=True

        Returns
        -------
        torch.Tensor — shape (N, out_ch, H', W') where
            H' = (H - kH + 2*padding) // stride + 1
            W' = (W - kW + 2*padding) // stride + 1
        """
        ###########################################################
        # YOUR CODE HERE — 6-10 lines                              #
        #                                                          #
        # Step 1: Get quantized weights from qweight()             #
        #         qw = ...   shape: (out_ch, in_ch, kH, kW)        #
        #                                                          #
        # Step 2: Apply STE rounding                               #
        #         w = ste_round(qw)                                #
        #         Forward: same as qw.round() — discrete integers  #
        #         Backward: gradient flows through as identity      #
        #                                                          #
        # Step 3: Dequantize — scale integers back to weight space #
        #         dw = (2 ** self.e) * w                           #
        #         self.e has shape (out_ch, 1, 1, 1) → broadcasts  #
        #                                                          #
        # Step 4: Apply standard convolution                       #
        #         return F.conv2d(x, dw,                           #
        #                         stride=self.stride,              #
        #                         padding=self.padding)            #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################


# ---------------------------------------------------------------------------
# __main__ — validation harness (do not modify)
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
    x = torch.randn(8, 1, 28, 28, requires_grad=True)   # batch of 8 MNIST images
    out = layer(x)

    expected_shape = (8, 32, 24, 24)   # 28-5+1=24
    shape_ok = tuple(out.shape) == expected_shape
    print(f"\n[Test 1] Output shape")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Expected: {expected_shape}")
    print(f"  Shape  : {'✓ PASS' if shape_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2: Output requires_grad (critical — gradients must flow)
    # -------------------------------------------------------------------
    grad_ok = out.requires_grad
    print(f"\n[Test 2] Output requires_grad (gradient computation enabled)")
    print(f"  out.requires_grad: {out.requires_grad}")
    print(f"  Grad flows      : {'✓ PASS' if grad_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 3: Output statistics (finite, non-trivial values)
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
    # Test 4: Dequantized weights are on the quantization grid
    # -------------------------------------------------------------------
    # The dequantized weight should be 2^e * round(qw)
    # Check that dw / 2^e is close to an integer
    with torch.no_grad():
        qw = layer.qweight()
        # When b=2.0, qweight values should already be in [-2, 1]
        # After rounding, the unique values should be at most {-2,-1,0,1}
        rounded_qw = qw.round()
        unique_vals = rounded_qw.unique()
        grid_ok = len(unique_vals) <= 4
    print(f"\n[Test 4] Quantization grid")
    print(f"  Unique integer values in qw.round(): {sorted(unique_vals.tolist())}")
    print(f"  Count (<= 4 for 2-bit)              : {'✓ PASS' if grid_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 5: Backward pass — all params get gradients
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
    pad_ok = tuple(out2.shape) == (2, 8, 28, 28)   # padding=1 preserves spatial size
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

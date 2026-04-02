"""
Exercise 1: Per-Channel Quantized Weights — qweight()
======================================================
Module 2: Building the QConv2d Layer — Learnable Quantization

In this exercise you implement the core quantization formula of the
Self-Compressing Neural Networks paper. The `qweight()` method takes the
raw floating-point weight tensor and maps it to an integer-valued
representation using two per-channel parameters:

  - e  (exponent): controls the scale / resolution of the quantization grid
  - b  (bit-width): controls the range (how many distinct values)

The result is NOT rounded yet — that happens in the forward pass via the
straight-through estimator (Exercise 2). Here we produce the *clamped*
representation: values in the correct integer range, stored as floats.

Reference formula (from the paper, tinygrad original):
    qweight = clip(2^(-e) * w, -2^(relu(b)-1), 2^(relu(b)-1) - 1)

Learning goal:
    Understand WHY relu(b) is essential and HOW the bounds produce
    emergent channel pruning when b falls below zero during training.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Helper: Kaiming uniform initialization matching the reference implementation
# ---------------------------------------------------------------------------

def _kaiming_uniform_like_ref(out_channels: int, in_channels: int,
                               kH: int, kW: int) -> torch.Tensor:
    """Initialize weights with Kaiming uniform (same formula as reference).

    The reference uses scale = 1/sqrt(in_channels * kH * kW), not PyTorch's
    default which uses a different fan_in calculation.

    Parameters
    ----------
    out_channels : int
    in_channels  : int
    kH, kW       : int — kernel spatial dimensions

    Returns
    -------
    torch.Tensor — shape (out_channels, in_channels, kH, kW)
    """
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


# ---------------------------------------------------------------------------
# QConv2d — skeleton (you implement qweight below)
# ---------------------------------------------------------------------------

class QConv2d(nn.Module):
    """Quantization-aware Conv2d with per-channel learnable bit-widths.

    Each output channel has its own scalar exponent (e) and bit-width (b)
    parameter, stored as tensors of shape (out_channels, 1, 1, 1). This
    shape broadcasts cleanly against the weight tensor (out_ch, in_ch, kH, kW).

    Parameters
    ----------
    in_channels  : int — number of input feature map channels
    out_channels : int — number of output channels (filters)
    kernel_size  : int or tuple — spatial size of each convolutional kernel
    stride       : int — convolution stride (default 1)
    padding      : int — zero-padding added to both sides (default 0)

    Attributes
    ----------
    weight : nn.Parameter, shape (out_ch, in_ch, kH, kW)
        Raw floating-point weights, Kaiming-uniform initialized.
    e      : nn.Parameter, shape (out_ch, 1, 1, 1), init -8.0
        Per-channel exponent. Controls quantization grid resolution.
        Grid spacing = 2^e. Smaller e → finer grid (more precision).
    b      : nn.Parameter, shape (out_ch, 1, 1, 1), init 2.0
        Per-channel bit-width. Controls how many distinct integer values
        are representable in each channel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride   = stride
        self.padding  = padding
        self.kernel_size = kernel_size

        # Weight tensor — Kaiming uniform, matching reference implementation
        self.weight = nn.Parameter(
            _kaiming_uniform_like_ref(out_channels, in_channels, *kernel_size)
        )

        # Per-channel quantization parameters — shape (out_ch, 1, 1, 1)
        # Broadcasting: each scalar e_i / b_i applies to all weights in filter i
        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), -8.0))
        self.b = nn.Parameter(torch.full((out_channels, 1, 1, 1),  2.0))

    def qweight(self) -> torch.Tensor:
        """Compute per-channel clamped quantized weights.

        Maps raw floating-point weights to the signed integer range
        determined by the effective bit-width relu(b). Weights are scaled
        into the integer domain by 2^(-e), clamped to the representable
        range, then returned as floats (rounding happens in the forward pass).

        The formula (from the paper):
            eff_b  = relu(b)                           # non-negative effective bits
            lower  = -2^(eff_b - 1)                   # min signed integer
            upper  =  2^(eff_b - 1) - 1               # max signed integer
            scaled = 2^(-e) * weight                   # map to integer domain
            qw     = clip(scaled, lower, upper)        # clamp to range

        Key insight: when b_i < 0, relu(b_i) = 0, making lower = upper = -0.5.
        After rounding (in Exercise 2), this forces the entire channel to zero
        — emergent channel pruning without any explicit pruning algorithm.

        Parameters
        ----------
        (uses self.weight, self.e, self.b — all shape-compatible)

        Returns
        -------
        torch.Tensor — shape (out_ch, in_ch, kH, kW)
            Values are in the signed integer range for the current b,
            stored as float32. NOT yet rounded to integers.
        """
        ###########################################################
        # YOUR CODE HERE — 8-12 lines                              #
        #                                                          #
        # Step 1: Compute eff_b = relu(self.b)                    #
        #         Shape: (out_ch, 1, 1, 1)                         #
        #                                                          #
        # Step 2: Compute lower = -2^(eff_b - 1)                  #
        #         Use Python's ** operator or torch.pow            #
        #         Shape: (out_ch, 1, 1, 1)                         #
        #                                                          #
        # Step 3: Compute upper = 2^(eff_b - 1) - 1               #
        #         Shape: (out_ch, 1, 1, 1)                         #
        #                                                          #
        # Step 4: Scale weights: scaled = 2^(-self.e) * self.weight#
        #         Shape: (out_ch, in_ch, kH, kW) — broadcasts!     #
        #                                                          #
        # Step 5: Clamp: torch.minimum(torch.maximum(scaled, lower), upper)
        #         Do NOT use torch.clamp — it requires scalar bounds#
        #         torch.minimum/maximum handle per-channel tensors  #
        #                                                          #
        # Return the clamped result.                               #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder forward — implemented in Exercise 2."""
        raise NotImplementedError("Forward pass implemented in Exercise 2")


# ---------------------------------------------------------------------------
# __main__ — validation harness (do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Exercise 1: Per-Channel Quantized Weights — qweight()")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Test 1: Shape correctness
    # -------------------------------------------------------------------
    layer = QConv2d(in_channels=3, out_channels=16, kernel_size=3)
    qw = layer.qweight()

    expected_shape = layer.weight.shape  # (16, 3, 3, 3)
    shape_ok = qw.shape == expected_shape
    print(f"\n[Test 1] Shape check")
    print(f"  weight.shape : {layer.weight.shape}")
    print(f"  qweight shape: {qw.shape}")
    print(f"  Match        : {'✓ PASS' if shape_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2: With b=2.0, values should be in [-2, 1]
    # -------------------------------------------------------------------
    qw_min = qw.min().item()
    qw_max = qw.max().item()
    # At b=2.0: eff_b=2, lower=-2^(2-1)=-2, upper=2^(2-1)-1=1
    range_ok = (qw_min >= -2.01) and (qw_max <= 1.01)
    print(f"\n[Test 2] Range check (b=2.0 → integer range [-2, 1])")
    print(f"  qweight min  : {qw_min:.4f}  (expected >= -2.0)")
    print(f"  qweight max  : {qw_max:.4f}  (expected <=  1.0)")
    print(f"  In range     : {'✓ PASS' if range_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 3: After rounding, values should be exact integers
    # -------------------------------------------------------------------
    rounded = qw.round()
    are_integer_valued = (qw - rounded).abs().max().item() < 0.5
    # The clamped values should be between integers (fractional parts present)
    # but WITHIN the integer bounds — not necessarily integers themselves yet
    print(f"\n[Test 3] Values are within integer bounds (pre-round)")
    print(f"  Max deviation from nearest int: {(qw - rounded).abs().max().item():.4f}")
    print(f"  All within ±0.5 of an integer : {'✓ PASS' if are_integer_valued else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 4: Per-channel statistics
    # -------------------------------------------------------------------
    print(f"\n[Test 4] Per-channel statistics (first 4 of 16 channels)")
    print(f"  {'ch':>3}  {'min':>8}  {'max':>8}  {'mean':>8}  {'unique':>7}")
    for ch in range(4):
        ch_vals = qw[ch]
        ch_unique = ch_vals.round().unique().numel()
        print(f"  {ch:>3}  {ch_vals.min().item():>8.4f}  "
              f"{ch_vals.max().item():>8.4f}  "
              f"{ch_vals.mean().item():>8.4f}  "
              f"{ch_unique:>7}")

    # -------------------------------------------------------------------
    # Test 5: Pruning behavior — set b < 0 for one channel
    # -------------------------------------------------------------------
    layer2 = QConv2d(in_channels=1, out_channels=4, kernel_size=3)
    with torch.no_grad():
        layer2.b[2] = -1.0   # channel 2: effectively pruned
    qw2 = layer2.qweight()
    # When b_eff = relu(-1.0) = 0:
    #   lower = -2^(-1) = -0.5
    #   upper =  2^(-1) - 1 = -0.5
    # So all weights in channel 2 should equal -0.5
    ch2_vals = qw2[2]
    ch2_const = (ch2_vals - ch2_vals[0, 0, 0]).abs().max().item()
    ch2_val   = ch2_vals.mean().item()
    prune_ok  = abs(ch2_val - (-0.5)) < 0.01
    print(f"\n[Test 5] Pruning behavior (b=-1.0 → channel forced to -0.5)")
    print(f"  channel 2 mean value : {ch2_val:.4f}  (expected -0.5)")
    print(f"  channel 2 is constant: {'yes' if ch2_const < 1e-6 else 'no'}")
    print(f"  Pruning works        : {'✓ PASS' if prune_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    all_pass = shape_ok and range_ok and are_integer_valued and prune_ok
    print(f"\n{'=' * 60}")
    if all_pass:
        print("qweight shape: correct, range: correct, unique values per channel: ~4")
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — check your implementation")
    print("=" * 60)

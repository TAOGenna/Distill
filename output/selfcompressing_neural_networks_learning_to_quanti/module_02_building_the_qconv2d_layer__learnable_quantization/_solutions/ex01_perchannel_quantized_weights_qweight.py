"""
Exercise 1: Per-Channel Quantized Weights — qweight()
======================================================
Module 2: Building the QConv2d Layer — Learnable Quantization

SOLUTION FILE — do not distribute to students before they attempt the exercise.

Reference formula (from the paper, tinygrad original):
    qweight = clip(2^(-e) * w, -2^(relu(b)-1), 2^(relu(b)-1) - 1)
"""

import torch
import torch.nn as nn
import math


def _kaiming_uniform_like_ref(out_channels: int, in_channels: int,
                               kH: int, kW: int) -> torch.Tensor:
    """Initialize weights with Kaiming uniform (same formula as reference).

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
    weight : nn.Parameter, shape (out_ch, in_ch, kH, kW)
    e      : nn.Parameter, shape (out_ch, 1, 1, 1), init -8.0
    b      : nn.Parameter, shape (out_ch, 1, 1, 1), init 2.0
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

        Maps raw floating-point weights to the signed integer range
        determined by the effective bit-width relu(b).

        Formula:
            eff_b  = relu(b)
            lower  = -2^(eff_b - 1)
            upper  =  2^(eff_b - 1) - 1
            scaled = 2^(-e) * weight
            return clip(scaled, lower, upper)

        Returns
        -------
        torch.Tensor — shape (out_ch, in_ch, kH, kW)
            Values in the signed integer range, stored as float32.
            NOT yet rounded to integers (rounding via STE in Exercise 2).
        """
        # Step 1: effective bit-width — relu gates out negative b values
        # When b < 0: eff_b = 0, so lower = upper = -0.5 → channel pruned
        eff_b = torch.relu(self.b)                          # (out_ch, 1, 1, 1)

        # Step 2: signed integer bounds for eff_b bits
        # For eff_b=2: lower=-2, upper=1  → {-2,-1,0,1}
        # For eff_b=3: lower=-4, upper=3  → {-4,...,3}
        # For eff_b=0: lower=-0.5, upper=-0.5 → rounds to 0 (pruned)
        lower = -(2 ** (eff_b - 1))                         # (out_ch, 1, 1, 1)
        upper =  (2 ** (eff_b - 1)) - 1                     # (out_ch, 1, 1, 1)

        # Step 3: scale weights into the integer domain
        # 2^(-e) with e=-8 gives factor 2^8=256: fine-grained mapping
        scaled = (2 ** (-self.e)) * self.weight              # broadcasts over (in_ch, kH, kW)

        # Step 4: clamp to the representable integer range
        # MUST use torch.minimum/maximum (not torch.clamp) because bounds
        # are per-channel tensors, not scalars
        return torch.minimum(torch.maximum(scaled, lower), upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder forward — implemented in Exercise 2."""
        raise NotImplementedError("Forward pass implemented in Exercise 2")


# ---------------------------------------------------------------------------
# __main__ — validation harness (identical to scaffold)
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

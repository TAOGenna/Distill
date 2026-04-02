"""
Exercise 3: Counting Bits — The qbits() Method and Model Size
=============================================================
Module 2: Building the QConv2d Layer — Learnable Quantization

Now that QConv2d's forward pass works, we need a way to measure how
compressed the model is. That's the job of `qbits()` and related helpers.

The compression penalty in the training loss (Module 3) is:
    L = L_task + λ * Q

where Q is the average bits per weight across all layers:
    Q = sum_layers(layer.qbits()) / total_weight_count

And qbits() for a single layer:
    Q_layer = relu(b).sum() * fan_in      (paper equation: z_l = I_l*H_l*W_l * sum_i b_{l,i})

where fan_in = in_channels * kH * kW is the number of weights per output channel.

In this exercise you implement:
    1. QConv2d.qbits()          — bits for one layer
    2. compute_avg_bits(model)  — average Q across all QConv2d layers
    3. compute_model_bytes(Q, n) — bytes from Q and parameter count

Reference numbers to reproduce:
    - At init (b=2.0 everywhere): Q ≈ 2.0 bits/weight, model ≈ 21.5 KB
    - After training (reference): Q ≈ 1.64 bits/weight, model ≈ 18.1 KB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


# ---------------------------------------------------------------------------
# STE + QConv2d (complete from Exercises 1 & 2 — provided)
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator rounding."""
    return (x.round() - x).detach() + x


def _kaiming_uniform_like_ref(out_channels, in_channels, kH, kW):
    fan_in = in_channels * kH * kW
    scale  = 1.0 / math.sqrt(fan_in)
    return torch.empty(out_channels, in_channels, kH, kW).uniform_(-scale, scale)


class QConv2d(nn.Module):
    """Quantization-aware Conv2d (complete implementation from Exercises 1 & 2).

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
        """Clamped quantized weights (integer range, float dtype)."""
        eff_b  = torch.relu(self.b)
        lower  = -(2 ** (eff_b - 1))
        upper  =  (2 ** (eff_b - 1)) - 1
        scaled = (2 ** (-self.e)) * self.weight
        return torch.minimum(torch.maximum(scaled, lower), upper)

    def qbits(self) -> torch.Tensor:
        """Total bits required to store this layer's weights at current bit-widths.

        Each output channel contributes relu(b_i) * fan_in bits, where
        fan_in = in_channels * kH * kW (weights per output channel).

        Summing over all output channels:
            qbits = sum_i(relu(b_i)) * fan_in
                  = relu(b).sum() * fan_in

        Key: use relu(b), NOT raw b. Channels with b < 0 are pruned
        (contribute 0 bits). Using raw b would give negative bit counts,
        which would decrease the compression penalty — the opposite of what
        we want.

        The gradient of qbits w.r.t. b_i is fan_in when b_i > 0, and 0
        otherwise. This constant-magnitude gradient is what pushes each
        channel's bit-width down during training (via the λ*Q term in the loss).

        Returns
        -------
        torch.Tensor — scalar (0-d tensor)
            Total bits for this layer. Differentiable w.r.t. self.b.
        """
        ###########################################################
        # YOUR CODE HERE — 4-6 lines                               #
        #                                                          #
        # Step 1: Compute fan_in — product of all weight dims      #
        #         EXCEPT the first (output channel) dimension.     #
        #         self.weight.shape = (out_ch, in_ch, kH, kW)     #
        #         fan_in = in_ch * kH * kW                         #
        #         Use: math.prod(self.weight.shape[1:])            #
        #                                                          #
        # Step 2: Sum the effective bit-widths over output channels #
        #         torch.relu(self.b).sum()                         #
        #         Result is scalar-like (shape: (), or (1,))       #
        #                                                          #
        # Step 3: Multiply by fan_in and return                    #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self.qweight()
        w  = ste_round(qw)
        dw = (2 ** self.e) * w
        return F.conv2d(x, dw, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# Model-level compression metrics — YOU IMPLEMENT THESE
# ---------------------------------------------------------------------------

def compute_avg_bits(model: nn.Module) -> torch.Tensor:
    """Compute average bits per parameter across all QConv2d layers.

    Iterates over all modules in the model, sums qbits() from every QConv2d,
    then divides by the total parameter count (all parameters, including
    weight, e, and b tensors — matching the reference implementation).

    This is the Q used in the training loss: L = L_task + λ * Q

    Parameters
    ----------
    model : nn.Module — any model containing QConv2d layers

    Returns
    -------
    torch.Tensor — scalar, average bits per parameter
        At initialization (b=2.0): Q ≈ 2.0
        After self-compression training: Q ≈ 1.64 (reference result)
    """
    ###########################################################
    # YOUR CODE HERE — 8-12 lines                              #
    #                                                          #
    # Step 1: Collect all QConv2d instances from the model     #
    #         Use model.modules() which iterates recursively   #
    #         Filter with isinstance(m, QConv2d)              #
    #                                                          #
    # Step 2: Sum qbits() from each QConv2d layer             #
    #         Use functools.reduce or a loop with accumulation  #
    #         Handle the edge case: no QConv2d layers          #
    #                                                          #
    # Step 3: Count total parameters in the model              #
    #         sum(p.numel() for p in model.parameters())       #
    #         This counts weight + e + b for each QConv2d,    #
    #         matching the reference: 87,860 total params       #
    #                                                          #
    # Step 4: Return total_bits / total_param_count            #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_model_bytes(Q: float, weight_count: int) -> float:
    """Convert average bits per weight to total model size in bytes.

    Given Q bits per parameter and a total parameter count, compute
    the number of bytes required to store all weights at that precision.

    Parameters
    ----------
    Q            : float — average bits per parameter (e.g. 2.0 at init, 1.64 trained)
    weight_count : int   — total number of parameters (e.g. 87,860)

    Returns
    -------
    float — model size in bytes
        At Q=2.0, weight_count=87860: 87860 * 2 / 8 = 21965 bytes ≈ 21.5 KB
        At Q=1.64, weight_count=87860: 87860 * 1.64 / 8 = 18017 bytes ≈ 17.6 KB
    """
    ###########################################################
    # YOUR CODE HERE — 2-3 lines                               #
    #                                                          #
    # Formula: model_bytes = Q / 8 * weight_count             #
    # This converts bits to bytes (8 bits per byte)           #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Reference MNIST model architecture (matches the paper exactly)
# ---------------------------------------------------------------------------

class MNISTModel(nn.Module):
    """Self-compressing CNN for MNIST — reference architecture from the paper.

    Architecture: 5 QConv2d layers + BatchNorm + max pooling
        Layer 0: QConv2d(1,   32, 5) → relu
        Layer 2: QConv2d(32,  32, 5) → relu → BatchNorm → MaxPool
        Layer 6: QConv2d(32,  64, 3) → relu
        Layer 8: QConv2d(64,  64, 3) → relu → BatchNorm → MaxPool
        Layer 13: QConv2d(576, 10, 1) → flatten (classifier)

    Total weight parameters: 87,860 (including e and b for all layers)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = QConv2d(1,   32, 5)
        self.conv2 = QConv2d(32,  32, 5)
        self.bn1   = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.conv3 = QConv2d(32,  64, 3)
        self.conv4 = QConv2d(64,  64, 3)
        self.bn2   = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.conv5 = QConv2d(576, 10, 1)   # classifier (applied after flatten+reshape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))          # (N, 32, 24, 24)
        x = torch.relu(self.conv2(x))          # (N, 32, 20, 20)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)                 # (N, 32, 10, 10)
        x = torch.relu(self.conv3(x))          # (N, 64,  8,  8)
        x = torch.relu(self.conv4(x))          # (N, 64,  6,  6)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)                 # (N, 64,  3,  3) → 576 flat
        x = x.flatten(1).reshape(-1, 576, 1, 1)
        x = self.conv5(x)                       # (N, 10, 1, 1)
        return x.flatten(1)                     # (N, 10)


# ---------------------------------------------------------------------------
# __main__ — validation harness (do not modify)
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
    # fan_in = 1 * 5 * 5 = 25, sum(relu(b)) = 32 * 2.0 = 64, total = 1600
    expected_qbits = 1600.0
    qbits_ok = abs(qb.item() - expected_qbits) < 1.0
    print(f"\n[Test 1] Single layer qbits() — QConv2d(1, 32, 5), b=2.0")
    print(f"  fan_in = 1*5*5 = 25, sum(relu(b)) = 32*2.0 = 64.0")
    print(f"  Expected qbits: {expected_qbits:.1f}")
    print(f"  Got     qbits: {qb.item():.1f}")
    print(f"  Result        : {'✓ PASS' if qbits_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 2: qbits() for QConv2d(32, 64, 3) — larger layer
    # -------------------------------------------------------------------
    layer2 = QConv2d(in_channels=32, out_channels=64, kernel_size=3)
    qb2 = layer2.qbits()
    # fan_in = 32 * 3 * 3 = 288, sum(relu(b)) = 64 * 2.0 = 128, total = 36864
    expected_qbits2 = 36864.0
    qbits2_ok = abs(qb2.item() - expected_qbits2) < 1.0
    print(f"\n[Test 2] Layer qbits() — QConv2d(32, 64, 3), b=2.0")
    print(f"  fan_in = 32*3*3 = 288, sum(relu(b)) = 64*2.0 = 128.0")
    print(f"  Expected qbits: {expected_qbits2:.1f}")
    print(f"  Got     qbits: {qb2.item():.1f}")
    print(f"  Result        : {'✓ PASS' if qbits2_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 3: qbits() with one pruned channel (b < 0)
    # -------------------------------------------------------------------
    layer3 = QConv2d(in_channels=1, out_channels=4, kernel_size=3)
    with torch.no_grad():
        layer3.b[2] = -0.5   # channel 2: pruned
    qb3 = layer3.qbits()
    # fan_in = 9, sum(relu(b)) = 3 * 2.0 + relu(-0.5) = 6.0 + 0 = 6.0
    # total = 6.0 * 9 = 54.0
    expected_qbits3 = 54.0
    qbits3_ok = abs(qb3.item() - expected_qbits3) < 0.1
    print(f"\n[Test 3] qbits() with pruned channel (b=-0.5 for ch 2)")
    print(f"  Expected: 3 active * 2.0 * 9 = {expected_qbits3:.1f}")
    print(f"  Got     : {qb3.item():.4f}")
    print(f"  Result  : {'✓ PASS' if qbits3_ok else '✗ FAIL'}")

    # -------------------------------------------------------------------
    # Test 4: Full MNIST model — avg bits per weight and model size
    # -------------------------------------------------------------------
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Test 4] Full MNIST model metrics")
    print(f"  Total parameters: {total_params:,}")

    Q = compute_avg_bits(model)
    Q_val = Q.item() if hasattr(Q, 'item') else float(Q)
    q_ok = abs(Q_val - 2.0) < 0.05   # at init, b=2.0 → Q≈2.0

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
    # Test 6: qbits() is differentiable (gradient w.r.t. b)
    # -------------------------------------------------------------------
    layer4 = QConv2d(in_channels=3, out_channels=8, kernel_size=3)
    qb4 = layer4.qbits()
    qb4.backward()
    b_grad = layer4.b.grad
    # Gradient of relu(b).sum() * fan_in w.r.t. b_i is fan_in when b_i > 0
    expected_grad = float(math.prod(layer4.weight.shape[1:]))   # 27
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

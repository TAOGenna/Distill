"""
Exercise 3: Analyze Learned Bit-Width Distributions — SOLUTION
==============================================================
Module 3 — Self-Compression Training on MNIST

Key finding from reference implementation:
    layers.0.b (32, 1, 1, 1) [ 2.5146  2.3283  -0.0062  2.3243  2.8054 ... ]
    Channel 2: b = -0.0062 -> relu(b) = 0 -> PRUNED
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict


# ──────────────────────────────────────────────────────────────────────────────
# QConv2d + model
# ──────────────────────────────────────────────────────────────────────────────

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class QConv2d(nn.Module):
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


class SelfCompressingCNN(nn.Module):
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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.bn1(F.relu(self.conv2(x))), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(self.bn2(F.relu(self.conv4(x))), 2)
        x = x.flatten(1).reshape(x.shape[0], 576, 1, 1)
        return self.conv5(x).flatten(1)


def compute_compression_term(model: nn.Module) -> torch.Tensor:
    total_bits = torch.tensor(0.0)
    weight_count = 0
    for layer in model.modules():
        if isinstance(layer, QConv2d):
            total_bits = total_bits + layer.qbits()
            weight_count += layer.weight.numel()
    return total_bits / weight_count


def quick_train(model: nn.Module, steps: int = 5_000, lam: float = 0.05) -> None:
    from tqdm import tqdm
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = torchvision.datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters())
    train_iter = iter(loader)

    for step in tqdm(range(steps), desc=f"Quick training ({steps} steps)"):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            images, labels = next(train_iter)

        model.train()
        optimizer.zero_grad()
        logits = model(images)
        task_loss = F.cross_entropy(logits, labels)
        Q = compute_compression_term(model)
        loss = task_loss + lam * Q
        loss.backward()
        optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────
# SOLUTION implementations
# ──────────────────────────────────────────────────────────────────────────────

def extract_bit_widths(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract effective bit-widths (relu(b)) from all QConv2d layers.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: layer names, values: 1D arrays of effective bit-widths (>= 0).
    """
    bit_widths = {}
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, QConv2d):
                # b has shape (out_channels, 1, 1, 1)
                b_vals = layer.b.flatten()
                # Apply relu: negative b → 0 bits (pruned)
                eff_b = torch.relu(b_vals).numpy()
                bit_widths[name] = eff_b
    return bit_widths


def compute_layer_stats(
    bit_widths_dict: Dict[str, np.ndarray]
) -> Dict[str, dict]:
    """Compute per-layer statistics about learned bit-width distributions.

    Returns
    -------
    dict[str, dict]
        Per-layer statistics: num_channels, num_pruned, num_active,
        mean_bits, max_bits, compression_ratio.
    """
    stats = {}
    for name, eff_b in bit_widths_dict.items():
        num_channels = len(eff_b)
        num_pruned   = int(np.sum(eff_b <= 0.01))
        num_active   = num_channels - num_pruned
        mean_bits    = float(np.mean(eff_b))
        max_bits     = float(np.max(eff_b))
        compression_ratio = 32.0 / mean_bits if mean_bits > 0.0 else float('inf')

        stats[name] = {
            'num_channels':      num_channels,
            'num_pruned':        num_pruned,
            'num_active':        num_active,
            'mean_bits':         mean_bits,
            'max_bits':          max_bits,
            'compression_ratio': compression_ratio,
        }
    return stats


def visualize_bit_distributions(
    bit_widths_dict: Dict[str, np.ndarray],
    save_path: str = "bitwidth_distributions.png",
) -> None:
    """Create a multi-panel histogram of bit-width distributions per layer."""
    n = len(bit_widths_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, eff_b) in zip(axes, bit_widths_dict.items()):
        pruned_mask = eff_b <= 0.01

        if pruned_mask.any():
            ax.hist(
                eff_b[pruned_mask], bins=10,
                color='crimson', alpha=0.75,
                label=f'{pruned_mask.sum()} pruned'
            )
        if (~pruned_mask).any():
            ax.hist(
                eff_b[~pruned_mask], bins=20,
                color='royalblue', alpha=0.75,
                label=f'{(~pruned_mask).sum()} active'
            )

        ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.6, label='prune threshold')
        ax.set_title(name)
        ax.set_xlabel('Effective bits (relu(b))')
        ax.set_ylabel('# channels')
        ax.legend(fontsize=8)

    plt.suptitle('Learned Bit-Width Distributions per QConv2d Layer', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 65)
    print("Exercise 3: Analyze Learned Bit-Width Distributions")
    print("=" * 65)

    model = SelfCompressingCNN()
    saved_model_path = os.path.join(os.path.dirname(__file__), "trained_model.pt")

    if os.path.exists(saved_model_path):
        print(f"\n[1] Loading trained model from: {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))
        print("    Loaded  ✓")
    else:
        print("\n[1] No saved model found — training for 5,000 steps (quick mode).")
        print("    For full reference results, run Exercise 2 first.")
        quick_train(model, steps=5_000, lam=0.05)

    model.eval()

    # Extract bit-widths
    print("\n[2] Extracting learned bit-widths from each QConv2d layer:")
    bit_widths = extract_bit_widths(model)

    assert isinstance(bit_widths, dict), "extract_bit_widths must return a dict"
    assert len(bit_widths) == 5, f"Expected 5 QConv2d layers, got {len(bit_widths)}"
    for name, bw in bit_widths.items():
        assert isinstance(bw, np.ndarray), f"{name}: expected np.ndarray"
        assert (bw >= 0).all(), f"{name}: bit-widths must be >= 0 (relu applied)"
        print(f"    {name}: {len(bw)} channels, "
              f"range [{bw.min():.3f}, {bw.max():.3f}] bits")

    # Compute stats
    print("\n[3] Per-layer statistics:")
    stats = compute_layer_stats(bit_widths)

    total_pruned = 0
    total_channels = 0
    for name, s in stats.items():
        n_ch = s['num_channels']
        n_pr = s['num_pruned']
        n_ac = s['num_active']
        mean = s['mean_bits']
        maxi = s['max_bits']
        ratio = s['compression_ratio']
        total_pruned += n_pr
        total_channels += n_ch
        ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "∞"
        print(f"    {name}: {n_ch} channels, "
              f"{n_pr} pruned, {n_ac} active, "
              f"mean_bits={mean:.2f}, max_bits={maxi:.2f}, "
              f"compression={ratio_str}")

    active_bws = []
    for name, bw in bit_widths.items():
        active_bws.extend(bw[bw > 0.01].tolist())
    avg_active_bits = np.mean(active_bws) if active_bws else 0.0

    print(f"\n    Network learned to prune {total_pruned} of {total_channels} channels, "
          f"averaging {avg_active_bits:.2f} bits/weight for remaining channels")

    # Visualize
    print("\n[4] Generating bit-width distribution plots...")
    plot_path = os.path.join(os.path.dirname(__file__), "bitwidth_distributions.png")
    visualize_bit_distributions(bit_widths, save_path=plot_path)
    print(f"    Saved: {plot_path}")

    # Overall Q
    Q = compute_compression_term(model).item()
    weight_count = sum(l.weight.numel() for l in model.modules() if isinstance(l, QConv2d))
    model_bytes = Q / 8 * weight_count
    print(f"\n[5] Overall compression: Q={Q:.4f} bits/weight, "
          f"model size={model_bytes:.1f} bytes")

    print("\n✓ Analysis complete!")

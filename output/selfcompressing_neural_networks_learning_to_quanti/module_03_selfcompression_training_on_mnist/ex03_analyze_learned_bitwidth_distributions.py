"""
Exercise 3: Analyze Learned Bit-Width Distributions
=====================================================
Module 3 — Self-Compression Training on MNIST

After training the self-compressing CNN (Exercise 2), we inspect the learned
bit-width parameters to understand HOW the network compressed itself.

Key insight: The learned bit-width distribution is NON-UNIFORM:
  - Some channels get pruned (b driven below 0, relu(b) = 0 bits)
  - Active channels retain 2-4 bits depending on their importance
  - The final classifier layer (576→10) resists pruning — accuracy depends on it
  - The distribution is bimodal: cluster near 0 (pruned) + cluster at 2-3 (active)

From the reference notebook (layers.0.b values after training):
  [ 2.5146  2.3283  -0.0062  2.3243  2.8054  3.2020  2.3323  3.1703  3.5180  2.9075 ]
  Channel 2 has b = -0.0062 → relu(b) = 0 → effectively PRUNED

Your tasks:
  1. extract_bit_widths(model) — extract relu(b) from each QConv2d layer
  2. compute_layer_stats(bit_widths_dict) — compute pruned/active/mean/max per layer
  3. visualize_bit_distributions(bit_widths_dict) — multi-panel histogram

This exercise can run in two modes:
  a) Standalone: trains a short model (5,000 steps, lower accuracy) for quick analysis
  b) Load saved model from Exercise 2 (if trained_model.pt exists in _solutions/)
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
# QConv2d + model (provided — same as Exercise 2)
# ──────────────────────────────────────────────────────────────────────────────

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class QConv2d(nn.Module):
    """Quantization-Aware Conv2d with per-channel learnable (e, b) parameters."""

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
    """Five-layer QConv2d CNN for MNIST (same architecture as Exercise 2)."""

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


# ──────────────────────────────────────────────────────────────────────────────
# PROVIDED: Quick training for standalone mode
# ──────────────────────────────────────────────────────────────────────────────

def quick_train(model: nn.Module, steps: int = 5_000, lam: float = 0.05) -> None:
    """Train for a reduced number of steps for quick analysis.

    Note: 5,000 steps achieves ~95% accuracy and partial compression.
    For the full reference result, run Exercise 2 first and load its model.
    """
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
# YOUR CODE — implement the three analysis functions below
# ──────────────────────────────────────────────────────────────────────────────

def extract_bit_widths(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract effective bit-widths (relu(b)) from all QConv2d layers.

    IMPORTANT: Use relu(b), not raw b. Negative b values are treated as 0 bits
    because relu clamps them. Plotting raw b would show negative values which
    have no physical meaning as bit-widths.

    Parameters
    ----------
    model : nn.Module
        A trained SelfCompressingCNN (or any model with QConv2d layers).

    Returns
    -------
    dict[str, np.ndarray]
        Keys: layer names (e.g. 'conv1', 'conv2', ...)
        Values: 1D numpy arrays of effective bit-widths per output channel.
                Shape: (out_channels,). Values >= 0 (relu applied).

    Notes
    -----
    - Use model.named_modules() to get (name, layer) pairs.
    - Only include QConv2d layers (isinstance check).
    - layer.b has shape (out_channels, 1, 1, 1). Call .flatten() then .numpy().
    - Apply relu: np.maximum(b_vals, 0) or use F.relu before .detach().numpy().
    - Wrap in torch.no_grad() to avoid tracking gradients.
    """
    ###########################################################################
    # YOUR CODE HERE — 8-12 lines                                             #
    #                                                                         #
    # bit_widths = {}                                                          #
    # with torch.no_grad():                                                   #
    #     for name, layer in model.named_modules():                           #
    #         if isinstance(layer, QConv2d):                                  #
    #             b_vals = layer.b.flatten()   # shape: (out_channels,)       #
    #             eff_b  = torch.relu(b_vals).numpy()   # shape: (out_ch,)   #
    #             bit_widths[name] = eff_b                                    #
    # return bit_widths                                                        #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


def compute_layer_stats(
    bit_widths_dict: Dict[str, np.ndarray]
) -> Dict[str, dict]:
    """Compute per-layer statistics about learned bit-width distributions.

    Parameters
    ----------
    bit_widths_dict : dict[str, np.ndarray]
        From extract_bit_widths(). Keys: layer names, values: bit-width arrays.

    Returns
    -------
    dict[str, dict]
        For each layer name, a dict with:
          'num_channels'   : int   — total output channels
          'num_pruned'     : int   — channels with eff_b <= 0.01 (effectively 0)
          'num_active'     : int   — channels with eff_b > 0.01
          'mean_bits'      : float — mean eff_b over all channels
          'max_bits'       : float — maximum eff_b
          'compression_ratio' : float — 32 / mean_bits  (vs float32 baseline)

    Notes
    -----
    - A channel is "pruned" if its effective bit-width <= 0.01 (not strictly 0
      because floating-point b may be tiny but not exactly 0 after relu).
    - compression_ratio = 32 / mean_bits tells you how much smaller this layer
      is vs storing weights in float32. Use mean_bits in the denominator.
    - If mean_bits == 0 (all pruned), set compression_ratio = float('inf').
    """
    ###########################################################################
    # YOUR CODE HERE — 10-15 lines                                            #
    #                                                                         #
    # stats = {}                                                               #
    # for name, eff_b in bit_widths_dict.items():                             #
    #     num_channels = len(eff_b)                                           #
    #     num_pruned   = int(np.sum(eff_b <= 0.01))                          #
    #     num_active   = num_channels - num_pruned                            #
    #     mean_bits    = float(np.mean(eff_b))                                #
    #     max_bits     = float(np.max(eff_b))                                 #
    #     comp_ratio   = 32 / mean_bits if mean_bits > 0 else float('inf')   #
    #     stats[name]  = {...}                                                 #
    # return stats                                                             #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


def visualize_bit_distributions(
    bit_widths_dict: Dict[str, np.ndarray],
    save_path: str = "bitwidth_distributions.png",
) -> None:
    """Create a multi-panel histogram of bit-width distributions per layer.

    One subplot per QConv2d layer. Each histogram shows:
      - X axis: effective bit-width (0 to max_bits+0.5)
      - Y axis: number of channels
      - Red bars: pruned channels (eff_b <= 0.01)
      - Blue bars: active channels (eff_b > 0.01)
      - Title: layer name + (N_pruned pruned / N_active active)

    Parameters
    ----------
    bit_widths_dict : dict[str, np.ndarray]
    save_path : str
        Where to save the figure.

    Notes
    -----
    - Use plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    - For each layer, plot a single histogram using ax.hist(eff_b, bins=20, ...)
    - Color channels by whether they are pruned or active:
        pruned_mask = eff_b <= 0.01
        ax.hist(eff_b[pruned_mask], bins=10, color='red', label='pruned')
        ax.hist(eff_b[~pruned_mask], bins=20, color='royalblue', label='active')
    - Add ax.axvline(x=0.01, color='gray', linestyle='--') threshold line
    - Add labels: ax.set_xlabel('Effective bits'), ax.set_ylabel('# channels')
    """
    ###########################################################################
    # YOUR CODE HERE — 15-20 lines                                            #
    #                                                                         #
    # n = len(bit_widths_dict)                                                 #
    # fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)         #
    # if n == 1: axes = [axes]                                                #
    # for ax, (name, eff_b) in zip(axes, bit_widths_dict.items()):           #
    #     pruned_mask = eff_b <= 0.01                                         #
    #     if pruned_mask.any():                                               #
    #         ax.hist(eff_b[pruned_mask], bins=10, color='red', alpha=0.7,   #
    #                 label=f'{pruned_mask.sum()} pruned')                    #
    #     ax.hist(eff_b[~pruned_mask], bins=20, color='royalblue', alpha=0.7,#
    #             label=f'{(~pruned_mask).sum()} active')                     #
    #     ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5)        #
    #     ax.set_title(name); ax.set_xlabel('Effective bits')                 #
    #     ax.legend(fontsize=8)                                               #
    # plt.suptitle('Learned Bit-Width Distributions per Layer')              #
    # plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()       #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ──────────────────────────────────────────────────────────────────────────────
# Main — DO NOT MODIFY
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 65)
    print("Exercise 3: Analyze Learned Bit-Width Distributions")
    print("=" * 65)

    # Load or train model
    model = SelfCompressingCNN()
    saved_model_path = os.path.join(os.path.dirname(__file__), "_solutions", "trained_model.pt")

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

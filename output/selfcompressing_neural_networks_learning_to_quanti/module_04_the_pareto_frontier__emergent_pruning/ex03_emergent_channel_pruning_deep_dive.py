"""
Exercise 3: Emergent Channel Pruning Deep Dive
==============================================
Module 4 — The Pareto Frontier & Emergent Pruning

After training, examine WHICH channels the network chose to prune and WHY.
The self-compressing network drives some channels' bit-widths b to negative
values. After relu(b)=0, the channel contributes exactly zero to all outputs —
structurally pruned without any explicit pruning algorithm.

Reference output from the paper's notebook (layers.2.b, first 10 values):
    [ 2.3977  2.3232  2.3248  -0.0090  -0.0067  2.3268  2.3121  -0.0070  -0.0092  2.8098 ]

Key question: Are pruned channels pruned because they are (a) low-magnitude
(small weights, little contribution) or (b) redundant (highly similar to
another active channel)?
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
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────────────────
# QConv2d + SelfCompressingCNN
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
        return torch.relu(self.b).sum() * math.prod(self.weight.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self.qweight()
        w = ste_round(qw)
        return F.conv2d(x, (2 ** self.e) * w, stride=self.stride, padding=self.padding)


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
    total_bits, weight_count = torch.tensor(0.0), 0
    for layer in model.modules():
        if isinstance(layer, QConv2d):
            total_bits = total_bits + layer.qbits()
            weight_count += layer.weight.numel()
    return total_bits / weight_count


def get_mnist_loaders(batch_size: int = 512, data_dir: str = "/tmp/mnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size=2000,       shuffle=False, num_workers=0))


def quick_train(model: nn.Module, steps: int = 5_000, lam: float = 0.05) -> None:
    """Quick training loop for when no saved model exists."""
    train_loader, _ = get_mnist_loaders(batch_size=512)
    optimizer = torch.optim.Adam(model.parameters())
    train_iter = iter(train_loader)
    from tqdm import trange
    for _ in trange(steps, desc=f"Training ({steps} steps)"):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels) + lam * compute_compression_term(model)
        loss.backward()
        optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────
# YOUR CODE — 3 functions to implement
# ──────────────────────────────────────────────────────────────────────────────

def identify_pruned_channels(model: nn.Module) -> Dict[str, dict]:
    """For each QConv2d layer, identify pruned and active channels.

    A channel is pruned if relu(b) ≤ 0.01 (effectively zero bits).

    Parameters
    ----------
    model : nn.Module
        Trained SelfCompressingCNN.

    Returns
    -------
    dict[str, dict]
        Keys: layer names (e.g. 'conv1', 'conv2', ...).
        Values: dicts with:
          - 'pruned_indices': list[int] — indices of pruned output channels
          - 'active_indices': list[int] — indices of active output channels
          - 'b_values': np.ndarray shape (C_out,) — raw b values (not relu'd)
          - 'n_pruned': int
          - 'n_active': int

    Notes
    -----
    b has shape (out_channels, 1, 1, 1) — flatten it before indexing.
    The prune threshold is 0.01 (relu(b) = 0 exactly, but float imprecision
    means we use a small epsilon).
    """
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint: use model.named_modules() to iterate QConv2d      #
    # layers. For each layer:                                 #
    #   b_flat = layer.b.detach().flatten()                   #
    #   pruned_indices = (b_flat <= 0.01).nonzero()...        #
    # Return a dict keyed by layer name.                      #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def analyze_pruning_pattern(
    model: nn.Module,
    pruning_info: Dict[str, dict],
) -> Dict[str, dict]:
    """For each layer, analyze WHY channels were pruned.

    Two hypotheses:
      H1 (low magnitude): pruned channels have smaller weight L2 norms
      H2 (redundancy):    pruned channels are highly correlated with active channels

    For redundancy, compute the maximum cosine similarity between each pruned
    channel's weight vector and ALL active channel weight vectors. If this max
    similarity > 0.8, the pruned channel is "redundant."

    Parameters
    ----------
    model : nn.Module
        Trained SelfCompressingCNN.
    pruning_info : dict
        Output of identify_pruned_channels().

    Returns
    -------
    dict[str, dict]
        Per-layer analysis with keys:
          - 'pruned_mean_norm': float — mean L2 norm of pruned channel weights
          - 'active_mean_norm': float — mean L2 norm of active channel weights
          - 'redundant_count': int    — number of pruned channels with max cosine > 0.8
          - 'max_similarities': list[float] — max cosine sim per pruned channel

    Notes
    -----
    weight shape: (C_out, C_in, kH, kW). For channel c, use weight[c] flattened.
    Cosine similarity: dot(u, v) / (||u|| * ||v||). Use F.normalize() and matmul.
    Skip layers with zero pruned channels.
    """
    ###########################################################
    # YOUR CODE HERE - 20-30 lines                            #
    #                                                         #
    # Hint: for each layer in pruning_info:                   #
    #   Get weight tensor (C_out, ...), flatten to (C_out, D) #
    #   pruned_w = weight[pruned_indices]  shape: (N_pr, D)   #
    #   active_w = weight[active_indices]  shape: (N_ac, D)   #
    #   Compute norms: pruned_w.norm(dim=1), active_w.norm... #
    #   Cosine sim: normalize then matmul                      #
    #   max_sim per pruned channel = max along active axis     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def visualize_pruning(
    model: nn.Module,
    pruning_info: Dict[str, dict],
    save_path: str = "pruning_analysis.png",
) -> None:
    """Multi-panel visualization of channel pruning.

    Panel layout (1 row, 3 panels):
      1. Bar chart: b-values per channel in conv2 (color: red=pruned, blue=active)
      2. Histogram: weight L2 norms for pruned vs active channels (all layers pooled)
      3. Similarity matrix: pairwise cosine similarity between ALL channels in conv2

    Parameters
    ----------
    model : nn.Module
        Trained SelfCompressingCNN.
    pruning_info : dict
        Output of identify_pruned_channels().
    save_path : str
        Output filename.

    Notes
    -----
    Panel 1: x-axis = channel index, y-axis = b value, red bars for pruned.
    Panel 2: overlapping histograms (blue=active, red=pruned).
    Panel 3: imshow of (C_out, C_out) cosine similarity matrix for conv2.
    """
    ###########################################################
    # YOUR CODE HERE - 15-20 lines                            #
    #                                                         #
    # Hint: fig, axes = plt.subplots(1, 3, figsize=(15, 5))  #
    # Panel 1: use conv2 info from pruning_info['conv2']      #
    # Panel 2: pool norms from all layers                     #
    # Panel 3: compute cosine similarity for conv2.weight     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ──────────────────────────────────────────────────────────────────────────────
# Main — provided, do not modify
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    OUT_DIR = os.path.dirname(__file__)

    print("=" * 65)
    print("Exercise 3: Emergent Channel Pruning Deep Dive")
    print("=" * 65)

    # Load or train model
    model = SelfCompressingCNN()
    # Look for model saved by Module 3's exercise 2
    module3_path = os.path.join(
        os.path.dirname(__file__), "..",
        "module_03_selfcompression_training_on_mnist",
        "_solutions", "trained_model.pt"
    )
    local_path = os.path.join(OUT_DIR, "trained_model.pt")

    if os.path.exists(local_path):
        print(f"\n[1] Loading model from: {local_path}")
        model.load_state_dict(torch.load(local_path, weights_only=True))
    elif os.path.exists(module3_path):
        print(f"\n[1] Loading model from Module 3: {module3_path}")
        model.load_state_dict(torch.load(module3_path, weights_only=True))
    else:
        STEPS = int(os.environ.get("TRAIN_STEPS", "5000"))
        print(f"\n[1] No saved model found — training for {STEPS} steps...")
        quick_train(model, steps=STEPS, lam=0.05)
        torch.save(model.state_dict(), local_path)
        print(f"    Saved: {local_path}")

    model.train()  # keep BN in batch-stats mode

    # ── Identify pruned channels ──
    print("\n[2] Identifying pruned channels per layer:")
    pruning_info = identify_pruned_channels(model)

    assert isinstance(pruning_info, dict), "identify_pruned_channels must return a dict"
    total_pruned = 0
    total_channels = 0
    for name, info in pruning_info.items():
        n_pr = info['n_pruned']
        n_ac = info['n_active']
        total_pruned   += n_pr
        total_channels += n_pr + n_ac
        b_min = info['b_values'].min()
        b_max = info['b_values'].max()
        print(f"    {name}: {n_pr}/{n_pr+n_ac} pruned channels, "
              f"b range=[{b_min:.4f}, {b_max:.4f}]")

    print(f"\n    Total pruned channels: {total_pruned}/{total_channels} "
          f"({100*total_pruned/total_channels:.1f}%)")

    # ── Analyze pruning pattern ──
    print("\n[3] Analyzing WHY channels are pruned:")
    analysis = analyze_pruning_pattern(model, pruning_info)

    assert isinstance(analysis, dict), "analyze_pruning_pattern must return a dict"
    for name, stats in analysis.items():
        n_pruned = pruning_info[name]['n_pruned']
        n_redund = stats['redundant_count']
        print(f"    {name}:")
        print(f"      Pruned channels avg weight norm:  {stats['pruned_mean_norm']:.4f}")
        print(f"      Active channels avg weight norm:  {stats['active_mean_norm']:.4f}")
        print(f"      Redundant (cosine_sim > 0.8):    "
              f"{n_redund}/{n_pruned} pruned channels")

    # ── Print key finding ──
    print("\n[4] Summary — pruned channels are either:")
    print("    (a) Low-magnitude: small weights = small gradient signal")
    print("    (b) Redundant: high cosine similarity with an active channel")
    print()
    print("    The network prunes channels that are either low-magnitude or")
    print("    redundant with other channels, achieving structured sparsity")
    print("    as an emergent property of bit-width optimization.")

    # ── Visualize ──
    print("\n[5] Generating pruning visualizations...")
    plot_path = os.path.join(OUT_DIR, "pruning_analysis.png")
    visualize_pruning(model, pruning_info, save_path=plot_path)
    print(f"    Saved: {plot_path}")

    # ── Overall compression check ──
    Q = compute_compression_term(model).item()
    w_count = sum(l.weight.numel() for l in model.modules() if isinstance(l, QConv2d))
    model_bytes = Q / 8 * w_count
    print(f"\n[6] Overall: Q={Q:.4f} bits/weight, "
          f"model_bytes={model_bytes:.1f} bytes")
    print(f"    Float32 baseline: {w_count * 4} bytes ({w_count * 4 / model_bytes:.1f}x compression)")

    print(f"\n✓ Pruning analysis complete! Found {total_pruned} pruned channels "
          f"out of {total_channels} total.")

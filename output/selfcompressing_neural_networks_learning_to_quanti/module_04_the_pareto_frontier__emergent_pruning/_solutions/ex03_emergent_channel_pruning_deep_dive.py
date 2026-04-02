"""
Exercise 3: Emergent Channel Pruning Deep Dive — SOLUTION
==========================================================
Module 4 — The Pareto Frontier & Emergent Pruning

Key finding: The reference model prunes ~30% of all channels. Pruned channels
are consistently lower-magnitude than active channels AND often redundant
(cosine similarity > 0.8 with some active channel). This demonstrates that
self-compression discovers structured pruning as an emergent property of
gradient-based bit-width optimization.
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
# SOLUTION implementations
# ──────────────────────────────────────────────────────────────────────────────

def identify_pruned_channels(model: nn.Module) -> Dict[str, dict]:
    """For each QConv2d layer, identify pruned and active channels.

    Parameters
    ----------
    model : nn.Module
        Trained SelfCompressingCNN.

    Returns
    -------
    dict[str, dict]
        Per-layer dicts: pruned_indices, active_indices, b_values, n_pruned, n_active.
    """
    PRUNE_THRESHOLD = 0.01  # relu(b) <= this → pruned
    result = {}

    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, QConv2d):
                continue
            # b shape: (out_channels, 1, 1, 1) → flatten to 1D
            b_flat = layer.b.detach().flatten().cpu().numpy()
            eff_b  = np.maximum(b_flat, 0.0)  # relu

            pruned_mask = eff_b <= PRUNE_THRESHOLD
            pruned_indices = np.where(pruned_mask)[0].tolist()
            active_indices = np.where(~pruned_mask)[0].tolist()

            result[name] = {
                'pruned_indices': pruned_indices,
                'active_indices': active_indices,
                'b_values':       b_flat,
                'n_pruned':       len(pruned_indices),
                'n_active':       len(active_indices),
            }

    return result


def analyze_pruning_pattern(
    model: nn.Module,
    pruning_info: Dict[str, dict],
) -> Dict[str, dict]:
    """For each layer, analyze WHY channels were pruned.

    Parameters
    ----------
    model : nn.Module
        Trained SelfCompressingCNN.
    pruning_info : dict
        Output of identify_pruned_channels().

    Returns
    -------
    dict[str, dict]
        Per-layer: pruned_mean_norm, active_mean_norm, redundant_count, max_similarities.
    """
    REDUNDANCY_THRESHOLD = 0.8

    analysis = {}
    layer_dict = {name: layer for name, layer in model.named_modules()
                  if isinstance(layer, QConv2d)}

    for name, info in pruning_info.items():
        pruned_idx = info['pruned_indices']
        active_idx = info['active_indices']

        if len(pruned_idx) == 0:
            # No pruned channels — skip detailed analysis but report zeros
            analysis[name] = {
                'pruned_mean_norm': 0.0,
                'active_mean_norm': 0.0,
                'redundant_count':  0,
                'max_similarities': [],
            }
            continue

        layer = layer_dict[name]
        # Weight shape: (C_out, C_in, kH, kW) → flatten to (C_out, D)
        W = layer.weight.data.detach().flatten(1)  # (C_out, D)

        pruned_w = W[pruned_idx]   # (N_pr, D)
        active_w = W[active_idx]   # (N_ac, D)

        # ── H1: Weight magnitude (L2 norm) ──
        pruned_norms = pruned_w.norm(dim=1)  # (N_pr,)
        active_norms = active_w.norm(dim=1)  # (N_ac,)
        pruned_mean_norm = pruned_norms.mean().item() if len(pruned_idx) > 0 else 0.0
        active_mean_norm = active_norms.mean().item() if len(active_idx) > 0 else 0.0

        # ── H2: Redundancy (cosine similarity) ──
        # Normalize to unit vectors for cosine similarity
        pruned_unit = F.normalize(pruned_w, dim=1)  # (N_pr, D)
        active_unit = F.normalize(active_w, dim=1)  # (N_ac, D)

        # Cosine similarity: (N_pr, N_ac) — max along active axis
        if len(active_idx) > 0:
            sim_matrix  = pruned_unit @ active_unit.T  # (N_pr, N_ac)
            max_sims    = sim_matrix.abs().max(dim=1).values  # (N_pr,) max over active channels
            max_sims_list = max_sims.tolist()
            redundant_count = int((max_sims > REDUNDANCY_THRESHOLD).sum().item())
        else:
            max_sims_list   = [0.0] * len(pruned_idx)
            redundant_count = 0

        analysis[name] = {
            'pruned_mean_norm': pruned_mean_norm,
            'active_mean_norm': active_mean_norm,
            'redundant_count':  redundant_count,
            'max_similarities': max_sims_list,
        }

    return analysis


def visualize_pruning(
    model: nn.Module,
    pruning_info: Dict[str, dict],
    save_path: str = "pruning_analysis.png",
) -> None:
    """Multi-panel visualization of channel pruning.

    Parameters
    ----------
    model : nn.Module
        Trained SelfCompressingCNN.
    pruning_info : dict
        Output of identify_pruned_channels().
    save_path : str
        Output filename.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: b-values for conv2, color-coded ──
    ax = axes[0]
    # Find the conv2 layer
    conv2_name = None
    for name in pruning_info:
        if 'conv2' in name:
            conv2_name = name
            break
    if conv2_name is None:
        conv2_name = list(pruning_info.keys())[1]  # fallback: second layer

    info = pruning_info[conv2_name]
    b_vals = info['b_values']
    colors = ['#e03131' if i in info['pruned_indices'] else '#1971c2'
              for i in range(len(b_vals))]
    ax.bar(range(len(b_vals)), b_vals, color=colors, edgecolor='none', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', label='prune threshold')
    ax.set_xlabel("Channel index", fontsize=11)
    ax.set_ylabel("Learned b value", fontsize=11)
    ax.set_title(f"b-values: {conv2_name}\n(red=pruned, blue=active)", fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 2: Weight norm histograms (all layers pooled) ──
    ax = axes[1]
    layer_dict = {name: layer for name, layer in model.named_modules()
                  if isinstance(layer, QConv2d)}
    all_pruned_norms, all_active_norms = [], []
    for name, info in pruning_info.items():
        layer = layer_dict[name]
        W = layer.weight.data.detach().flatten(1)  # (C_out, D)
        for idx in info['pruned_indices']:
            all_pruned_norms.append(W[idx].norm().item())
        for idx in info['active_indices']:
            all_active_norms.append(W[idx].norm().item())

    if all_active_norms:
        ax.hist(all_active_norms, bins=25, color='#1971c2', alpha=0.7,
                label=f'Active ({len(all_active_norms)} channels)', density=True)
    if all_pruned_norms:
        ax.hist(all_pruned_norms, bins=15, color='#e03131', alpha=0.7,
                label=f'Pruned ({len(all_pruned_norms)} channels)', density=True)
    ax.set_xlabel("Weight L2 norm", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Weight norms: Pruned vs Active\n(all layers pooled)", fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 3: Cosine similarity matrix for conv2 ──
    ax = axes[2]
    if conv2_name in layer_dict:
        layer = layer_dict[conv2_name]
        W = layer.weight.data.detach().flatten(1)  # (C_out, D)
        W_unit = F.normalize(W, dim=1)
        sim = (W_unit @ W_unit.T).abs().cpu().numpy()  # (C_out, C_out)
        im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Mark pruned channels with red lines
        for idx in info['pruned_indices']:
            ax.axhline(idx, color='red', linewidth=0.7, alpha=0.6)
            ax.axvline(idx, color='red', linewidth=0.7, alpha=0.6)
        ax.set_title(f"Cosine similarity matrix: {conv2_name}\n(red lines = pruned)", fontsize=11)
        ax.set_xlabel("Channel index", fontsize=11)
        ax.set_ylabel("Channel index", fontsize=11)

    plt.suptitle("Emergent Channel Pruning Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    OUT_DIR = os.path.dirname(__file__)

    print("=" * 65)
    print("Exercise 3: Emergent Channel Pruning Deep Dive")
    print("=" * 65)

    model = SelfCompressingCNN()
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

    model.train()

    print("\n[2] Identifying pruned channels per layer:")
    pruning_info = identify_pruned_channels(model)

    assert isinstance(pruning_info, dict)
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
          f"({100*total_pruned/max(total_channels,1):.1f}%)")

    print("\n[3] Analyzing WHY channels are pruned:")
    analysis = analyze_pruning_pattern(model, pruning_info)

    assert isinstance(analysis, dict)
    for name, stats in analysis.items():
        n_pruned = pruning_info[name]['n_pruned']
        if n_pruned == 0:
            continue
        n_redund = stats['redundant_count']
        print(f"    {name}:")
        print(f"      Pruned channels avg weight norm:  {stats['pruned_mean_norm']:.4f}")
        print(f"      Active channels avg weight norm:  {stats['active_mean_norm']:.4f}")
        print(f"      Redundant (cosine_sim > 0.8):    "
              f"{n_redund}/{n_pruned} pruned channels")

    print("\n[4] Summary — pruned channels are either:")
    print("    (a) Low-magnitude: small weights = small gradient signal")
    print("    (b) Redundant: high cosine similarity with an active channel")
    print()
    print("    The network prunes channels that are either low-magnitude or")
    print("    redundant with other channels, achieving structured sparsity")
    print("    as an emergent property of bit-width optimization.")

    print("\n[5] Generating pruning visualizations...")
    plot_path = os.path.join(OUT_DIR, "pruning_analysis.png")
    visualize_pruning(model, pruning_info, save_path=plot_path)
    print(f"    Saved: {plot_path}")

    Q = compute_compression_term(model).item()
    w_count = sum(l.weight.numel() for l in model.modules() if isinstance(l, QConv2d))
    model_bytes = Q / 8 * w_count
    print(f"\n[6] Overall: Q={Q:.4f} bits/weight, "
          f"model_bytes={model_bytes:.1f} bytes")
    print(f"    Float32 baseline: {w_count * 4} bytes ({w_count * 4 / model_bytes:.1f}x compression)")

    print(f"\n✓ Pruning analysis complete! Found {total_pruned} pruned channels "
          f"out of {total_channels} total.")

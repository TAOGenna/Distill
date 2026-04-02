"""
Exercise 1: Map the Size-Accuracy Pareto Frontier
==================================================
Module 4 — The Pareto Frontier & Emergent Pruning

The training loss of our self-compressing CNN is:
    L = L_task + λ · Q
where Q is the average bits-per-weight and λ controls compression strength.

Different values of λ sweep out a curve in (model_bytes, accuracy) space —
the Pareto frontier. Your job is to:
  1. Sweep λ over a log-uniform range
  2. Train a fresh model for each λ
  3. Identify which configurations are Pareto-optimal
  4. Plot the frontier

Reference from the paper:
  λ=0.05 → ~98.2% accuracy @ ~18,075 bytes (20× compression from float32)
  The full frontier spans 3% to 100% of bits from 80% to 99% accuracy.
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
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# QConv2d + SelfCompressingCNN (provided — same as Module 3)
# ──────────────────────────────────────────────────────────────────────────────

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for rounding."""
    return (x.round() - x).detach() + x


class QConv2d(nn.Module):
    """Quantized Conv2d with per-channel learnable exponent e and bit-width b."""

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
    """5-layer QConv2d CNN for MNIST. 87,860 weight parameters."""

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
    """Average bits-per-weight across all QConv2d layers."""
    total_bits = torch.tensor(0.0)
    weight_count = 0
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


@torch.no_grad()
def get_test_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    model.train()  # keep BatchNorm in batch-stats mode
    correct = total = 0
    for images, labels in test_loader:
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def train_self_compressing(lam: float, steps: int = 3_000,
                           seed: int = 42) -> tuple[float, float]:
    """Train a fresh SelfCompressingCNN with the given lambda.

    Parameters
    ----------
    lam : float
        Compression regularization strength (λ in L = L_task + λ·Q).
    steps : int
        Number of training steps (gradient updates).
    seed : int
        Random seed for reproducible but independent initializations.

    Returns
    -------
    tuple[float, float]
        (accuracy_pct, model_bytes) — final test accuracy and model size.
    """
    torch.manual_seed(seed)
    model = SelfCompressingCNN()
    train_loader, test_loader = get_mnist_loaders(batch_size=512)
    optimizer = torch.optim.Adam(model.parameters())

    weight_count = sum(l.weight.numel() for l in model.modules() if isinstance(l, QConv2d))

    train_iter = iter(train_loader)
    for step in range(steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        optimizer.zero_grad()
        logits = model(images)
        task_loss = F.cross_entropy(logits, labels)
        Q = compute_compression_term(model)
        loss = task_loss + lam * Q
        loss.backward()
        optimizer.step()

    final_acc   = get_test_accuracy(model, test_loader)
    final_Q     = compute_compression_term(model).item()
    model_bytes = final_Q / 8 * weight_count
    return final_acc, model_bytes


# ──────────────────────────────────────────────────────────────────────────────
# YOUR CODE — 3 functions to implement
# ──────────────────────────────────────────────────────────────────────────────

def sweep_lambda(lam_values: list[float],
                 steps_per_run: int = 3_000) -> list[dict]:
    """Train a fresh model for each lambda value and record the results.

    Each lambda value should produce an independent run: fresh random init,
    fresh optimizer. Use train_self_compressing() for each run.

    Parameters
    ----------
    lam_values : list[float]
        Lambda values to sweep (compression regularization strengths).
    steps_per_run : int
        Training steps per lambda value.

    Returns
    -------
    list[dict]
        One dict per lambda with keys: 'lambda', 'accuracy', 'model_bytes'.
        Dicts should be sorted by model_bytes ascending (smallest model first).

    Example
    -------
    >>> results = sweep_lambda([0.01, 0.05, 0.1], steps_per_run=1000)
    >>> results[0].keys()
    dict_keys(['lambda', 'accuracy', 'model_bytes'])
    """
    ###########################################################
    # YOUR CODE HERE - 12-18 lines                            #
    #                                                         #
    # Hint: iterate over lam_values, call train_self_         #
    # compressing(lam, steps=steps_per_run) for each.        #
    # Print progress. Sort results by model_bytes ascending.  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def find_pareto_optimal(results: list[dict]) -> list[dict]:
    """Identify Pareto-optimal configurations from sweep results.

    A configuration is Pareto-optimal if no other configuration has
    BOTH smaller model_bytes AND higher accuracy.

    Parameters
    ----------
    results : list[dict]
        Output from sweep_lambda(): list of dicts with 'lambda',
        'accuracy', 'model_bytes'.

    Returns
    -------
    list[dict]
        The subset of results that are Pareto-optimal.

    Notes
    -----
    Since the sweep is monotone (higher lambda → smaller model AND lower
    accuracy), most points will be Pareto-optimal. Non-optimal points
    only occur if two runs accidentally flip the monotone ordering.
    """
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint: for each result r, check if any OTHER result      #
    # dominates it: other.bytes <= r.bytes AND other.acc >=   #
    # r.acc, with at least one strict inequality.             #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def plot_pareto_frontier(results: list[dict],
                         pareto: list[dict],
                         save_path: str = "pareto_frontier.png") -> None:
    """Scatter plot of model_bytes (x) vs accuracy (y), annotated with lambda.

    Parameters
    ----------
    results : list[dict]
        All sweep results (lambda, accuracy, model_bytes).
    pareto : list[dict]
        Pareto-optimal subset (for highlighting and connecting with a line).
    save_path : str
        Output filename for the saved figure.

    Notes
    -----
    Plot layout:
      - All points as scatter (grey or colored by lambda)
      - Pareto-optimal points highlighted (different color/marker)
      - Connect Pareto points with a line (sorted by model_bytes)
      - Annotate each point with its lambda value
      - x-axis: model size in bytes, y-axis: accuracy (%)
    """
    ###########################################################
    # YOUR CODE HERE - 12-15 lines                            #
    #                                                         #
    # Hint: use plt.scatter() for all points, then            #
    # plt.scatter() again for pareto-optimal with different   #
    # color. Sort pareto by model_bytes before plotting line. #
    # Use ax.annotate() to label lambda values.               #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ──────────────────────────────────────────────────────────────────────────────
# Main — provided, do not modify
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    OUT_DIR = os.path.dirname(__file__)

    print("=" * 65)
    print("Exercise 1: Map the Size-Accuracy Pareto Frontier")
    print("=" * 65)

    # Lambda sweep: log-uniform from 1e-3 to ~0.3
    # Paper samples gamma log-uniformly from [10^-3, 10^-0.5]
    LAM_VALUES = list(np.logspace(-3, -0.5, num=7))
    STEPS = int(os.environ.get("PARETO_STEPS", "2000"))

    print(f"\n[1] Sweeping λ over {len(LAM_VALUES)} values: "
          f"{[f'{v:.4f}' for v in LAM_VALUES]}")
    print(f"    Steps per run: {STEPS}")
    print(f"    (Set PARETO_STEPS=5000 for smoother results)")

    results = sweep_lambda(LAM_VALUES, steps_per_run=STEPS)

    assert isinstance(results, list), "sweep_lambda must return a list"
    assert len(results) == len(LAM_VALUES), "One result per lambda value"
    assert all('lambda' in r and 'accuracy' in r and 'model_bytes' in r
               for r in results), "Each result needs lambda, accuracy, model_bytes"

    print("\n[2] Results table:")
    print(f"  {'lambda':>8} | {'accuracy (%)':>12} | {'model_bytes':>11} | pareto_optimal")
    print("  " + "-" * 58)

    pareto = find_pareto_optimal(results)
    pareto_set = {r['lambda'] for r in pareto}

    for r in results:
        flag = "✓" if r['lambda'] in pareto_set else ""
        print(f"  {r['lambda']:>8.4f} | {r['accuracy']:>12.2f} | "
              f"{r['model_bytes']:>11.1f} | {flag}")

    pareto_strs = [f"lambda={r['lambda']:.4f}" for r in pareto]
    print(f"\n[3] Pareto-optimal configurations: {pareto_strs}")
    assert len(pareto) > 0, "Should find at least one Pareto-optimal point"

    # Check that lambda=0.05 (or closest) achieves reasonable compression
    closest_idx = min(range(len(LAM_VALUES)), key=lambda i: abs(LAM_VALUES[i] - 0.05))
    ref = results[closest_idx]
    print(f"\n[4] Reference λ≈0.05 result:")
    print(f"    λ={ref['lambda']:.4f}: accuracy={ref['accuracy']:.2f}%, "
          f"model_bytes={ref['model_bytes']:.1f}")

    print("\n[5] Plotting Pareto frontier...")
    plot_path = os.path.join(OUT_DIR, "pareto_frontier.png")
    plot_pareto_frontier(results, pareto, save_path=plot_path)
    print(f"    Saved: {plot_path}")

    # Summary
    best_acc  = max(r['accuracy']    for r in results)
    best_size = min(r['model_bytes'] for r in results)
    float32_bytes = 87860 * 4
    print(f"\n[6] Summary:")
    print(f"    Best accuracy: {best_acc:.2f}%  (λ={min(LAM_VALUES):.4f})")
    print(f"    Smallest model: {best_size:.1f} bytes  (λ={max(LAM_VALUES):.4f})")
    print(f"    Max compression vs float32: {float32_bytes/best_size:.1f}x")
    print(f"\n✓ Pareto frontier mapped! ({len(pareto)} Pareto-optimal configurations)")

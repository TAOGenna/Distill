"""
Exercise 2: Train the Self-Compressing CNN — SOLUTION
======================================================
Module 3 — Self-Compression Training on MNIST

Reference result (from the paper's notebook):
    loss:   0.14  bytes: 18075.4  acc: 98.20%
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────────────────────────────────────
# QConv2d + compression loss (provided)
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


def compute_compression_term(model: nn.Module) -> torch.Tensor:
    total_bits = torch.tensor(0.0)
    weight_count = 0
    for layer in model.modules():
        if isinstance(layer, QConv2d):
            total_bits = total_bits + layer.qbits()
            weight_count += layer.weight.numel()
    return total_bits / weight_count


def self_compression_loss(model, logits, targets, lam):
    task_loss = F.cross_entropy(logits, targets)
    Q = compute_compression_term(model)
    total_loss = task_loss + lam * Q
    return total_loss, task_loss.item(), Q.item()


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_mnist_loaders(batch_size: int = 512, data_dir: str = "/tmp/mnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=2000,       shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation and plotting
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_test_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate test accuracy.

    Note: We keep the model in train mode because BatchNorm uses
    track_running_stats=False (matching the tinygrad reference). In eval mode
    with no running stats, BatchNorm falls back to batch stats anyway, but
    staying in train mode is cleaner and consistent with the reference.
    """
    was_training = model.training
    model.train()   # keep BN in batch-stats mode
    correct = total = 0
    for images, labels in test_loader:
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    if not was_training:
        model.eval()
    return 100.0 * correct / total


def plot_training_dynamics(
    test_accs: list[float],
    bytes_used: list[float],
    save_path: str = "training_dynamics.png",
) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(bytes_used, color="red", alpha=0.8, label="Model bytes")
    ax1.set_ylabel("Model Size (bytes)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xlabel("Training step")

    ax2 = ax1.twinx()
    ax2.plot(test_accs, color="blue", alpha=0.8, label="Test accuracy")
    ax2.set_ylabel("Test Accuracy (%)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(80, 100)

    plt.title("Self-Compressing CNN: Accuracy vs Model Size over Training")
    fig.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# SOLUTION implementations
# ──────────────────────────────────────────────────────────────────────────────

class SelfCompressingCNN(nn.Module):
    """Five-layer QConv2d CNN for MNIST self-compression.

    Input shape:  (B, 1, 28, 28)
    Output shape: (B, 10) — logits

    Shape trace (no padding on any layer):
      conv1 k=5: (B,32,24,24)
      conv2 k=5: (B,32,20,20) → BN → MaxPool(2): (B,32,10,10)
      conv3 k=3: (B,64,8,8)
      conv4 k=3: (B,64,6,6)  → BN → MaxPool(2): (B,64,3,3)
      flatten+reshape: (B,576,1,1)
      conv5 k=1: (B,10,1,1) → flatten: (B,10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = QConv2d(1,   32, 5)
        self.conv2 = QConv2d(32,  32, 5)
        # track_running_stats=False matches the reference tinygrad implementation:
        # nn.BatchNorm(32, affine=False, track_running_stats=False)
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


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    lam: float,
) -> tuple[float, float]:
    """Single training step: forward, loss, backward, update.

    Returns
    -------
    tuple[float, float]
        (total_loss_value, Q_value)
    """
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss, _, Q_val = self_compression_loss(model, logits, labels, lam)
    loss.backward()
    optimizer.step()
    return loss.item(), Q_val


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    steps: int = 20_000,
    lam: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Full self-compression training loop.

    Returns
    -------
    tuple[list[float], list[float]]
        (test_accs, bytes_used) — one entry per step
    """
    optimizer = torch.optim.Adam(model.parameters())

    weight_count = sum(
        l.weight.numel() for l in model.modules()
        if isinstance(l, QConv2d)
    )

    test_accs: list[float] = []
    bytes_used: list[float] = []
    test_acc = float('nan')

    train_iter = iter(train_loader)

    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        loss_val, Q_val = train_step(model, optimizer, images, labels, lam)
        model_bytes = Q_val / 8 * weight_count

        eval_interval = max(1, min(10, steps // 5))   # at most 5 evals total
        if step % eval_interval == (eval_interval - 1):
            test_acc = get_test_accuracy(model, test_loader)

        test_accs.append(test_acc)
        bytes_used.append(model_bytes)

        pbar.set_description(
            f"loss: {loss_val:.4f}  bytes: {model_bytes:.0f}  accuracy: {test_acc:.2f}%"
        )

    return test_accs, bytes_used


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 65)
    print("Exercise 2: Train the Self-Compressing CNN")
    print("=" * 65)

    # Architecture sanity check
    print("\n[1] Architecture shape check:")
    dummy_model = SelfCompressingCNN()
    dummy_input = torch.randn(4, 1, 28, 28)
    dummy_out = dummy_model(dummy_input)
    assert dummy_out.shape == (4, 10), \
        f"Expected output shape (4, 10), got {dummy_out.shape}"
    print(f"    Input:  {dummy_input.shape}")
    print(f"    Output: {dummy_out.shape}  ✓")

    weight_count = sum(
        l.weight.numel() for l in dummy_model.modules() if isinstance(l, QConv2d)
    )
    print(f"    Weight parameters: {weight_count:,}")
    print(f"    At 32-bit floats: {weight_count * 4 / 1024:.1f} KB")

    # Load MNIST
    print("\n[2] Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=512)
    print(f"    Train batches: {len(train_loader)},  Test batches: {len(test_loader)}")

    # Determine training steps: default 25 steps for fast structural validation.
    # Set TRAIN_STEPS=20000 for paper-quality result (requires GPU or ~3hrs CPU).
    # Reference: 20k steps → 98.2% accuracy @ 18,075 bytes (~19x compression).
    TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "25"))
    print(f"\n[3] Training for {TRAIN_STEPS} steps (lambda=0.05)...")
    print(f"    (Reference: TRAIN_STEPS=20000 → 98.2% accuracy @ ~18 KB)")
    model = SelfCompressingCNN()
    test_accs, bytes_used = training_loop(
        model, train_loader, test_loader, steps=TRAIN_STEPS, lam=0.05
    )

    # Final evaluation
    final_acc = get_test_accuracy(model, test_loader)
    final_Q   = compute_compression_term(model).item()
    final_bytes = final_Q / 8 * weight_count
    compression_ratio = (weight_count * 4) / (final_bytes / 1)

    print(f"\n[4] Final results:")
    print(f"    Final — accuracy: {final_acc:.2f}%, model size: {final_bytes:.1f} bytes, "
          f"compression: {compression_ratio:.1f}x")

    # Structural validations (work at any training length):
    # 1. accuracy is a valid number (not NaN)
    # 2. model bytes < initial bytes (compression pressure is working)
    # 3. Q value is differentiable (gradient flows)
    init_bytes = 2.0 / 8 * weight_count  # Q=2.0 at init (b=2.0 everywhere)
    assert not (final_acc != final_acc), "accuracy is NaN — check forward pass"
    assert final_bytes <= init_bytes + 1.0, \
        f"Model bytes {final_bytes:.0f} should not exceed init {init_bytes:.0f}"
    print(f"    ✓ compression is active: {final_bytes:.0f} bytes "
          f"(init: {init_bytes:.0f} bytes)")

    # Save model for exercise 3
    model_path = os.path.join(os.path.dirname(__file__), "trained_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"    Saved model: {model_path}")

    # Plot
    plot_path = os.path.join(os.path.dirname(__file__), "training_dynamics.png")
    plot_training_dynamics(test_accs, bytes_used, save_path=plot_path)
    print(f"\n✓ Training complete!")

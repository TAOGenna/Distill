"""
Exercise 2: Train the Self-Compressing CNN
==========================================
Module 3 — Self-Compression Training on MNIST

Implement the full self-compression training loop that jointly optimizes:
  - Test accuracy on MNIST (via cross-entropy task loss)
  - Model compression (via the lambda * Q compression penalty)

Reference result (from the paper's notebook):
  loss:   0.14  bytes: 18075.4  acc: 98.20%

After 20,000 steps you should see:
  - Test accuracy >= 97.5%
  - Model size <= 25,000 bytes
  - A dual-axis plot showing accuracy rising and bytes decreasing over time

Key implementation details:
  - Architecture: 5 QConv2d layers (1->32->32->64->64->10)
  - NO padding on any conv layer (gives 3x3 spatial after two pools)
  - Reshape: x.flatten(1).reshape(B, 576, 1, 1) before the final 1x1 conv
  - Batch size: 512  |  Optimizer: Adam (default lr=1e-3)
  - lambda = 0.05  |  Steps: 20,000
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
# QConv2d + compression loss (provided — matches Exercise 1)
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
# PROVIDED: Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_mnist_loaders(batch_size: int = 512, data_dir: str = "/tmp/mnist"):
    """Download MNIST and return (train_loader, test_loader).

    Parameters
    ----------
    batch_size : int
        Training batch size. Paper uses 512.
    data_dir : str
        Where to cache the dataset.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, test_loader)
    """
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
# PROVIDED: Evaluation and plotting
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_test_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate test accuracy on the full MNIST test set.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader

    Returns
    -------
    float
        Accuracy in percent (0-100).

    Note: Model stays in train() mode since BatchNorm uses track_running_stats=False,
    matching the tinygrad reference (batch stats are always used).
    """
    model.train()   # BatchNorm uses batch stats regardless (track_running_stats=False)
    correct = total = 0
    for images, labels in test_loader:
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def plot_training_dynamics(
    test_accs: list[float],
    bytes_used: list[float],
    save_path: str = "training_dynamics.png",
) -> None:
    """Dual-axis plot: model size (bytes) and test accuracy vs training steps.

    Parameters
    ----------
    test_accs : list[float]
        Test accuracy per step (NaN for steps without evaluation).
    bytes_used : list[float]
        Model bytes per step.
    save_path : str
        Where to save the PNG.
    """
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
# YOUR CODE — implement the three components below
# ──────────────────────────────────────────────────────────────────────────────

class SelfCompressingCNN(nn.Module):
    """Five-layer CNN for MNIST using QConv2d layers throughout.

    Architecture (NO padding on any layer — spatial dims reduce):
      Input:        (B, 1, 28, 28)
      conv1 (k=5):  (B, 32, 24, 24)  <- 28-5+1=24
      conv2 (k=5):  (B, 32, 20, 20)  <- 24-5+1=20
      BN + MaxPool: (B, 32, 10, 10)  <- 20//2=10
      conv3 (k=3):  (B, 64,  8,  8)  <- 10-3+1=8
      conv4 (k=3):  (B, 64,  6,  6)  <- 8-3+1=6
      BN + MaxPool: (B, 64,  3,  3)  <- 6//2=3
      Flatten+Reshape: (B, 576, 1, 1)   <- 64*3*3=576
      conv5 (k=1):  (B, 10,  1,  1)
      Flatten:      (B, 10)

    Total weight parameters:
      32*1*5*5 + 32*32*5*5 + 64*32*3*3 + 64*64*3*3 + 10*576*1*1
      = 800 + 25600 + 18432 + 36864 + 5760 = 87456

    BatchNorm is used WITHOUT learnable affine params (affine=False) to
    avoid interference with the quantized weights.
    """

    def __init__(self):
        super().__init__()
        ###########################################################################
        # YOUR CODE HERE — 10-15 lines                                            #
        #                                                                         #
        # Define these attributes (in order):                                     #
        #   self.conv1 = QConv2d(1,   32, 5)    # no padding                    #
        #   self.conv2 = QConv2d(32,  32, 5)                                     #
        #   self.bn1   = nn.BatchNorm2d(32, affine=False,                        #
        #                               track_running_stats=False)               #
        #   self.conv3 = QConv2d(32,  64, 3)                                     #
        #   self.conv4 = QConv2d(64,  64, 3)                                     #
        #   self.bn2   = nn.BatchNorm2d(64, affine=False,                        #
        #                               track_running_stats=False)               #
        #   self.conv5 = QConv2d(576, 10, 1)    # 1x1 conv as linear layer      #
        #                                                                         #
        # NOTE: track_running_stats=False matches the tinygrad reference where   #
        # BatchNorm always uses batch statistics (no EMA running mean/var).      #
        ###########################################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the self-compressing CNN.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, 28, 28). MNIST images (normalized).

        Returns
        -------
        torch.Tensor
            Shape (B, 10). Logits (unnormalized class scores).

        Notes
        -----
        Follow this exact sequence:
          1. relu(conv1(x))
          2. max_pool2d(bn1(relu(conv2(...))), kernel_size=2)
          3. relu(conv3(...))
          4. max_pool2d(bn2(relu(conv4(...))), kernel_size=2)
          5. x.flatten(1).reshape(B, 576, 1, 1)
          6. conv5(x).flatten(1)
        """
        ###########################################################################
        # YOUR CODE HERE — 8-10 lines                                             #
        #                                                                         #
        # Hint: The reshape step is:                                              #
        #   x = x.flatten(1).reshape(x.shape[0], 576, 1, 1)                     #
        # After conv4+bn2+pool, x has shape (B, 64, 3, 3).                      #
        # flatten(1) gives (B, 576), then reshape gives (B, 576, 1, 1).         #
        ###########################################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################################


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    lam: float,
) -> tuple[float, float]:
    """Single training step: forward, loss, backward, update.

    Parameters
    ----------
    model : nn.Module
        SelfCompressingCNN (or any nn.Module with QConv2d layers).
    optimizer : torch.optim.Optimizer
        Adam optimizer (or any optimizer).
    images : torch.Tensor
        Shape (B, 1, 28, 28). Training batch.
    labels : torch.Tensor
        Shape (B,). Integer class labels.
    lam : float
        Compression strength. Reference: 0.05.

    Returns
    -------
    tuple[float, float]
        (total_loss_value, Q_value)
        Both are plain floats (use .item()).

    Notes
    -----
    - Call model.train() at the start.
    - Zero gradients BEFORE forward pass (optimizer.zero_grad()).
    - Return loss.item() and Q_val (already float from self_compression_loss).
    """
    ###########################################################################
    # YOUR CODE HERE — 8-12 lines                                             #
    #                                                                         #
    # 1. model.train()                                                        #
    # 2. optimizer.zero_grad()                                                #
    # 3. logits = model(images)                                               #
    # 4. loss, _, Q_val = self_compression_loss(model, logits, labels, lam)  #
    # 5. loss.backward()                                                      #
    # 6. optimizer.step()                                                     #
    # 7. return loss.item(), Q_val                                            #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    steps: int = 20_000,
    lam: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Full training loop for self-compression training.

    Parameters
    ----------
    model : nn.Module
        SelfCompressingCNN instance.
    train_loader : DataLoader
        Infinite-ish training data (will cycle through as needed).
    test_loader : DataLoader
        MNIST test set for periodic evaluation.
    steps : int
        Total training steps. Reference: 20,000.
    lam : float
        Compression hyperparameter. Reference: 0.05.

    Returns
    -------
    tuple[list[float], list[float]]
        (test_accs, bytes_used)
        test_accs : float per step (NaN except every 10 steps)
        bytes_used : bytes per step

    Notes
    -----
    - Create Adam optimizer: torch.optim.Adam(model.parameters())
    - Compute weight_count once: sum(layer.weight.numel() for QConv2d layers)
    - model_bytes = Q / 8 * weight_count  (from Q, the avg bits/weight)
    - Evaluate accuracy every 10 steps (when step % 10 == 9)
    - Use tqdm for the progress bar with step description showing:
        f"loss: {loss:.4f}  bytes: {model_bytes:.0f}  acc: {test_acc:.2f}%"
    - Cycle through train_loader: use iter() + next(), reset iter on StopIteration
    """
    ###########################################################################
    # YOUR CODE HERE — 20-30 lines                                            #
    #                                                                         #
    # Skeleton:                                                                #
    #   optimizer = torch.optim.Adam(model.parameters())                     #
    #   weight_count = sum(l.weight.numel() for l in model.modules()         #
    #                      if isinstance(l, QConv2d))                        #
    #   test_accs, bytes_used = [], []                                        #
    #   test_acc = float('nan')                                               #
    #   train_iter = iter(train_loader)                                       #
    #   for step in tqdm(range(steps)):                                       #
    #       try: images, labels = next(train_iter)                            #
    #       except StopIteration: train_iter = ...; images, labels = ...      #
    #       loss_val, Q_val = train_step(model, optimizer, images, labels, l) #
    #       model_bytes = Q_val / 8 * weight_count                           #
    #       if step % 10 == 9: test_acc = get_test_accuracy(model, test_ldr)  #
    #       test_accs.append(test_acc)                                        #
    #       bytes_used.append(model_bytes)                                    #
    #   return test_accs, bytes_used                                          #
    ###########################################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################################


# ──────────────────────────────────────────────────────────────────────────────
# Main — DO NOT MODIFY
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

    # Train — default 25 steps for fast structural check (set TRAIN_STEPS=20000 for
    # the reference result: 98.2% accuracy @ ~18 KB on GPU)
    TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "25"))
    print(f"\n[3] Training for {TRAIN_STEPS} steps (lambda=0.05)...")
    print(f"    (TRAIN_STEPS=20000 → 98.2% accuracy @ ~18 KB per reference)")
    model = SelfCompressingCNN()
    test_accs, bytes_used = training_loop(
        model, train_loader, test_loader, steps=TRAIN_STEPS, lam=0.05
    )

    # Final evaluation
    final_acc = get_test_accuracy(model, test_loader)
    final_Q   = compute_compression_term(model).item()
    final_bytes = final_Q / 8 * weight_count
    compression_ratio = (weight_count * 4) / (final_bytes / 1)   # vs float32

    print(f"\n[4] Final results:")
    print(f"    Final — accuracy: {final_acc:.2f}%, model size: {final_bytes:.1f} bytes, "
          f"compression: {compression_ratio:.1f}x")

    # Structural validation: verify compression is active and code is correct
    init_bytes = 2.0 / 8 * weight_count   # Q=2.0 bits/weight at initialization
    assert not (final_acc != final_acc), "accuracy is NaN — check forward pass"
    assert final_bytes <= init_bytes + 1.0, \
        "Model bytes unexpectedly grew during training — check compression loss"
    print(f"    ✓ compression active: {final_bytes:.0f} bytes (init: {init_bytes:.0f})")
    print(f"    ✓ accuracy in output (run TRAIN_STEPS=20000 for target >= 97.5%)")

    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), "training_dynamics.png")
    plot_training_dynamics(test_accs, bytes_used, save_path=plot_path)
    print(f"\n✓ Training complete!")

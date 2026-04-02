"""
Exercise 3: Naive Rounding vs. Proper Quantization
====================================================
Module 0: Quantization Fundamentals

Compare two approaches to weight quantization:
  1. NAIVE: round weights to 2^b uniform levels spanning [min, max]
  2. PROPER: use the exponent-based quantize/dequantize from Exercises 1-2

At low bit-widths (1-4 bits), naive rounding fails badly when the weight
distribution is bell-shaped (Gaussian) — the uniform grid spanning [min, max]
wastes most of its precision on the tails, where there are very few weights.

Your tasks:
  1. naive_quantize(weights, num_bits) — quantize weights using a simple
     uniform grid over [min(weights), max(weights)].
  2. evaluate_both(weights, bit_widths) — for each bit-width, compare
     MSE of both approaches.

After completing this exercise, you'll understand WHY the paper uses a
learnable exponent e rather than a simple round-to-nearest approach.
"""

import torch
import torch.nn as nn
import math
import numpy as np


# ---------------------------------------------------------------------------
# Provided: proper quantize/dequantize from Exercises 1-2
# ---------------------------------------------------------------------------

def quantize(x: torch.Tensor, num_bits: int, exponent: float) -> torch.Tensor:
    """Proper quantization: scale → clamp → round. From Exercise 1."""
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1
    return (x * (2 ** (-exponent))).clamp(q_min, q_max).round()


def dequantize(qx: torch.Tensor, exponent: float) -> torch.Tensor:
    """Dequantize integers back to float scale."""
    return qx * (2 ** exponent)


def find_optimal_exponent(weights: torch.Tensor, num_bits: int) -> float:
    """Grid search for MSE-minimizing exponent. From Exercise 2."""
    best_e, best_mse = -12.0, float("inf")
    for e in np.arange(-12.0, 4.0, 0.25):
        q = quantize(weights, num_bits, float(e))
        rec = dequantize(q, float(e))
        mse = ((weights - rec) ** 2).mean().item()
        if mse < best_mse:
            best_mse, best_e = mse, float(e)
    return best_e


def make_gaussian_weights(n_weights: int = 800, std: float = 0.08,
                          seed: int = 42) -> torch.Tensor:
    """
    Generate Gaussian-distributed weights with heavy-tail outliers.

    Trained network weights follow approximately Gaussian distributions
    concentrated near zero — very different from uniform init.
    The ~2% outliers create a large [min, max] range that stresses naive grids.
    """
    torch.manual_seed(seed)
    weights = torch.randn(n_weights) * std
    n_outliers = int(0.02 * n_weights)
    outlier_idx = torch.randperm(n_weights)[:n_outliers]
    weights[outlier_idx] *= 5.0
    return weights


# ---------------------------------------------------------------------------
# Provided: a pretrained 2-layer MLP on a synthetic regression task
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """A 2-layer MLP for tabular regression. Trained to fit a sinusoidal target."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64,
                 output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def train_simple_mlp(seed: int = 0) -> tuple:
    """Train a small MLP on a synthetic task, return (model, X_test, y_test)."""
    torch.manual_seed(seed)
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_train = torch.randn(2000, 8)
    y_train = torch.sin(X_train.sum(dim=1, keepdim=True))
    for step in range(500):
        idx = torch.randperm(len(X_train))[:64]
        X_batch, y_batch = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    X_test = torch.randn(500, 8)
    y_test = torch.sin(X_test.sum(dim=1, keepdim=True))
    return model, X_test, y_test


def evaluate_model(model: SimpleMLP, X_test: torch.Tensor,
                   y_test: torch.Tensor) -> float:
    """Evaluate model test MSE."""
    with torch.no_grad():
        preds = model(X_test)
    return ((preds - y_test) ** 2).mean().item()


def apply_quantization_to_model(original_model: SimpleMLP, quantize_fn) -> SimpleMLP:
    """Create a copy of the model with all linear weights quantized."""
    import copy
    q_model = copy.deepcopy(original_model)
    with torch.no_grad():
        for module in q_model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = quantize_fn(module.weight.data.flatten()).reshape(
                    module.weight.shape)
    return q_model


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION
# ---------------------------------------------------------------------------

def naive_quantize(weights: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Quantize weights using a naive uniform grid over [min(w), max(w)].

    This approach:
      1. Finds the range [w_min, w_max] of the weight tensor
      2. Creates 2^num_bits uniformly-spaced levels in that range
      3. Maps each weight to its nearest level

    The key flaw: on bell-shaped (Gaussian) distributions, a few large
    outliers force a large [min, max] span, while most weights cluster
    near zero. The grid wastes 3/4 of its buckets on the sparse tails.

    Parameters
    ----------
    weights : torch.Tensor
        Weight tensor to quantize (any shape).
    num_bits : int
        Number of bits b, giving 2^b quantization levels.

    Returns
    -------
    torch.Tensor
        Quantized weight tensor, same shape as input. Values lie on the
        uniform grid {w_min, w_min + step, ..., w_max}.

    Notes
    -----
    - num_levels = 2 ** num_bits
    - step = (w_max - w_min) / (num_levels - 1)
    - Handle degenerate case: if w_min == w_max, return weights.clone()
    - Map to nearest level: (weights - w_min) / step → round → clamp → rescale
    - Do NOT use loops — operate on the whole tensor at once
    """
    ###########################################################
    # YOUR CODE HERE — 8-12 lines                             #
    #                                                         #
    # 1. w_min = weights.min(), w_max = weights.max()         #
    # 2. num_levels = 2 ** num_bits                           #
    # 3. If w_min == w_max: return weights.clone()            #
    # 4. step = (w_max - w_min) / (num_levels - 1)           #
    # 5. normalized = (weights - w_min) / step                #
    # 6. indices = normalized.round().clamp(0, num_levels - 1)#
    # 7. return w_min + indices * step                        #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def evaluate_both(
    weights: torch.Tensor,
    bit_widths: list,
) -> dict:
    """
    Compare naive vs proper quantization MSE for each bit-width.

    For each bit-width, compute:
      - naive_mse: apply naive_quantize, measure reconstruction MSE
      - proper_mse: find optimal exponent, apply proper quantize/dequantize,
        measure reconstruction MSE
      - mse_ratio: naive_mse / proper_mse (how much worse naive is)

    Parameters
    ----------
    weights : torch.Tensor
        Weight tensor to analyze (1D or multi-dimensional — will be flattened).
    bit_widths : list of int
        Bit-widths to evaluate (e.g., [1, 2, 3, 4, 8]).

    Returns
    -------
    dict
        results[num_bits] = {
            "naive_mse": float,
            "proper_mse": float,
            "mse_ratio": float,    # naive_mse / proper_mse
        }

    Notes
    -----
    - Flatten weights first: w_flat = weights.flatten()
    - For proper: call find_optimal_exponent(w_flat, b) then quantize/dequantize
    - For ratio: if proper_mse < 1e-12, set mse_ratio = float('inf')
    """
    ###########################################################
    # YOUR CODE HERE — 15-20 lines                            #
    #                                                         #
    # w_flat = weights.flatten()                              #
    # results = {}                                            #
    # For each b in bit_widths:                               #
    #                                                         #
    # --- Naive ---                                           #
    #   w_naive = naive_quantize(w_flat, b)                   #
    #   naive_mse = mean((w_flat - w_naive)^2)               #
    #                                                         #
    # --- Proper ---                                          #
    #   e_opt = find_optimal_exponent(w_flat, b)              #
    #   q = quantize(w_flat, b, e_opt)                        #
    #   w_proper = dequantize(q, e_opt)                       #
    #   proper_mse = mean((w_flat - w_proper)^2)             #
    #                                                         #
    # --- Ratio ---                                           #
    #   ratio = naive_mse / proper_mse (or inf if tiny denom)#
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Main harness (provided — do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 76)
    print("Exercise 3: Naive Rounding vs. Proper Quantization")
    print("=" * 76)

    # --- Part A: Gaussian weights (bell-shaped, with outliers) ---
    print("\n--- Part A: Gaussian weights (typical of trained networks) ---")
    print("NOTE: Kaiming UNIFORM is flat — both methods tie. Gaussian reveals")
    print("the advantage of proper quantization (sparse-tail problem).")
    w_gaussian = make_gaussian_weights(n_weights=800, std=0.08, seed=42)
    print(f"Weight stats: std={w_gaussian.std():.4f}, "
          f"range=[{w_gaussian.min():.4f}, {w_gaussian.max():.4f}]")
    print(f"(Note: ~2% outliers make max_abs >> 3*std, stressing naive grid)")

    bit_widths = [1, 2, 3, 4, 8]
    results = evaluate_both(w_gaussian, bit_widths)

    print(f"\n{'proper_quantization vs naive_quantize — MSE comparison':^76}")
    print(f"{'bits':>6} | {'naive_MSE':>12} | {'proper_MSE':>12} | "
          f"{'ratio (naive/proper)':>22} | {'winner':>8}")
    print("-" * 70)
    for b in bit_widths:
        r = results[b]
        nmse = r["naive_mse"]
        pmse = r["proper_mse"]
        ratio = r["mse_ratio"]
        ratio_str = f"{ratio:.1f}x" if ratio != float("inf") else "   inf"
        winner = "proper" if pmse < nmse else "naive "
        print(f"{b:>6} | {nmse:>12.6f} | {pmse:>12.6f} | "
              f"{ratio_str:>22} | {winner:>8}")

    # --- Part B: MLP regression task comparison ---
    print("\n--- Part B: MLP regression task — impact on model accuracy ---")
    print("Training a 2-layer MLP on a synthetic task...")
    model, X_test, y_test = train_simple_mlp(seed=42)
    baseline_mse = evaluate_model(model, X_test, y_test)
    print(f"Baseline (float32) test MSE: {baseline_mse:.6f}")

    all_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.flatten())
    trained_weights = torch.cat(all_weights)
    print(f"Trained weight stats: std={trained_weights.std():.4f}, "
          f"range=[{trained_weights.min():.4f}, {trained_weights.max():.4f}]")

    print(f"\n{'bits':>6} | {'naive_test_MSE':>16} | {'proper_test_MSE':>17} | "
          f"{'naive_degr':>12} | {'proper_degr':>12}")
    print("-" * 70)

    for b in [2, 3, 4, 8]:
        def make_naive_fn(bits):
            return lambda w: naive_quantize(w, bits)

        def make_proper_fn(bits):
            def proper_fn(w):
                e_opt = find_optimal_exponent(w, bits)
                return dequantize(quantize(w, bits, e_opt), e_opt)
            return proper_fn

        naive_model = apply_quantization_to_model(model, make_naive_fn(b))
        proper_model = apply_quantization_to_model(model, make_proper_fn(b))

        naive_mse = evaluate_model(naive_model, X_test, y_test)
        proper_mse = evaluate_model(proper_model, X_test, y_test)

        naive_deg = naive_mse / baseline_mse
        proper_deg = proper_mse / baseline_mse

        print(f"{b:>6} | {naive_mse:>16.6f} | {proper_mse:>17.6f} | "
              f"{naive_deg:>10.1f}x   | {proper_deg:>10.1f}x")

    # --- Part C: Grid mismatch visualization ---
    print("\n--- Part C: Why naive fails on bell-shaped distributions ---")
    print(f"\nFor b=2 (4 levels) on Gaussian weights:")
    b_demo = 2

    w_min, w_max = w_gaussian.min().item(), w_gaussian.max().item()
    num_levels = 2 ** b_demo
    step_naive = (w_max - w_min) / (num_levels - 1)
    naive_levels = [w_min + i * step_naive for i in range(num_levels)]
    print(f"  Naive grid: {[f'{v:.4f}' for v in naive_levels]}")
    print(f"  Step size:  {step_naive:.4f}  (spans full range incl. outliers)")

    e_opt = find_optimal_exponent(w_gaussian, b_demo)
    step_proper = 2 ** e_opt
    q_min_p = -(2 ** (b_demo - 1))
    proper_levels = [(q_min_p + i) * step_proper for i in range(num_levels)]
    print(f"  Proper grid (e={e_opt:.2f}): {[f'{v:.4f}' for v in proper_levels]}")
    print(f"  Step size:  {step_proper:.4f}  (tuned to center of distribution)")

    w_naive_flat = naive_quantize(w_gaussian, b_demo)
    print(f"\n  Naive bucket utilization (800 weights total):")
    for level in naive_levels:
        count = ((w_naive_flat - level).abs() < 1e-6).sum().item()
        pct = count / len(w_gaussian) * 100
        bar = "#" * int(pct / 2)
        print(f"    {level:7.4f}: {count:4d} weights ({pct:5.1f}%)  {bar}")
    print("  → Most weights pile into ONE bucket! Other 3 buckets wasted.")

    q_proper = quantize(w_gaussian, b_demo, e_opt)
    w_proper_flat = dequantize(q_proper, e_opt)
    print(f"\n  Proper bucket utilization:")
    for level in proper_levels:
        count = ((w_proper_flat - level).abs() < 1e-6).sum().item()
        pct = count / len(w_gaussian) * 100
        bar = "#" * int(pct / 2)
        print(f"    {level:7.4f}: {count:4d} weights ({pct:5.1f}%)  {bar}")
    print("  → Weights spread across ALL buckets — much better utilization!")

    # --- Assertions ---
    print("\n--- Assertions ---")
    r2 = results[2]
    assert r2["naive_mse"] > r2["proper_mse"], (
        "At 2 bits on Gaussian weights, naive should be worse than proper!\n"
        f"naive_mse={r2['naive_mse']:.6f}, proper_mse={r2['proper_mse']:.6f}"
    )
    assert r2["mse_ratio"] > 1.5, (
        f"Expected naive/proper ratio > 1.5 at 2 bits, got {r2['mse_ratio']:.2f}"
    )
    r8 = results[8]
    assert r8["proper_mse"] < 1e-4, \
        f"8-bit proper MSE={r8['proper_mse']:.2e} should be near-zero!"

    print("proper_quantization outperforms naive at low bit-widths: ✓")
    print(f"2-bit MSE ratio (naive/proper) on Gaussian weights: {r2['mse_ratio']:.1f}x")

    print("\n" + "=" * 76)
    print("All assertions passed!")
    print("\nKey takeaway: naive rounding spans the full [min, max] range.")
    print("On bell-shaped (Gaussian) distributions, outliers force a large")
    print("step size that wastes precision on the central mass of weights.")
    print("The exponent e in proper quantization adapts the grid center")
    print("and step to match the actual distribution — much more efficient.")
    print("This is why the paper makes e a LEARNABLE parameter.")
    print("=" * 76)

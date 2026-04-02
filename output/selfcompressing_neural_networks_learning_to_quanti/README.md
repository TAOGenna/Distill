# Self-Compressing Neural Networks: Learning to Quantize from Scratch

A hands-on course that reproduces the [Self-Compressing Neural Networks](https://arxiv.org/abs/2301.13142) paper from scratch using PyTorch.

## Overview

Self-compression is a strikingly elegant idea: what if a neural network could learn *how much precision each of its own weights needs* during training? By making the quantization bit-width a differentiable, per-channel parameter, the network jointly optimizes task performance and its own compression. Channels that don't contribute get their bit-widths pushed to zero — effectively pruning themselves. The result: a network that achieves ~98% accuracy on MNIST while using only ~18KB of storage (down from ~342KB at 32-bit floats), a ~20x compression.

This course builds the entire self-compression pipeline from scratch across 5 progressive modules.

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **PyTorch** >= 2.0.0 (with torchvision for MNIST)
- **NumPy** >= 1.24.0
- **Matplotlib** >= 3.7.0
- **tqdm** >= 4.65.0

> **Note:** The [reference implementation](https://github.com/geohot/ai-notebooks/blob/master/mnist_self_compression.ipynb) uses tinygrad. This course uses PyTorch for wider accessibility, but the concepts transfer directly. If you want to try with tinygrad or JAX after completing the course, the translation is straightforward — the core patterns (STE, per-channel quantization, compression loss) are framework-agnostic.

## Learning Path

```
Module 0: Quantization Fundamentals
    │
    ▼
Module 1: Straight-Through Estimator
    │
    ▼
Module 2: Building QConv2d  ◄── depends on Modules 0 & 1
    │
    ▼
Module 3: Self-Compression Training on MNIST
    │
    ▼
Module 4: Pareto Frontier & Emergent Pruning
```

### Module 0 — Quantization Fundamentals: From Floats to Fixed-Point
Build intuition for how quantization maps continuous weights to discrete integers. Implement quantize/dequantize, explore the bit-width vs. precision tradeoff, and see why the exponent parameter is critical.

**Exercises:** 3 | **Scaffolding:** Heavy

### Module 1 — The Straight-Through Estimator: Gradients Through Rounding
The mathematical trick that makes self-compression possible. See rounding kill gradients, then rescue them with the STE. Implement and verify the `(qw.round() - qw).detach() + qw` pattern.

**Exercises:** 3 | **Scaffolding:** Heavy → Medium

### Module 2 — Building the QConv2d Layer: Learnable Quantization
The core implementation: a convolutional layer with per-channel learnable bit-width (b) and exponent (e) parameters. Build `qweight()`, `forward()`, and `qbits()` piece by piece.

**Exercises:** 4 | **Scaffolding:** Medium → Light

### Module 3 — Self-Compression Training on MNIST
Wire everything together: build the full 5-layer CNN, implement the compression loss L = L_task + λ·Q, train for 20K steps, and reproduce ~98% accuracy at ~18KB.

**Exercises:** 3 | **Scaffolding:** Medium → Light

### Module 4 — The Pareto Frontier & Emergent Pruning
Explore the compression-accuracy tradeoff by varying λ. Compare self-compression against uniform post-training quantization. Analyze which channels the network learns to prune and why.

**Exercises:** 3 | **Scaffolding:** Light

## What You'll Build

By the end of this course, you will have:
- A complete `QConv2d` layer with learnable quantization
- A straight-through estimator implementation verified with gradient checks
- A self-compressing CNN that learns to prune and quantize itself
- Pareto frontier analysis showing the size-accuracy tradeoff
- Comparison between self-compression and naive uniform quantization

## What's Next

After completing this course, consider exploring:
- **Uniform vs. Non-Uniform Quantization**: How learned mixed-precision compares to fixed precision across a network
- **Post-Training Quantization (PTQ)**: Methods like GPTQ and AWQ that quantize pre-trained models without retraining
- **Knowledge Distillation**: Combining self-compression with distillation from a larger teacher model
- **Structured Pruning**: Dedicated pruning methods that remove channels/attention heads based on importance scores
- **Quantization for LLMs**: Applying quantization-aware training to transformer architectures (QLoRA, GGML formats)
- **Hardware-Aware Quantization**: Targeting specific hardware constraints (INT4/INT8 on GPUs, binary networks on FPGAs)

---
_Generated from https://arxiv.org/abs/2301.13142 on 2026-04-02 by distill._

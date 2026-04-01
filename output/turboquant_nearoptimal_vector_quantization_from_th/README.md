# TurboQuant: Near-Optimal Vector Quantization from Theory to Practice

A hands-on course that walks you through reproducing the core algorithms and theoretical results of the TurboQuant paper (Zandieh et al., ICLR 2026). You will build a near-optimal vector quantizer from scratch, starting from the problem definition and ending with applications to nearest neighbor search and KV cache compression.

## What You'll Build

By the end of this course, you will have implemented:

- **Uniform and distribution-aware scalar quantizers** to build intuition for why codebook design matters
- **Random rotation preprocessing** via QR decomposition of Gaussian matrices
- **The Beta distribution of hypersphere coordinates** — verified empirically with KS tests
- **Lloyd-Max optimal codebooks** for the Beta distribution at bit-widths 1-4
- **The full TurboQuant_mse pipeline** (rotate → quantize → dequantize → rotate back)
- **QJL (Quantized Johnson-Lindenstrauss)** 1-bit inner product quantizer
- **The two-stage TurboQuant_prod** algorithm with unbiased inner product estimation
- **Shannon's distortion-rate lower bound** proving TurboQuant is within 2.7× of optimal
- **Nearest neighbor search** using quantized vectors with recall measurement
- **Simplified KV cache quantization** in a toy attention computation

## Setup

### Prerequisites

- Python 3.9+
- Familiarity with linear algebra (orthogonal matrices, inner products, norms)
- Basic probability (expected value, variance, distributions)
- Experience with NumPy for numerical computing
- Basic understanding of transformer architecture and attention mechanism

### Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy >= 1.24` — core numerical computing
- `scipy >= 1.10` — special functions (Gamma), numerical integration, statistical tests
- `matplotlib >= 3.7` — optional, for plotting distributions and distortion curves

### Running Exercises

Each exercise is a standalone Python file. Run it directly:

```bash
cd module_01_random_rotations/
python ex1_rotation_matrix.py
```

Exercises contain `# YOUR CODE HERE` blocks with line-count hints. Fill them in and run the file — the `__main__` block will test your implementation and print results.

Solutions are in the `_solutions/` directory of each module.

## Learning Path

```
Module 0: The Vector Quantization Problem
    │   (no prerequisites)
    │   Uniform quantization, MSE/IP distortion, motivation
    ▼
Module 1: Random Rotations & Hypersphere Geometry
    │   (depends on: Module 0)
    │   QR decomposition, Beta distribution, concentration
    ▼
Module 2: Optimal Scalar Quantization & TurboQuant_mse
    │   (depends on: Module 1)
    │   Lloyd-Max algorithm, codebooks, full MSE pipeline
    ▼
Module 3: Inner Product Quantization: QJL & TurboQuant_prod
    │   (depends on: Module 2)
    │   QJL transform, bias demonstration, two-stage algorithm
    ▼
Module 4: Information-Theoretic Lower Bounds & Applications
        (depends on: Modules 2, 3)
        Shannon bound, NN search, KV cache quantization
```

Modules are designed to be completed in order. Each module builds on code and concepts from previous modules.

## What's Next

After completing this course, explore these related topics:

- **Entropy Encoding of Codebook Indices** — lossless compression of quantization indices using the known probability distribution. Can reduce effective bit-width by ~5% at b=4.
- **PolarQuant** — an alternative approach that quantizes KV vectors in polar coordinates after random preconditioning. Shares the rotation insight with TurboQuant but differs in how coordinates are quantized.
- **RaBitQ** — a related grid-based quantization method for nearest neighbor search that also uses random projection/rotation.
- **Structured Random Rotations** — replacing dense random Π with random Hadamard transforms to achieve O(d log d) rotation cost instead of O(d²).
- **GPU-Optimized Implementation** — CUDA kernels for the rotation and quantization steps, enabling real-time KV cache quantization during LLM inference.
- **Non-Euclidean Quantization** — extending TurboQuant ideas to other metric spaces (cosine similarity, hyperbolic embeddings).

---
_Generated from [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874) on 2026-04-01 by distill._

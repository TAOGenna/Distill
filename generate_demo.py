#!/usr/bin/env python3
"""Generate a demo course from Andrew Chan's 'Fast LLM Inference From Scratch' blog post.

This script creates a complete, high-quality CS231n-style coursework without
requiring an API key. It demonstrates the kind of output Scaffoldly produces.

Usage:
    python generate_demo.py
"""

import json
from pathlib import Path

from scaffoldly.notebook import (
    _slugify,
    cells_to_notebook,
    create_course_readme_notebook,
    save_notebook,
)

OUTPUT_DIR = Path("output/fast_llm_inference_from_scratch")


# =============================================================================
# Course metadata
# =============================================================================

ANALYSIS = {
    "title": "Fast LLM Inference From Scratch",
    "summary": (
        "A deep dive into building a high-performance LLM inference engine from "
        "scratch in C/CUDA, progressing from a naive CPU implementation (~1 tok/s) "
        "to an optimized GPU implementation (~64 tok/s) through systematic optimization "
        "of matrix operations, memory access patterns, and kernel design."
    ),
    "domain": "systems programming / ML infrastructure",
    "overall_difficulty": "advanced",
    "key_concepts": [
        {"name": "Transformer architecture", "importance": "core", "difficulty": "intermediate"},
        {"name": "Matrix-vector multiplication", "importance": "core", "difficulty": "beginner"},
        {"name": "Memory bandwidth vs compute boundedness", "importance": "core", "difficulty": "intermediate"},
        {"name": "Roofline model", "importance": "core", "difficulty": "intermediate"},
        {"name": "SIMD / vectorization", "importance": "supporting", "difficulty": "advanced"},
        {"name": "GPU kernel programming", "importance": "core", "difficulty": "advanced"},
        {"name": "Warp-level parallelism", "importance": "core", "difficulty": "advanced"},
        {"name": "Memory coalescing", "importance": "core", "difficulty": "advanced"},
        {"name": "Weight quantization", "importance": "supporting", "difficulty": "intermediate"},
        {"name": "KV cache", "importance": "core", "difficulty": "intermediate"},
    ],
    "source_url": "https://andrewkchan.dev/posts/yalm.html",
    "source_author": "Andrew Chan",
}

CURRICULUM = {
    "course_title": "Fast LLM Inference From Scratch",
    "course_description": (
        "Build an LLM inference engine from scratch, starting with the math "
        "behind transformers and progressing through systematic optimization "
        "techniques. By the end, you'll understand exactly why LLMs are fast "
        "(or slow) and how to make them faster."
    ),
    "target_level": "Mid-level Python developer familiar with NumPy but new to GPU programming and LLM internals",
    "modules": [
        {
            "module_index": 1,
            "title": "Transformer Math From Scratch",
            "description": (
                "Implement the core mathematical operations of a transformer — "
                "RMSNorm, softmax, attention, and feed-forward layers — using only NumPy."
            ),
        },
        {
            "module_index": 2,
            "title": "Building a Naive Inference Engine",
            "description": (
                "Wire the transformer building blocks into a complete inference engine "
                "that generates text token by token. Measure baseline performance."
            ),
        },
        {
            "module_index": 3,
            "title": "Performance Analysis and the Roofline Model",
            "description": (
                "Understand WHY the naive implementation is slow using the roofline model. "
                "Learn to identify memory-bound vs compute-bound operations."
            ),
        },
        {
            "module_index": 4,
            "title": "Optimization Techniques",
            "description": (
                "Apply key optimization techniques: quantization, KV caching, "
                "parallelism, and memory access pattern optimization."
            ),
        },
    ],
}


# =============================================================================
# Module 1: Transformer Math From Scratch
# =============================================================================

MODULE_1_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# Module 1: Transformer Math From Scratch\n\n"
            "In this module, you'll implement the core mathematical building blocks of a "
            "transformer model using only NumPy. These are the same operations that run "
            "billions of times during LLM inference — understanding them at the math level "
            "is the foundation for understanding performance optimization.\n\n"
            "**What you'll build:**\n"
            "- RMSNorm (the normalization used in modern LLMs like LLaMA)\n"
            "- Numerically stable softmax\n"
            "- Scaled dot-product attention\n"
            "- A complete transformer block\n\n"
            "**Prerequisites:** NumPy basics, linear algebra (matrix multiplication)\n\n"
            "**Source:** Based on [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) by Andrew Chan"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "import numpy as np\n"
            "import time\n"
            "\n"
            "# Utility function for checking your implementations\n"
            "def check_close(actual, expected, name, rtol=1e-5):\n"
            '    """Check that actual and expected arrays are close."""\n'
            "    if not np.allclose(actual, expected, rtol=rtol):\n"
            "        max_diff = np.max(np.abs(actual - expected))\n"
            '        raise AssertionError(\n'
            '            f"{name}: max difference {max_diff:.2e} exceeds tolerance.\\n"\n'
            '            f"  Expected: {expected}\\n"\n'
            '            f"  Got:      {actual}"\n'
            "        )\n"
            '    print(f"\\u2713 {name} passed!")\n'
            "\n"
            "np.random.seed(42)\n"
            'print("Setup complete.")'
        ),
    },
    # --- Exercise 1: RMSNorm ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 1: RMSNorm\n\n"
            "Modern LLMs (LLaMA, Mistral, etc.) use **RMS Normalization** instead of "
            "Layer Normalization. It's simpler and slightly faster because it skips the "
            "mean-centering step.\n\n"
            "The formula is:\n\n"
            "$$\\text{RMSNorm}(x) = \\frac{x}{\\text{RMS}(x)} \\cdot \\gamma$$\n\n"
            "where:\n\n"
            "$$\\text{RMS}(x) = \\sqrt{\\frac{1}{d} \\sum_{i=1}^{d} x_i^2 + \\epsilon}$$\n\n"
            "- $x$ is an input vector of dimension $d$\n"
            "- $\\gamma$ (gamma) is a learned scale parameter of the same dimension\n"
            "- $\\epsilon$ is a small constant for numerical stability (typically 1e-6)"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def rmsnorm(x, gamma, eps=1e-6):\n"
            '    """Compute RMS Normalization.\n'
            "\n"
            "    RMSNorm normalizes the input by its root-mean-square value,\n"
            "    then scales by a learned parameter gamma. Unlike LayerNorm,\n"
            "    it does NOT subtract the mean first.\n"
            "\n"
            "    The computation:\n"
            "    1. Compute RMS = sqrt(mean(x^2) + eps)\n"
            "    2. Normalize: x_norm = x / RMS\n"
            "    3. Scale: output = x_norm * gamma\n"
            "\n"
            "    Args:\n"
            "        x: Input array of shape (*, d) where * means any number of\n"
            "           leading dimensions. Normalization happens over the last axis.\n"
            "        gamma: Scale parameter of shape (d,)\n"
            "        eps: Small constant for numerical stability\n"
            "\n"
            "    Returns:\n"
            "        Normalized and scaled array, same shape as x\n"
            '    """\n'
            "    result = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement RMSNorm.                                                #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Compute the mean of x^2 along the last axis (keep dimensions)     #\n"
            "    #   2. Add eps and take the square root to get RMS                        #\n"
            "    #   3. Divide x by RMS                                                   #\n"
            "    #   4. Multiply by gamma                                                  #\n"
            "    #                                                                         #\n"
            "    # Hint: Use np.mean(..., axis=-1, keepdims=True)                          #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return result"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your RMSNorm implementation =====\n"
            "\n"
            "# Test 1: Simple case\n"
            "x1 = np.array([[1.0, 2.0, 3.0, 4.0]])\n"
            "gamma1 = np.ones(4)\n"
            "expected1 = x1 / np.sqrt(np.mean(x1**2, axis=-1, keepdims=True) + 1e-6)\n"
            "result1 = rmsnorm(x1, gamma1)\n"
            "check_close(result1, expected1, 'RMSNorm basic')\n"
            "\n"
            "# Test 2: Batch of vectors\n"
            "x2 = np.array([[1.0, 0.0, -1.0, 0.0],\n"
            "               [2.0, 2.0, 2.0, 2.0]])\n"
            "gamma2 = np.array([1.0, 2.0, 0.5, 1.0])\n"
            "result2 = rmsnorm(x2, gamma2)\n"
            "# After RMSNorm, the output should have the same shape\n"
            "assert result2.shape == x2.shape, f'Shape mismatch: {result2.shape} vs {x2.shape}'\n"
            "# For the uniform row [2,2,2,2], RMS=2, so normalized = [1,1,1,1], then * gamma\n"
            "check_close(result2[1], gamma2, 'RMSNorm uniform row')\n"
            "\n"
            "# Test 3: Gamma scaling\n"
            "x3 = np.random.randn(3, 8)\n"
            "gamma3 = np.random.randn(8) * 0.5 + 1.0\n"
            "result3 = rmsnorm(x3, gamma3)\n"
            "assert result3.shape == x3.shape\n"
            "print('\\u2713 RMSNorm shape test passed!')\n"
            "\n"
            "print('\\nAll RMSNorm tests passed!')"
        ),
    },
    # --- Exercise 2: Softmax ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 2: Numerically Stable Softmax\n\n"
            "Softmax converts a vector of scores into a probability distribution. "
            "In transformers, it's used in the attention mechanism to convert "
            "similarity scores into attention weights.\n\n"
            "The naive formula is:\n\n"
            "$$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$$\n\n"
            "But this overflows for large values of $x$! The **numerically stable** version "
            "subtracts the max first:\n\n"
            "$$\\text{softmax}(x_i) = \\frac{e^{x_i - \\max(x)}}{\\sum_j e^{x_j - \\max(x)}}$$\n\n"
            "This is mathematically identical but avoids overflow."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def softmax(x):\n"
            '    """Compute numerically stable softmax over the last axis.\n'
            "\n"
            "    The numerically stable version:\n"
            "    1. Subtract max(x) along the last axis (prevents overflow)\n"
            "    2. Compute exp of the shifted values\n"
            "    3. Divide by the sum of exp values\n"
            "\n"
            "    Args:\n"
            "        x: Input array of shape (*, n) where * is any leading dimensions.\n"
            "           Softmax is computed over the last axis.\n"
            "\n"
            "    Returns:\n"
            "        Array of same shape where each row along the last axis sums to 1.0\n"
            '    """\n'
            "    result = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement numerically stable softmax.                             #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Find the max of x along axis=-1 (keepdims=True)                   #\n"
            "    #   2. Subtract the max from x (this is the stability trick)              #\n"
            "    #   3. Compute exp of the shifted values                                  #\n"
            "    #   4. Divide by the sum along axis=-1 (keepdims=True)                   #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return result"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your softmax implementation =====\n"
            "\n"
            "# Test 1: Output sums to 1\n"
            "x1 = np.array([[1.0, 2.0, 3.0]])\n"
            "result1 = softmax(x1)\n"
            "assert np.allclose(result1.sum(axis=-1), 1.0), f'Softmax should sum to 1, got {result1.sum()}'\n"
            "print('\\u2713 Softmax sums to 1')\n"
            "\n"
            "# Test 2: Known values\n"
            "x2 = np.array([[0.0, 0.0, 0.0]])\n"
            "result2 = softmax(x2)\n"
            "expected2 = np.array([[1/3, 1/3, 1/3]])\n"
            "check_close(result2, expected2, 'Softmax uniform')\n"
            "\n"
            "# Test 3: Numerical stability — this would overflow with naive softmax!\n"
            "x3 = np.array([[1000.0, 1001.0, 1002.0]])\n"
            "result3 = softmax(x3)\n"
            "assert np.all(np.isfinite(result3)), 'Softmax overflowed! Did you subtract the max?'\n"
            "assert np.allclose(result3.sum(axis=-1), 1.0)\n"
            "print('\\u2713 Softmax numerical stability passed!')\n"
            "\n"
            "# Test 4: Batch softmax\n"
            "x4 = np.random.randn(4, 6)\n"
            "result4 = softmax(x4)\n"
            "assert result4.shape == x4.shape\n"
            "assert np.allclose(result4.sum(axis=-1), np.ones(4))\n"
            "print('\\u2713 Batch softmax passed!')\n"
            "\n"
            "print('\\nAll softmax tests passed!')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 1:** Why do we subtract `max(x)` before computing `exp`? "
            "What happens if we don't, and the values in `x` are very large (e.g., 1000)? "
            "Why is this transformation mathematically valid (i.e., why does it give the "
            "same result)?\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Exercise 3: Attention ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 3: Scaled Dot-Product Attention\n\n"
            "This is the heart of the transformer. Attention allows each position to "
            "\"look at\" every other position and decide what information to gather.\n\n"
            "The formula:\n\n"
            "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$\n\n"
            "where:\n"
            "- $Q$ (query): what am I looking for? — shape `(seq_len, d_k)`\n"
            "- $K$ (key): what do I contain? — shape `(seq_len, d_k)`\n"
            "- $V$ (value): what information do I provide? — shape `(seq_len, d_v)`\n"
            "- $d_k$: the dimension of keys (used for scaling)\n\n"
            "The scaling by $\\sqrt{d_k}$ prevents the dot products from growing too "
            "large as the dimension increases, which would push softmax into regions "
            "where its gradients are tiny.\n\n"
            "**For autoregressive (causal) LLMs**, we also apply a causal mask: position $i$ "
            "can only attend to positions $\\leq i$. We implement this by setting future "
            "positions to $-\\infty$ before softmax."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def attention(Q, K, V, causal=True):\n"
            '    """Compute scaled dot-product attention.\n'
            "\n"
            "    The computation:\n"
            "    1. Compute attention scores: Q @ K^T / sqrt(d_k)\n"
            "    2. If causal, mask future positions with -inf\n"
            "    3. Apply softmax to get attention weights\n"
            "    4. Multiply weights by V to get output\n"
            "\n"
            "    Args:\n"
            "        Q: Query matrix of shape (seq_len, d_k)\n"
            "        K: Key matrix of shape (seq_len, d_k)\n"
            "        V: Value matrix of shape (seq_len, d_v)\n"
            "        causal: If True, apply causal mask (positions can only attend\n"
            "                to earlier positions). Required for autoregressive LLMs.\n"
            "\n"
            "    Returns:\n"
            "        output: Attention output of shape (seq_len, d_v)\n"
            "        weights: Attention weights of shape (seq_len, seq_len)\n"
            '    """\n'
            "    seq_len, d_k = Q.shape\n"
            "    output = None\n"
            "    weights = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement scaled dot-product attention.                           #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Compute scores = Q @ K^T / sqrt(d_k)                              #\n"
            "    #   2. If causal, create a mask where mask[i,j] = True if j > i           #\n"
            "    #      and set masked positions to -1e9 (approximating -inf)              #\n"
            "    #   3. Apply your softmax function to scores                              #\n"
            "    #   4. Compute output = weights @ V                                       #\n"
            "    #                                                                         #\n"
            "    # Hint: Use np.triu(np.ones(...), k=1) to create the causal mask         #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return output, weights"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your attention implementation =====\n"
            "\n"
            "# Test 1: Shape check\n"
            "seq_len, d_k, d_v = 4, 8, 8\n"
            "Q_test = np.random.randn(seq_len, d_k)\n"
            "K_test = np.random.randn(seq_len, d_k)\n"
            "V_test = np.random.randn(seq_len, d_v)\n"
            "\n"
            "out, wts = attention(Q_test, K_test, V_test, causal=True)\n"
            "assert out.shape == (seq_len, d_v), f'Output shape should be ({seq_len}, {d_v}), got {out.shape}'\n"
            "assert wts.shape == (seq_len, seq_len), f'Weights shape should be ({seq_len}, {seq_len}), got {wts.shape}'\n"
            "print('\\u2713 Attention shapes correct')\n"
            "\n"
            "# Test 2: Attention weights sum to 1\n"
            "assert np.allclose(wts.sum(axis=-1), 1.0), 'Attention weights should sum to 1 along last axis'\n"
            "print('\\u2713 Attention weights sum to 1')\n"
            "\n"
            "# Test 3: Causal mask — first row should only attend to position 0\n"
            "assert np.allclose(wts[0, 0], 1.0), 'First position should attend only to itself'\n"
            "assert np.allclose(wts[0, 1:], 0.0, atol=1e-6), 'First position should not attend to future'\n"
            "print('\\u2713 Causal mask works correctly')\n"
            "\n"
            "# Test 4: Without causal mask, all positions can attend everywhere\n"
            "out_nc, wts_nc = attention(Q_test, K_test, V_test, causal=False)\n"
            "# No positions should be zero (unless by coincidence)\n"
            "assert wts_nc.min() > 0, 'Without causal mask, all weights should be positive'\n"
            "print('\\u2713 Non-causal attention works')\n"
            "\n"
            "print('\\nAll attention tests passed!')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 2:** The attention scores are divided by $\\sqrt{d_k}$ before "
            "softmax. What would happen if we skipped this scaling when $d_k$ is large "
            "(e.g., 128)? Think about what the dot product values look like and how softmax "
            "behaves with very large inputs.\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Exercise 4: Multi-Head Attention ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 4: Multi-Head Attention\n\n"
            "Instead of one big attention computation, transformers split the "
            "representation into multiple **heads**. Each head independently computes "
            "attention on a subspace of the full dimension, allowing the model to "
            "attend to different types of information simultaneously.\n\n"
            "With `n_heads` heads and model dimension `d_model`:\n"
            "- Each head operates on dimension `d_head = d_model / n_heads`\n"
            "- Q, K, V are projected to `(seq_len, n_heads, d_head)`\n"
            "- Attention is computed independently per head\n"
            "- Results are concatenated and projected back to `d_model`\n\n"
            "For this exercise, you'll use the `attention` function from Exercise 3."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads, causal=True):\n"
            '    """Compute multi-head attention.\n'
            "\n"
            "    The computation:\n"
            "    1. Project x into Q, K, V using weight matrices\n"
            "    2. Reshape to split into n_heads\n"
            "    3. Compute attention independently per head\n"
            "    4. Concatenate heads and project with W_o\n"
            "\n"
            "    Args:\n"
            "        x: Input of shape (seq_len, d_model)\n"
            "        W_q: Query projection weights (d_model, d_model)\n"
            "        W_k: Key projection weights (d_model, d_model)\n"
            "        W_v: Value projection weights (d_model, d_model)\n"
            "        W_o: Output projection weights (d_model, d_model)\n"
            "        n_heads: Number of attention heads\n"
            "        causal: Whether to apply causal masking\n"
            "\n"
            "    Returns:\n"
            "        output: Shape (seq_len, d_model)\n"
            '    """\n'
            "    seq_len, d_model = x.shape\n"
            "    d_head = d_model // n_heads\n"
            "    output = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement multi-head attention.                                   #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Project: Q = x @ W_q, K = x @ W_k, V = x @ W_v                   #\n"
            "    #   2. Reshape each to (seq_len, n_heads, d_head)                         #\n"
            "    #   3. For each head, call attention(Q_h, K_h, V_h, causal)               #\n"
            "    #      where Q_h = Q[:, h, :] etc.                                       #\n"
            "    #   4. Concatenate head outputs → shape (seq_len, d_model)                #\n"
            "    #   5. Project: output = concatenated @ W_o                               #\n"
            "    #                                                                         #\n"
            "    # Hint: You can loop over heads. A vectorized version is possible but     #\n"
            "    #       the loop is clearer and fine for learning.                        #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return output"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your multi-head attention implementation =====\n"
            "\n"
            "seq_len, d_model, n_heads = 4, 16, 4\n"
            "x_test = np.random.randn(seq_len, d_model)\n"
            "W_q = np.random.randn(d_model, d_model) * 0.1\n"
            "W_k = np.random.randn(d_model, d_model) * 0.1\n"
            "W_v = np.random.randn(d_model, d_model) * 0.1\n"
            "W_o = np.random.randn(d_model, d_model) * 0.1\n"
            "\n"
            "mha_out = multi_head_attention(x_test, W_q, W_k, W_v, W_o, n_heads)\n"
            "assert mha_out.shape == (seq_len, d_model), f'Expected ({seq_len}, {d_model}), got {mha_out.shape}'\n"
            "print('\\u2713 Multi-head attention shape correct')\n"
            "\n"
            "# Test with different number of heads\n"
            "mha_out2 = multi_head_attention(x_test, W_q, W_k, W_v, W_o, n_heads=2)\n"
            "assert mha_out2.shape == (seq_len, d_model)\n"
            "print('\\u2713 Works with different head counts')\n"
            "\n"
            "# The output should change when input changes\n"
            "x_test2 = np.random.randn(seq_len, d_model)\n"
            "mha_out3 = multi_head_attention(x_test2, W_q, W_k, W_v, W_o, n_heads)\n"
            "assert not np.allclose(mha_out, mha_out3), 'Different input should give different output'\n"
            "print('\\u2713 Output depends on input')\n"
            "\n"
            "print('\\nAll multi-head attention tests passed!')"
        ),
    },
    # --- Summary ---
    {
        "cell_type": "markdown",
        "source": (
            "## Module 1 Summary\n\n"
            "You've implemented the core mathematical operations of a transformer:\n\n"
            "| Component | Purpose | Complexity |\n"
            "|-----------|---------|------------|\n"
            "| **RMSNorm** | Stabilize activations | O(d) |\n"
            "| **Softmax** | Convert scores → probabilities | O(n) |\n"
            "| **Attention** | Let positions exchange information | O(n² · d) |\n"
            "| **Multi-Head Attention** | Multiple parallel attention patterns | O(n² · d) |\n\n"
            "**Key insight for optimization:** Notice that attention is O(n² · d) — "
            "it grows quadratically with sequence length. But during inference, when we "
            "generate one token at a time, the **query** is a single vector (not a matrix), "
            "so the dominant cost is actually the matrix-vector multiplications in the "
            "projections. This is what we'll optimize in later modules.\n\n"
            "**Next:** In Module 2, you'll wire these building blocks into a complete "
            "inference engine that generates text."
        ),
    },
]


# =============================================================================
# Module 2: Building a Naive Inference Engine
# =============================================================================

MODULE_2_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# Module 2: Building a Naive Inference Engine\n\n"
            "Now that you have the transformer building blocks, let's wire them into a "
            "complete inference engine. You'll build a tiny but complete LLM that can "
            "generate text token by token.\n\n"
            "**What you'll build:**\n"
            "- Token embedding and positional encoding\n"
            "- Feed-forward network (MLP)\n"
            "- Complete transformer block (attention + MLP + residuals)\n"
            "- Autoregressive text generation loop\n\n"
            "**Goal:** Generate text and measure our baseline performance (tok/s).\n\n"
            "**Source:** Based on [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) by Andrew Chan"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "import numpy as np\n"
            "import time\n"
            "\n"
            "np.random.seed(42)\n"
            "\n"
            "# Re-use implementations from Module 1\n"
            "# (In a real setup these would be imported; here we define them inline)\n"
            "\n"
            "def rmsnorm(x, gamma, eps=1e-6):\n"
            '    """RMSNorm from Module 1."""\n'
            "    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)\n"
            "    return (x / rms) * gamma\n"
            "\n"
            "def softmax(x):\n"
            '    """Numerically stable softmax from Module 1."""\n'
            "    x_max = np.max(x, axis=-1, keepdims=True)\n"
            "    exp_x = np.exp(x - x_max)\n"
            "    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)\n"
            "\n"
            "def attention(Q, K, V, causal=True):\n"
            '    """Scaled dot-product attention from Module 1."""\n'
            "    seq_len, d_k = Q.shape\n"
            "    scores = Q @ K.T / np.sqrt(d_k)\n"
            "    if causal:\n"
            "        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)\n"
            "        scores[mask] = -1e9\n"
            "    weights = softmax(scores)\n"
            "    return weights @ V, weights\n"
            "\n"
            "def check_close(actual, expected, name, rtol=1e-5):\n"
            "    if not np.allclose(actual, expected, rtol=rtol):\n"
            "        max_diff = np.max(np.abs(actual - expected))\n"
            "        raise AssertionError(f'{name}: max diff {max_diff:.2e}')\n"
            "    print(f'\\u2713 {name} passed!')\n"
            "\n"
            "# Define a tiny model config for our exercises\n"
            "class TinyConfig:\n"
            "    vocab_size = 256     # Small vocabulary (byte-level)\n"
            "    d_model = 64         # Model dimension\n"
            "    n_heads = 4          # Number of attention heads\n"
            "    n_layers = 2         # Number of transformer layers\n"
            "    d_ff = 256           # Feed-forward hidden dimension\n"
            "    max_seq_len = 128    # Maximum sequence length\n"
            "\n"
            "config = TinyConfig()\n"
            "print(f'Model config: d_model={config.d_model}, n_heads={config.n_heads}, '\n"
            "      f'n_layers={config.n_layers}, vocab_size={config.vocab_size}')"
        ),
    },
    # --- Exercise 1: Embedding ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 1: Token Embedding\n\n"
            "The first step in an LLM is converting token IDs (integers) into dense "
            "vectors. This is just a lookup table — each token ID indexes into an "
            "embedding matrix.\n\n"
            "For this exercise, we'll also add **positional embeddings** — another "
            "lookup table indexed by position. This tells the model where each token "
            "is in the sequence."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def embed_tokens(token_ids, token_embeddings, position_embeddings):\n"
            '    """Convert token IDs to dense vectors with positional information.\n'
            "\n"
            "    This is the very first operation in an LLM forward pass.\n"
            "    Each token ID is used to look up a row in the embedding matrix,\n"
            "    then the position embedding for that position is added.\n"
            "\n"
            "    Args:\n"
            "        token_ids: Array of token IDs, shape (seq_len,), dtype int\n"
            "        token_embeddings: Embedding matrix, shape (vocab_size, d_model)\n"
            "        position_embeddings: Position matrix, shape (max_seq_len, d_model)\n"
            "\n"
            "    Returns:\n"
            "        Embedded tokens, shape (seq_len, d_model)\n"
            '    """\n'
            "    result = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement token + position embedding.                             #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Look up token embeddings: index into token_embeddings              #\n"
            "    #      using token_ids (fancy indexing: token_embeddings[token_ids])       #\n"
            "    #   2. Look up position embeddings for positions 0..seq_len-1             #\n"
            "    #   3. Add them together                                                  #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return result"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your embedding implementation =====\n"
            "\n"
            "tok_emb = np.random.randn(config.vocab_size, config.d_model) * 0.02\n"
            "pos_emb = np.random.randn(config.max_seq_len, config.d_model) * 0.02\n"
            "\n"
            "# Test 1: Single token\n"
            "ids_1 = np.array([42])\n"
            "result_1 = embed_tokens(ids_1, tok_emb, pos_emb)\n"
            "expected_1 = tok_emb[42:43] + pos_emb[0:1]\n"
            "check_close(result_1, expected_1, 'Single token embedding')\n"
            "\n"
            "# Test 2: Multiple tokens\n"
            "ids_2 = np.array([10, 20, 30, 40])\n"
            "result_2 = embed_tokens(ids_2, tok_emb, pos_emb)\n"
            "assert result_2.shape == (4, config.d_model), f'Shape should be (4, {config.d_model})'\n"
            "# First token: tok_emb[10] + pos_emb[0]\n"
            "check_close(result_2[0], tok_emb[10] + pos_emb[0], 'First token position')\n"
            "# Last token: tok_emb[40] + pos_emb[3]\n"
            "check_close(result_2[3], tok_emb[40] + pos_emb[3], 'Last token position')\n"
            "\n"
            "print('\\nAll embedding tests passed!')"
        ),
    },
    # --- Exercise 2: Feed-Forward Network ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 2: Feed-Forward Network (MLP)\n\n"
            "Each transformer block contains a feed-forward network (FFN/MLP) applied "
            "independently to each position. Modern LLMs like LLaMA use a \"gated\" "
            "variant called **SwiGLU**:\n\n"
            "$$\\text{FFN}(x) = (\\text{SiLU}(x \\cdot W_1) \\odot (x \\cdot W_3)) \\cdot W_2$$\n\n"
            "where:\n"
            "- $W_1, W_3$: \"gate\" and \"up\" projections from `d_model` to `d_ff`\n"
            "- $W_2$: \"down\" projection from `d_ff` back to `d_model`\n"
            "- $\\text{SiLU}(x) = x \\cdot \\sigma(x)$ (SiLU / Swish activation)\n"
            "- $\\odot$: element-wise multiplication"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def silu(x):\n"
            '    """SiLU (Swish) activation: x * sigmoid(x)."""\n'
            "    return x * (1.0 / (1.0 + np.exp(-x)))\n"
            "\n"
            "\n"
            "def feed_forward(x, W_gate, W_up, W_down):\n"
            '    """SwiGLU feed-forward network.\n'
            "\n"
            "    This is the MLP component of each transformer block.\n"
            "    It independently transforms each position's representation.\n"
            "\n"
            "    The computation:\n"
            "    1. Gate: gate = SiLU(x @ W_gate)\n"
            "    2. Up:   up = x @ W_up\n"
            "    3. Element-wise product: hidden = gate * up\n"
            "    4. Down: output = hidden @ W_down\n"
            "\n"
            "    Args:\n"
            "        x: Input of shape (seq_len, d_model)\n"
            "        W_gate: Gate projection, shape (d_model, d_ff)\n"
            "        W_up: Up projection, shape (d_model, d_ff)\n"
            "        W_down: Down projection, shape (d_ff, d_model)\n"
            "\n"
            "    Returns:\n"
            "        Output of shape (seq_len, d_model)\n"
            '    """\n'
            "    output = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement the SwiGLU feed-forward network.                        #\n"
            "    #                                                                         #\n"
            "    # Follow the 4 steps in the docstring above.                              #\n"
            "    # Use the silu() function provided above for the activation.              #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return output"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your feed-forward implementation =====\n"
            "\n"
            "d_ff = config.d_ff\n"
            "d_model = config.d_model\n"
            "W_gate = np.random.randn(d_model, d_ff) * 0.02\n"
            "W_up = np.random.randn(d_model, d_ff) * 0.02\n"
            "W_down = np.random.randn(d_ff, d_model) * 0.02\n"
            "\n"
            "# Test 1: Shape\n"
            "x_ff = np.random.randn(4, d_model)\n"
            "result_ff = feed_forward(x_ff, W_gate, W_up, W_down)\n"
            "assert result_ff.shape == (4, d_model), f'Expected (4, {d_model}), got {result_ff.shape}'\n"
            "print('\\u2713 FFN shape correct')\n"
            "\n"
            "# Test 2: Verify the computation manually for a single position\n"
            "x_single = x_ff[0:1]\n"
            "gate = silu(x_single @ W_gate)\n"
            "up = x_single @ W_up\n"
            "expected_single = (gate * up) @ W_down\n"
            "check_close(result_ff[0:1], expected_single, 'FFN manual check')\n"
            "\n"
            "# Test 3: Different input gives different output\n"
            "x_ff2 = np.random.randn(4, d_model)\n"
            "result_ff2 = feed_forward(x_ff2, W_gate, W_up, W_down)\n"
            "assert not np.allclose(result_ff, result_ff2)\n"
            "print('\\u2713 FFN output varies with input')\n"
            "\n"
            "print('\\nAll FFN tests passed!')"
        ),
    },
    # --- Exercise 3: Transformer Block ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 3: Complete Transformer Block\n\n"
            "A transformer block combines attention and the feed-forward network with "
            "**residual connections** and **RMSNorm**. The structure (pre-norm variant "
            "used in LLaMA/Mistral):\n\n"
            "```\n"
            "x → RMSNorm → Attention → + (residual) → RMSNorm → FFN → + (residual) → output\n"
            "    |                      ↑                |               ↑\n"
            "    └──────────────────────┘                └───────────────┘\n"
            "```\n\n"
            "The residual connections (adding the input back) are crucial — they let "
            "gradients flow directly through the network, enabling training of very "
            "deep models."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def transformer_block(x, weights):\n"
            '    """One transformer block: attention + FFN with residuals and RMSNorm.\n'
            "\n"
            "    Pre-norm architecture (used in LLaMA, Mistral, etc.):\n"
            "    1. Normalize input with RMSNorm\n"
            "    2. Apply multi-head attention\n"
            "    3. Add residual connection (input + attention output)\n"
            "    4. Normalize with RMSNorm\n"
            "    5. Apply feed-forward network\n"
            "    6. Add residual connection\n"
            "\n"
            "    Args:\n"
            "        x: Input tensor of shape (seq_len, d_model)\n"
            "        weights: Dictionary containing:\n"
            "            'attn_norm': RMSNorm gamma for pre-attention, shape (d_model,)\n"
            "            'W_q', 'W_k', 'W_v', 'W_o': Attention weight matrices\n"
            "            'ffn_norm': RMSNorm gamma for pre-FFN, shape (d_model,)\n"
            "            'W_gate', 'W_up', 'W_down': FFN weight matrices\n"
            "\n"
            "    Returns:\n"
            "        Output tensor of shape (seq_len, d_model)\n"
            '    """\n'
            "    seq_len, d_model = x.shape\n"
            "    n_heads = d_model // (weights['W_q'].shape[1] // d_model) if d_model > 0 else 4\n"
            "    # Simpler: infer n_heads from config\n"
            "    n_heads = config.n_heads\n"
            "    output = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement a complete transformer block.                            #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. x_norm = rmsnorm(x, weights['attn_norm'])                          #\n"
            "    #   2. Compute Q, K, V by multiplying x_norm with W_q, W_k, W_v          #\n"
            "    #   3. Reshape Q, K, V to (seq_len, n_heads, d_head)                      #\n"
            "    #   4. Compute attention per head, concatenate, project with W_o          #\n"
            "    #      (You can reuse your multi_head_attention or do it manually)        #\n"
            "    #   5. Add residual: h = x + attn_output                                  #\n"
            "    #   6. h_norm = rmsnorm(h, weights['ffn_norm'])                           #\n"
            "    #   7. ffn_output = feed_forward(h_norm, W_gate, W_up, W_down)            #\n"
            "    #   8. output = h + ffn_output                                            #\n"
            "    #                                                                         #\n"
            "    # Hint: For simplicity, you can use a loop over heads for step 4.        #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return output"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your transformer block =====\n"
            "\n"
            "def make_block_weights(d_model, d_ff):\n"
            '    """Create random weights for one transformer block."""\n'
            "    scale = 0.02\n"
            "    return {\n"
            "        'attn_norm': np.ones(d_model),\n"
            "        'W_q': np.random.randn(d_model, d_model) * scale,\n"
            "        'W_k': np.random.randn(d_model, d_model) * scale,\n"
            "        'W_v': np.random.randn(d_model, d_model) * scale,\n"
            "        'W_o': np.random.randn(d_model, d_model) * scale,\n"
            "        'ffn_norm': np.ones(d_model),\n"
            "        'W_gate': np.random.randn(d_model, d_ff) * scale,\n"
            "        'W_up': np.random.randn(d_model, d_ff) * scale,\n"
            "        'W_down': np.random.randn(d_ff, d_model) * scale,\n"
            "    }\n"
            "\n"
            "block_w = make_block_weights(config.d_model, config.d_ff)\n"
            "x_block = np.random.randn(4, config.d_model)\n"
            "\n"
            "# Test 1: Shape preserved\n"
            "out_block = transformer_block(x_block, block_w)\n"
            "assert out_block.shape == x_block.shape, f'Shape should be preserved: {x_block.shape}'\n"
            "print('\\u2713 Transformer block shape correct')\n"
            "\n"
            "# Test 2: Residual connection — output should be close to input for small weights\n"
            "# (because attention and FFN outputs are small with small weights)\n"
            "diff = np.mean(np.abs(out_block - x_block))\n"
            "assert diff < 5.0, f'With small weights, output should be close to input (diff={diff:.4f})'\n"
            "print(f'\\u2713 Residual connection working (mean diff from input: {diff:.4f})')\n"
            "\n"
            "# Test 3: Two blocks in sequence\n"
            "block_w2 = make_block_weights(config.d_model, config.d_ff)\n"
            "out_block2 = transformer_block(out_block, block_w2)\n"
            "assert out_block2.shape == x_block.shape\n"
            "print('\\u2713 Blocks can be stacked')\n"
            "\n"
            "print('\\nAll transformer block tests passed!')"
        ),
    },
    # --- Exercise 4: Generation Loop ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 4: Autoregressive Generation\n\n"
            "This is the climax — the generation loop that makes an LLM produce text. "
            "The process is beautifully simple:\n\n"
            "```\n"
            "1. Start with a prompt (sequence of token IDs)\n"
            "2. Forward pass through the model → get logits for next token\n"
            "3. Sample next token from logits (or take argmax)\n"
            "4. Append new token to sequence\n"
            "5. Repeat from step 2\n"
            "```\n\n"
            "We'll combine everything into a tiny model and generate tokens, then "
            "**measure throughput** — the number of tokens generated per second."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "class TinyLLM:\n"
            '    """A complete (tiny) LLM for inference.\n'
            "\n"
            "    This combines all the building blocks into a full model.\n"
            "    Weights are randomly initialized (so output will be gibberish,\n"
            "    but the computation is identical to a real LLM).\n"
            '    """\n'
            "    def __init__(self, config):\n"
            "        self.config = config\n"
            "        scale = 0.02\n"
            "        # Token and position embeddings\n"
            "        self.tok_emb = np.random.randn(config.vocab_size, config.d_model) * scale\n"
            "        self.pos_emb = np.random.randn(config.max_seq_len, config.d_model) * scale\n"
            "        # Transformer blocks\n"
            "        self.blocks = [make_block_weights(config.d_model, config.d_ff)\n"
            "                       for _ in range(config.n_layers)]\n"
            "        # Final norm and output projection\n"
            "        self.final_norm = np.ones(config.d_model)\n"
            "        self.output_proj = np.random.randn(config.d_model, config.vocab_size) * scale\n"
            "\n"
            "    def forward(self, token_ids):\n"
            '        \"\"\"Forward pass: token IDs → logits for next token.\n'
            "\n"
            "        Args:\n"
            "            token_ids: Array of shape (seq_len,), dtype int\n"
            "\n"
            "        Returns:\n"
            "            logits: Array of shape (vocab_size,) — scores for next token\n"
            '        \"\"\"\n'
            "        # Embed\n"
            "        x = embed_tokens(token_ids, self.tok_emb, self.pos_emb)\n"
            "        # Transformer blocks\n"
            "        for block_w in self.blocks:\n"
            "            x = transformer_block(x, block_w)\n"
            "        # Final norm + project to vocab\n"
            "        x = rmsnorm(x, self.final_norm)\n"
            "        # Only need logits for the LAST position\n"
            "        logits = x[-1] @ self.output_proj\n"
            "        return logits\n"
            "\n"
            "\n"
            "def generate(model, prompt_ids, max_new_tokens=20, temperature=1.0):\n"
            '    \"\"\"Generate tokens autoregressively.\n'
            "\n"
            "    This is the core generation loop of an LLM.\n"
            "\n"
            "    Args:\n"
            "        model: TinyLLM instance\n"
            "        prompt_ids: Starting token IDs, list or array of ints\n"
            "        max_new_tokens: How many new tokens to generate\n"
            "        temperature: Sampling temperature (higher = more random)\n"
            "\n"
            "    Returns:\n"
            "        all_ids: Complete sequence including prompt + generated tokens\n"
            "        elapsed: Total generation time in seconds\n"
            '    \"\"\"\n'
            "    all_ids = list(prompt_ids)\n"
            "    elapsed = 0.0\n"
            "    ###########################################################################\n"
            "    # TODO: Implement the autoregressive generation loop.                     #\n"
            "    #                                                                         #\n"
            "    # For each new token:                                                     #\n"
            "    #   1. Start a timer (time.perf_counter())                                #\n"
            "    #   2. Run model.forward(np.array(all_ids)) to get logits                 #\n"
            "    #   3. Apply temperature: logits = logits / temperature                   #\n"
            "    #   4. Convert to probabilities with softmax                              #\n"
            "    #   5. Sample: next_token = np.random.choice(vocab_size, p=probs)         #\n"
            "    #   6. Append next_token to all_ids                                       #\n"
            "    #   7. Accumulate elapsed time                                            #\n"
            "    #                                                                         #\n"
            "    # After the loop, return (all_ids, elapsed)                               #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return all_ids, elapsed"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your generation implementation =====\n"
            "\n"
            "model = TinyLLM(config)\n"
            "prompt = [65, 66, 67]  # 'A', 'B', 'C' in ASCII/byte-level encoding\n"
            "\n"
            "# Generate!\n"
            "generated_ids, gen_time = generate(model, prompt, max_new_tokens=10, temperature=1.0)\n"
            "\n"
            "# Test 1: Output length\n"
            "expected_len = len(prompt) + 10\n"
            "assert len(generated_ids) == expected_len, (\n"
            "    f'Expected {expected_len} tokens, got {len(generated_ids)}'\n"
            ")\n"
            "print(f'\\u2713 Generated {expected_len} tokens')\n"
            "\n"
            "# Test 2: Prompt is preserved\n"
            "assert generated_ids[:3] == prompt, 'Prompt should be preserved at the start'\n"
            "print('\\u2713 Prompt preserved')\n"
            "\n"
            "# Test 3: All tokens are valid\n"
            "assert all(0 <= t < config.vocab_size for t in generated_ids), 'All tokens should be valid'\n"
            "print('\\u2713 All tokens valid')\n"
            "\n"
            "# Measure throughput\n"
            "n_generated = len(generated_ids) - len(prompt)\n"
            "if gen_time > 0:\n"
            "    tok_per_sec = n_generated / gen_time\n"
            "    print(f'\\n--- BASELINE PERFORMANCE ---')\n"
            "    print(f'Generated {n_generated} tokens in {gen_time:.3f}s')\n"
            "    print(f'Throughput: {tok_per_sec:.1f} tok/s')\n"
            "    print(f'----------------------------')\n"
            "else:\n"
            "    print('Generation was too fast to measure (unlikely but possible with tiny model)')\n"
            "\n"
            "# Show the generated \"text\" (will be gibberish since weights are random)\n"
            "generated_text = ''.join(chr(min(t, 127)) for t in generated_ids)\n"
            "print(f'\\nGenerated (random weights, so gibberish): {repr(generated_text)}')\n"
            "\n"
            "print('\\nAll generation tests passed!')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 3:** Look at the throughput number you just measured. "
            "Our tiny model is small but already not instant. Real LLMs have:\n"
            "- `d_model` = 4096 (64x larger)\n"
            "- `n_layers` = 32 (16x more)\n"
            "- `d_ff` = 11008 (43x larger)\n"
            "- `vocab_size` = 32000 (125x larger)\n\n"
            "Roughly estimate: how many times slower would a real LLM be compared to our "
            "tiny model, assuming the cost is dominated by matrix multiplications? "
            "What tok/s would you expect?\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Summary ---
    {
        "cell_type": "markdown",
        "source": (
            "## Module 2 Summary\n\n"
            "You've built a complete LLM inference engine! Here's what you assembled:\n\n"
            "```\n"
            "Token IDs → Embed → [Block 1 → Block 2 → ... → Block N] → Norm → Project → Sample\n"
            "                      ↕                                                    ↕\n"
            "              (Attention + FFN)                                    (next token ID)\n"
            "                      ↕\n"
            "              (back to start)\n"
            "```\n\n"
            "**Key observation:** The generation loop is fundamentally sequential — each "
            "token depends on all previous tokens. This means we can't parallelize across "
            "tokens during generation. The only way to make it faster is to make each "
            "individual forward pass faster.\n\n"
            "**The bottleneck:** In each forward pass, the dominant operations are "
            "matrix-vector multiplications (`x @ W`). For a model with `d_model=4096`, "
            "each projection is a 4096×4096 matrix times a vector — that's 16M multiply-adds. "
            "With ~20 such operations per layer and 32 layers, that's ~10 billion operations "
            "per token.\n\n"
            "**Next:** In Module 3, we'll analyze exactly WHY this is slow and build the "
            "mental model for optimization."
        ),
    },
]


# =============================================================================
# Module 3: Performance Analysis and the Roofline Model
# =============================================================================

MODULE_3_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# Module 3: Performance Analysis and the Roofline Model\n\n"
            "Before optimizing, we need to understand **why** things are slow. "
            "The key insight from Andrew Chan's blog post is that LLM inference is "
            "**memory-bandwidth bound**, not compute-bound.\n\n"
            "In this module, you'll learn:\n"
            "- The difference between memory-bound and compute-bound operations\n"
            "- The **Roofline Model** — a framework for analyzing performance limits\n"
            "- How to identify the bottleneck in LLM inference\n"
            "- How to predict the theoretical maximum throughput\n\n"
            "**Key insight:** Understanding the bottleneck tells you which optimizations "
            "will actually help. Optimizing compute when you're memory-bound is wasted effort.\n\n"
            "**Source:** Based on [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) by Andrew Chan"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "import numpy as np\n"
            "import time\n"
            "\n"
            "def check_close(actual, expected, name, rtol=1e-2):\n"
            "    if not np.allclose(actual, expected, rtol=rtol):\n"
            "        raise AssertionError(f'{name}: expected {expected}, got {actual}')\n"
            "    print(f'\\u2713 {name} passed!')\n"
            "\n"
            "# Hardware specs (approximate, for a typical modern system)\n"
            "# You can update these to match your actual hardware\n"
            "class HardwareSpec:\n"
            "    # CPU\n"
            "    cpu_peak_gflops = 100    # Peak GFLOPS (e.g., ~100 for a modern laptop CPU)\n"
            "    cpu_mem_bandwidth_gb = 50  # Memory bandwidth in GB/s (DDR5 dual channel)\n"
            "    \n"
            "    # GPU (e.g., RTX 4090)\n"
            "    gpu_peak_gflops = 1300   # FP32 TFLOPS\n"
            "    gpu_mem_bandwidth_gb = 1000  # Memory bandwidth in GB/s (GDDR6X)\n"
            "    \n"
            "hw = HardwareSpec()\n"
            "print(f'CPU: {hw.cpu_peak_gflops} GFLOPS, {hw.cpu_mem_bandwidth_gb} GB/s bandwidth')\n"
            "print(f'GPU: {hw.gpu_peak_gflops} GFLOPS, {hw.gpu_mem_bandwidth_gb} GB/s bandwidth')"
        ),
    },
    # --- Exercise 1: Arithmetic Intensity ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 1: Arithmetic Intensity\n\n"
            "**Arithmetic intensity** is the ratio of computation to memory access:\n\n"
            "$$\\text{Arithmetic Intensity} = \\frac{\\text{FLOPs}}{\\text{Bytes accessed}}$$\n\n"
            "This single number tells you whether an operation is:\n"
            "- **Memory-bound** (low intensity): waiting for data from memory\n"
            "- **Compute-bound** (high intensity): doing math as fast as the processor allows\n\n"
            "The threshold is the hardware's **compute-to-bandwidth ratio**:\n\n"
            "$$\\text{Ridge point} = \\frac{\\text{Peak FLOPS}}{\\text{Memory Bandwidth}}$$\n\n"
            "If your arithmetic intensity is below the ridge point → memory-bound.\n"
            "If above → compute-bound."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def compute_arithmetic_intensity(M, N, batch_size=1):\n"
            '    """Compute the arithmetic intensity of a matrix-vector multiplication.\n'
            "\n"
            "    For y = W @ x, where W is (M, N) and x is (N,):\n"
            "    - FLOPs: 2 * M * N (one multiply + one add per element)\n"
            "    - Bytes read: the weight matrix W (M * N * bytes_per_element)\n"
            "                  + the input vector x (N * bytes_per_element)\n"
            "    - Bytes written: the output y (M * bytes_per_element)\n"
            "\n"
            "    For batch_size > 1 (matrix-matrix multiply, y = W @ X):\n"
            "    - FLOPs: 2 * M * N * batch_size\n"
            "    - Bytes: weights + input + output\n"
            "\n"
            "    Assumes FP32 (4 bytes per element).\n"
            "\n"
            "    Args:\n"
            "        M: Number of rows in weight matrix\n"
            "        N: Number of columns in weight matrix\n"
            "        batch_size: Number of vectors (1 for matvec, >1 for matmul)\n"
            "\n"
            "    Returns:\n"
            "        Dictionary with 'flops', 'bytes', 'arithmetic_intensity'\n"
            '    """\n'
            "    bytes_per_elem = 4  # FP32\n"
            "    result = {'flops': 0, 'bytes': 0, 'arithmetic_intensity': 0.0}\n"
            "    ###########################################################################\n"
            "    # TODO: Compute FLOPs, bytes accessed, and arithmetic intensity.          #\n"
            "    #                                                                         #\n"
            "    # Remember:                                                               #\n"
            "    #   - FLOPs = 2 * M * N * batch_size (multiply + add for each element)   #\n"
            "    #   - Bytes = weights (M*N*4) + input (N*batch_size*4)                    #\n"
            "    #             + output (M*batch_size*4)                                   #\n"
            "    #   - Arithmetic intensity = FLOPs / Bytes                                #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return result"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your arithmetic intensity computation =====\n"
            "\n"
            "# Test 1: Small matrix-vector\n"
            "# W is (4, 4), x is (4,) → FLOPs = 2*4*4 = 32, Bytes = 4*4*4 + 4*4 + 4*4 = 96\n"
            "result_small = compute_arithmetic_intensity(4, 4, batch_size=1)\n"
            "assert result_small['flops'] == 32, f\"Expected 32 FLOPs, got {result_small['flops']}\"\n"
            "assert result_small['bytes'] == 96, f\"Expected 96 bytes, got {result_small['bytes']}\"\n"
            "check_close(result_small['arithmetic_intensity'], 32/96, 'Small matvec AI')\n"
            "\n"
            "# Test 2: Realistic LLM projection (d_model=4096)\n"
            "result_llm = compute_arithmetic_intensity(4096, 4096, batch_size=1)\n"
            "print(f'\\nLLM projection (4096x4096, matvec):')\n"
            "print(f'  FLOPs: {result_llm[\"flops\"]/1e6:.1f}M')\n"
            "print(f'  Bytes: {result_llm[\"bytes\"]/1e6:.1f}MB')\n"
            "print(f'  Arithmetic Intensity: {result_llm[\"arithmetic_intensity\"]:.2f} FLOP/byte')\n"
            "\n"
            "# Test 3: Same projection but with batch (prefill phase)\n"
            "result_batch = compute_arithmetic_intensity(4096, 4096, batch_size=512)\n"
            "print(f'\\nLLM projection (4096x4096, batch=512):')\n"
            "print(f'  FLOPs: {result_batch[\"flops\"]/1e9:.1f}G')\n"
            "print(f'  Bytes: {result_batch[\"bytes\"]/1e6:.1f}MB')\n"
            "print(f'  Arithmetic Intensity: {result_batch[\"arithmetic_intensity\"]:.1f} FLOP/byte')\n"
            "\n"
            "print('\\nAll arithmetic intensity tests passed!')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 4:** Look at the arithmetic intensity values you computed.\n"
            "- For the single-vector case (batch=1, which is the *generation* phase of LLMs), "
            "the AI is ~2.0.\n"
            "- For the batched case (batch=512, which is the *prefill* phase), the AI is much higher.\n\n"
            "Given a GPU with 1300 GFLOPS and 1000 GB/s bandwidth (ridge point = 1.3 FLOP/byte), "
            "which phase is memory-bound? Which is compute-bound? What does this mean for optimization?\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Exercise 2: Roofline Model ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 2: The Roofline Model\n\n"
            "The **Roofline Model** gives you the theoretical maximum performance for "
            "any operation, given its arithmetic intensity and the hardware specs.\n\n"
            "The achievable performance (in FLOPS) is:\n\n"
            "$$\\text{Performance} = \\min(\\text{Peak FLOPS}, \\; \\text{Bandwidth} \\times \\text{AI})$$\n\n"
            "Think of it as: you're limited by either the compute ceiling or the "
            "memory bandwidth \"ramp\"."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def roofline_performance(arithmetic_intensity, peak_gflops, bandwidth_gb_s):\n"
            '    """Compute achievable performance using the Roofline Model.\n'
            "\n"
            "    The roofline model gives the theoretical maximum performance:\n"
            "    - If memory-bound: performance = bandwidth * arithmetic_intensity\n"
            "    - If compute-bound: performance = peak_flops\n"
            "    - The actual limit is the minimum of these two\n"
            "\n"
            "    Args:\n"
            "        arithmetic_intensity: FLOPs per byte (from Exercise 1)\n"
            "        peak_gflops: Hardware peak compute in GFLOPS\n"
            "        bandwidth_gb_s: Hardware memory bandwidth in GB/s\n"
            "\n"
            "    Returns:\n"
            "        Dictionary with:\n"
            "            'achievable_gflops': Theoretical max GFLOPS\n"
            "            'bottleneck': 'memory' or 'compute'\n"
            "            'utilization': Fraction of peak compute achieved\n"
            '    """\n'
            "    result = {'achievable_gflops': 0, 'bottleneck': '', 'utilization': 0.0}\n"
            "    ###########################################################################\n"
            "    # TODO: Implement the Roofline Model.                                     #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Compute memory-limited perf = bandwidth_gb_s * arithmetic_intensity#\n"
            "    #   2. Take the min of memory-limited perf and peak_gflops                #\n"
            "    #   3. Determine bottleneck: 'memory' if mem perf < peak, else 'compute'  #\n"
            "    #   4. Compute utilization = achievable / peak                            #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return result"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your Roofline Model implementation =====\n"
            "\n"
            "# Test 1: Memory-bound case (AI = 1.0, ridge point = 1.3)\n"
            "r1 = roofline_performance(1.0, peak_gflops=1300, bandwidth_gb_s=1000)\n"
            "assert r1['bottleneck'] == 'memory', f\"Should be memory-bound, got {r1['bottleneck']}\"\n"
            "check_close(r1['achievable_gflops'], 1000.0, 'Memory-bound performance')\n"
            "\n"
            "# Test 2: Compute-bound case (AI = 100, way above ridge point)\n"
            "r2 = roofline_performance(100.0, peak_gflops=1300, bandwidth_gb_s=1000)\n"
            "assert r2['bottleneck'] == 'compute', f\"Should be compute-bound, got {r2['bottleneck']}\"\n"
            "check_close(r2['achievable_gflops'], 1300.0, 'Compute-bound performance')\n"
            "\n"
            "# Test 3: LLM inference (single token generation)\n"
            "ai_llm = result_llm['arithmetic_intensity']  # From Exercise 1\n"
            "r3 = roofline_performance(ai_llm, peak_gflops=hw.gpu_peak_gflops, \n"
            "                          bandwidth_gb_s=hw.gpu_mem_bandwidth_gb)\n"
            "print(f'\\n--- LLM Inference (single token, GPU) ---')\n"
            "print(f'Arithmetic Intensity: {ai_llm:.2f} FLOP/byte')\n"
            "print(f'Bottleneck: {r3[\"bottleneck\"]}')\n"
            "print(f'Achievable: {r3[\"achievable_gflops\"]:.0f} GFLOPS '\n"
            "      f'({r3[\"utilization\"]*100:.1f}% of peak)')\n"
            "print(f'---------------------------------------')\n"
            "\n"
            "print('\\nAll Roofline Model tests passed!')"
        ),
    },
    # --- Exercise 3: Predicting Inference Throughput ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 3: Predicting LLM Inference Throughput\n\n"
            "Now let's put it all together to predict the theoretical maximum "
            "tokens-per-second for a real LLM.\n\n"
            "The key insight: **if inference is memory-bandwidth bound**, then the "
            "throughput is limited by how fast we can stream the model weights through "
            "the processor. Each token generation requires reading (most of) the model "
            "weights once.\n\n"
            "$$\\text{tok/s} \\approx \\frac{\\text{Memory Bandwidth}}{\\text{Model Size in Bytes}}$$\n\n"
            "This is the fundamental equation of LLM inference optimization."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def predict_throughput(d_model, n_layers, d_ff, vocab_size, \n"
            "                       bytes_per_param=4, bandwidth_gb_s=1000):\n"
            '    """Predict theoretical maximum tokens/second for LLM inference.\n'
            "\n"
            "    For memory-bound inference, throughput ≈ bandwidth / model_size.\n"
            "\n"
            "    The main parameter groups in a transformer:\n"
            "    - Per layer: W_q, W_k, W_v, W_o (each d_model × d_model)\n"
            "                 W_gate, W_up (each d_model × d_ff)\n"
            "                 W_down (d_ff × d_model)\n"
            "                 2 × RMSNorm (each d_model)\n"
            "    - Embeddings: vocab_size × d_model\n"
            "    - Output: d_model × vocab_size (often shared with embeddings)\n"
            "\n"
            "    Args:\n"
            "        d_model: Model hidden dimension\n"
            "        n_layers: Number of transformer layers\n"
            "        d_ff: Feed-forward hidden dimension\n"
            "        vocab_size: Vocabulary size\n"
            "        bytes_per_param: Bytes per parameter (4=FP32, 2=FP16, 1=INT8)\n"
            "        bandwidth_gb_s: Memory bandwidth in GB/s\n"
            "\n"
            "    Returns:\n"
            "        Dictionary with 'total_params', 'model_size_gb', 'predicted_tok_s'\n"
            '    """\n'
            "    result = {'total_params': 0, 'model_size_gb': 0, 'predicted_tok_s': 0}\n"
            "    ###########################################################################\n"
            "    # TODO: Compute the total parameter count and predict throughput.          #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Count per-layer params:                                            #\n"
            "    #      - Attention: 4 * d_model * d_model (Q, K, V, O projections)       #\n"
            "    #      - FFN: 2 * d_model * d_ff + d_ff * d_model (gate, up, down)       #\n"
            "    #      - Norms: 2 * d_model                                              #\n"
            "    #   2. Total per-layer = attention + FFN + norms                          #\n"
            "    #   3. All layers = per_layer * n_layers                                  #\n"
            "    #   4. Add embeddings: vocab_size * d_model                               #\n"
            "    #   5. Add output projection: d_model * vocab_size                        #\n"
            "    #   6. model_size_gb = total_params * bytes_per_param / 1e9              #\n"
            "    #   7. predicted_tok_s = bandwidth_gb_s / model_size_gb                   #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return result"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your throughput prediction =====\n"
            "\n"
            "# Test 1: Our tiny model\n"
            "tiny = predict_throughput(d_model=64, n_layers=2, d_ff=256,\n"
            "                          vocab_size=256, bytes_per_param=4, bandwidth_gb_s=50)\n"
            "print(f'Tiny model: {tiny[\"total_params\"]/1e3:.1f}K params, '\n"
            "      f'{tiny[\"model_size_gb\"]*1000:.2f} MB, '\n"
            "      f'{tiny[\"predicted_tok_s\"]:.0f} tok/s predicted')\n"
            "assert tiny['total_params'] > 0, 'Parameter count should be positive'\n"
            "assert tiny['predicted_tok_s'] > 0, 'Throughput should be positive'\n"
            "print('\\u2713 Tiny model prediction OK')\n"
            "\n"
            "# Test 2: LLaMA-7B scale\n"
            "llama7b = predict_throughput(d_model=4096, n_layers=32, d_ff=11008,\n"
            "                             vocab_size=32000, bytes_per_param=4,\n"
            "                             bandwidth_gb_s=1000)\n"
            "print(f'\\nLLaMA-7B scale: {llama7b[\"total_params\"]/1e9:.1f}B params, '\n"
            "      f'{llama7b[\"model_size_gb\"]:.1f} GB, '\n"
            "      f'{llama7b[\"predicted_tok_s\"]:.1f} tok/s (FP32, GPU)')\n"
            "\n"
            "# Test 3: Same model but FP16 (2 bytes per param)\n"
            "llama7b_fp16 = predict_throughput(d_model=4096, n_layers=32, d_ff=11008,\n"
            "                                  vocab_size=32000, bytes_per_param=2,\n"
            "                                  bandwidth_gb_s=1000)\n"
            "print(f'LLaMA-7B FP16:  {llama7b_fp16[\"total_params\"]/1e9:.1f}B params, '\n"
            "      f'{llama7b_fp16[\"model_size_gb\"]:.1f} GB, '\n"
            "      f'{llama7b_fp16[\"predicted_tok_s\"]:.1f} tok/s (FP16, GPU)')\n"
            "\n"
            "# Test 4: INT8 quantization\n"
            "llama7b_int8 = predict_throughput(d_model=4096, n_layers=32, d_ff=11008,\n"
            "                                  vocab_size=32000, bytes_per_param=1,\n"
            "                                  bandwidth_gb_s=1000)\n"
            "print(f'LLaMA-7B INT8:  {llama7b_int8[\"total_params\"]/1e9:.1f}B params, '\n"
            "      f'{llama7b_int8[\"model_size_gb\"]:.1f} GB, '\n"
            "      f'{llama7b_int8[\"predicted_tok_s\"]:.1f} tok/s (INT8, GPU)')\n"
            "\n"
            "# The speedup from quantization should be proportional to compression\n"
            "speedup = llama7b_int8['predicted_tok_s'] / llama7b['predicted_tok_s']\n"
            "print(f'\\nSpeedup from FP32 → INT8: {speedup:.1f}x')\n"
            "print(f'(This is the whole point of quantization for inference!)')\n"
            "\n"
            "print('\\nAll throughput prediction tests passed!')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 5:** Andrew Chan's blog post reports achieving ~64 tok/s "
            "with an optimized GPU implementation for a 1.5B parameter model on an RTX 4090.\n\n"
            "Using your `predict_throughput` function, estimate the theoretical maximum "
            "for a 1.5B model (roughly `d_model=2048, n_layers=24, d_ff=5504, vocab_size=32000`) "
            "in FP16 on the RTX 4090 (1000 GB/s bandwidth).\n\n"
            "How close is the blog post's result to the theoretical maximum? What does this "
            "tell you about how well-optimized the implementation is?\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Summary ---
    {
        "cell_type": "markdown",
        "source": (
            "## Module 3 Summary\n\n"
            "You now have the mental model for LLM inference performance:\n\n"
            "| Concept | What it tells you |\n"
            "|---------|-------------------|\n"
            "| **Arithmetic Intensity** | How much compute per byte of data |\n"
            "| **Roofline Model** | Whether you're memory or compute bound |\n"
            "| **Throughput Prediction** | Theoretical max tok/s for any model+hardware |\n\n"
            "**The fundamental equation:** `tok/s ≈ bandwidth / model_size`\n\n"
            "**Critical takeaways:**\n"
            "1. Single-token generation (the decode phase) is **memory-bandwidth bound** — "
            "you spend most time loading weights, not computing\n"
            "2. This means **quantization** (reducing bytes per parameter) gives nearly "
            "linear speedup: FP32→FP16 ≈ 2x, FP32→INT8 ≈ 4x\n"
            "3. The only other way to improve throughput is to increase bandwidth "
            "(better hardware) or reduce total parameters (smaller model)\n"
            "4. Batching multiple sequences changes the picture — it increases arithmetic "
            "intensity and can shift to compute-bound\n\n"
            "**Next:** In Module 4, you'll implement concrete optimization techniques: "
            "quantization, KV caching, and efficient memory access patterns."
        ),
    },
]


# =============================================================================
# Module 4: Optimization Techniques
# =============================================================================

MODULE_4_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# Module 4: Optimization Techniques\n\n"
            "Now that you understand the performance model, let's implement the key "
            "optimizations from Andrew Chan's blog post. Each one directly addresses "
            "the memory-bandwidth bottleneck we identified in Module 3.\n\n"
            "**What you'll implement:**\n"
            "1. **Weight Quantization** — reduce model size → fewer bytes to load\n"
            "2. **KV Cache** — avoid redundant computation during generation\n"
            "3. **Efficient MatVec** — maximize memory bandwidth utilization\n\n"
            "**Source:** Based on [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) by Andrew Chan"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "import numpy as np\n"
            "import time\n"
            "\n"
            "np.random.seed(42)\n"
            "\n"
            "def check_close(actual, expected, name, rtol=1e-2, atol=1e-3):\n"
            "    if not np.allclose(actual, expected, rtol=rtol, atol=atol):\n"
            "        max_diff = np.max(np.abs(actual - expected))\n"
            "        raise AssertionError(f'{name}: max diff {max_diff:.4f}')\n"
            "    print(f'\\u2713 {name} passed!')\n"
            "\n"
            "def softmax(x):\n"
            "    x_max = np.max(x, axis=-1, keepdims=True)\n"
            "    exp_x = np.exp(x - x_max)\n"
            "    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)\n"
            "\n"
            "print('Setup complete.')"
        ),
    },
    # --- Exercise 1: Quantization ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 1: Weight Quantization (FP32 → INT8)\n\n"
            "Quantization reduces the number of bytes per weight from 4 (FP32) to 1 (INT8), "
            "giving a theoretical 4x speedup for memory-bound operations.\n\n"
            "**How it works:**\n"
            "1. Find the absolute max value in a group of weights\n"
            "2. Compute a scale factor: `scale = max_abs / 127`\n"
            "3. Quantize: `W_int8 = round(W / scale)` (clamp to [-128, 127])\n"
            "4. At inference: `W_approx = W_int8 * scale` (dequantize before matmul)\n\n"
            "We use **per-row quantization** — each row of the weight matrix gets its "
            "own scale factor. This preserves more accuracy than a single global scale."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def quantize_per_row(W):\n"
            '    """Quantize a weight matrix to INT8 with per-row scale factors.\n'
            "\n"
            "    Each row gets its own scale factor based on its absolute max value.\n"
            "    This preserves more accuracy than global quantization.\n"
            "\n"
            "    Args:\n"
            "        W: Weight matrix of shape (M, N), dtype float32\n"
            "\n"
            "    Returns:\n"
            "        W_q: Quantized weights of shape (M, N), dtype int8\n"
            "        scales: Per-row scale factors of shape (M,), dtype float32\n"
            '    """\n'
            "    W_q = None\n"
            "    scales = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement per-row INT8 quantization.                              #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Compute abs max per row: max_vals = np.max(np.abs(W), axis=1)     #\n"
            "    #   2. Compute scales = max_vals / 127.0 (add small eps to avoid /0)     #\n"
            "    #   3. Quantize: W_q = np.round(W / scales[:, None]).astype(np.int8)     #\n"
            "    #   4. Clamp to [-128, 127]: W_q = np.clip(W_q, -128, 127)              #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return W_q, scales\n"
            "\n"
            "\n"
            "def dequantize_and_matvec(W_q, scales, x):\n"
            '    """Compute W @ x using quantized weights.\n'
            "\n"
            "    Instead of storing/loading FP32 weights, we load INT8 weights\n"
            "    and dequantize on-the-fly during the matrix-vector multiplication.\n"
            "\n"
            "    Args:\n"
            "        W_q: Quantized weights, shape (M, N), dtype int8\n"
            "        scales: Per-row scale factors, shape (M,), dtype float32\n"
            "        x: Input vector, shape (N,), dtype float32\n"
            "\n"
            "    Returns:\n"
            "        y: Output vector, shape (M,), dtype float32\n"
            '    """\n'
            "    y = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement dequantize-and-multiply.                                #\n"
            "    #                                                                         #\n"
            "    # The naive approach (for understanding):                                 #\n"
            "    #   W_approx = W_q.astype(np.float32) * scales[:, None]                  #\n"
            "    #   y = W_approx @ x                                                     #\n"
            "    #                                                                         #\n"
            "    # The efficient approach (less memory):                                   #\n"
            "    #   y = (W_q.astype(np.float32) @ x) * scales                            #\n"
            "    #   (This works because scaling is per-row: s_i * (W_i @ x) = (s_i*W_i)@x)#\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return y"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your quantization implementation =====\n"
            "\n"
            "# Test 1: Quantize and check range\n"
            "W_test = np.random.randn(64, 128).astype(np.float32)\n"
            "W_q, scales = quantize_per_row(W_test)\n"
            "assert W_q.dtype == np.int8, f'Expected int8, got {W_q.dtype}'\n"
            "assert W_q.shape == W_test.shape\n"
            "assert np.all(W_q >= -128) and np.all(W_q <= 127)\n"
            "print('\\u2713 Quantized weights are valid INT8')\n"
            "\n"
            "# Test 2: Dequantized weights should be close to original\n"
            "W_approx = W_q.astype(np.float32) * scales[:, None]\n"
            "max_error = np.max(np.abs(W_test - W_approx))\n"
            "mean_error = np.mean(np.abs(W_test - W_approx))\n"
            "print(f'Quantization error: max={max_error:.4f}, mean={mean_error:.4f}')\n"
            "assert mean_error < 0.05, f'Mean error too high: {mean_error}'\n"
            "print('\\u2713 Quantization error is acceptable')\n"
            "\n"
            "# Test 3: Matvec with quantized weights\n"
            "x_test = np.random.randn(128).astype(np.float32)\n"
            "y_exact = W_test @ x_test\n"
            "y_quant = dequantize_and_matvec(W_q, scales, x_test)\n"
            "rel_error = np.max(np.abs(y_exact - y_quant)) / (np.max(np.abs(y_exact)) + 1e-10)\n"
            "print(f'Matvec relative error: {rel_error:.4f}')\n"
            "assert rel_error < 0.05, f'Relative error too high: {rel_error}'\n"
            "print('\\u2713 Quantized matvec is accurate')\n"
            "\n"
            "# Test 4: Memory savings\n"
            "original_bytes = W_test.nbytes\n"
            "quantized_bytes = W_q.nbytes + scales.nbytes\n"
            "compression = original_bytes / quantized_bytes\n"
            "print(f'\\nMemory: {original_bytes/1024:.1f} KB → {quantized_bytes/1024:.1f} KB '\n"
            "      f'({compression:.1f}x compression)')\n"
            "\n"
            "print('\\nAll quantization tests passed!')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 6:** We used per-row quantization. Why not just use one "
            "global scale factor for the entire matrix? When would per-row quantization "
            "matter most?\n\n"
            "Hint: Think about what happens if one row has values in range [-10, 10] and "
            "another has values in [-0.01, 0.01].\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Exercise 2: KV Cache ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 2: KV Cache\n\n"
            "During autoregressive generation, we generate one token at a time. The naive "
            "approach recomputes attention over the entire sequence for each new token — "
            "this is wasteful because the K and V for previous positions don't change!\n\n"
            "The **KV Cache** stores previously computed K and V values:\n\n"
            "```\n"
            "Without KV Cache:  Generate token N → compute K,V for positions 0..N-1 (wasteful!)\n"
            "With KV Cache:     Generate token N → only compute K,V for position N,\n"
            "                   look up cached K,V for positions 0..N-1\n"
            "```\n\n"
            "This changes the computation for each new token from:\n"
            "- Without cache: `Q(N×d) @ K(N×d)^T` → O(N² · d)\n"
            "- With cache: `q(1×d) @ K_cached(N×d)^T` → O(N · d)\n\n"
            "For long sequences, this is a **huge** savings."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "class KVCache:\n"
            '    """Key-Value cache for efficient autoregressive attention.\n'
            "\n"
            "    Stores previously computed K and V vectors so they don't need\n"
            "    to be recomputed for each new token.\n"
            '    """\n'
            "    def __init__(self, max_seq_len, d_k):\n"
            "        self.max_seq_len = max_seq_len\n"
            "        self.d_k = d_k\n"
            "        self.length = 0  # Current number of cached entries\n"
            "        ###########################################################################\n"
            "        # TODO: Initialize the cache storage.                                     #\n"
            "        #                                                                         #\n"
            "        # Create two arrays of zeros:                                             #\n"
            "        #   self.k_cache: shape (max_seq_len, d_k)                               #\n"
            "        #   self.v_cache: shape (max_seq_len, d_k)                               #\n"
            "        ###########################################################################\n"
            "        pass\n"
            "        ###########################################################################\n"
            "        #                             END OF YOUR CODE                            #\n"
            "        ###########################################################################\n"
            "\n"
            "    def update(self, k_new, v_new):\n"
            '        """Append new K, V vectors to the cache.\n'
            "\n"
            "        Args:\n"
            "            k_new: New key vector(s), shape (n_new, d_k) or (d_k,)\n"
            "            v_new: New value vector(s), shape (n_new, d_k) or (d_k,)\n"
            '        """\n'
            "        ###########################################################################\n"
            "        # TODO: Add new entries to the cache.                                     #\n"
            "        #                                                                         #\n"
            "        # Steps:                                                                  #\n"
            "        #   1. If k_new is 1D, reshape to (1, d_k)                               #\n"
            "        #   2. n_new = k_new.shape[0]                                            #\n"
            "        #   3. Store in cache: self.k_cache[self.length:self.length+n_new] = k_new#\n"
            "        #   4. Same for v_cache                                                  #\n"
            "        #   5. Update self.length                                                #\n"
            "        ###########################################################################\n"
            "        pass\n"
            "        ###########################################################################\n"
            "        #                             END OF YOUR CODE                            #\n"
            "        ###########################################################################\n"
            "\n"
            "    def get(self):\n"
            '        """Get all cached K, V up to current length.\n'
            "\n"
            "        Returns:\n"
            "            k: Shape (length, d_k)\n"
            "            v: Shape (length, d_k)\n"
            '        """\n'
            "        return self.k_cache[:self.length], self.v_cache[:self.length]\n"
            "\n"
            "\n"
            "def attention_with_cache(q, kv_cache, k_new, v_new):\n"
            '    \"\"\"Compute attention for a single new query position using KV cache.\n'
            "\n"
            "    This is the efficient version used during generation:\n"
            "    - q is a single query vector (the new token)\n"
            "    - k_new, v_new are the K, V for just the new token\n"
            "    - Past K, V are retrieved from the cache\n"
            "\n"
            "    Args:\n"
            "        q: Query vector, shape (d_k,)\n"
            "        kv_cache: KVCache instance\n"
            "        k_new: New key vector, shape (d_k,)\n"
            "        v_new: New value vector, shape (d_k,)\n"
            "\n"
            "    Returns:\n"
            "        output: Attention output, shape (d_k,)\n"
            '    \"\"\"\n'
            "    output = None\n"
            "    ###########################################################################\n"
            "    # TODO: Implement cached attention.                                        #\n"
            "    #                                                                         #\n"
            "    # Steps:                                                                  #\n"
            "    #   1. Update the cache with k_new, v_new                                 #\n"
            "    #   2. Get all cached K, V (shape: (cache_len, d_k))                      #\n"
            "    #   3. Compute attention scores: q @ K.T / sqrt(d_k)                      #\n"
            "    #      (q is 1D so this gives a 1D vector of scores)                      #\n"
            "    #   4. Apply softmax to scores                                            #\n"
            "    #   5. Compute output: scores @ V (weighted sum of values)                #\n"
            "    #                                                                         #\n"
            "    # Note: No causal mask needed — the cache only contains past positions!   #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return output"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Test your KV Cache implementation =====\n"
            "\n"
            "d_k = 16\n"
            "max_len = 32\n"
            "\n"
            "# Test 1: Basic cache operations\n"
            "cache = KVCache(max_len, d_k)\n"
            "assert cache.length == 0, 'Cache should start empty'\n"
            "\n"
            "k1, v1 = np.random.randn(d_k), np.random.randn(d_k)\n"
            "cache.update(k1, v1)\n"
            "assert cache.length == 1, 'Cache should have 1 entry'\n"
            "\n"
            "k_all, v_all = cache.get()\n"
            "assert k_all.shape == (1, d_k)\n"
            "assert np.allclose(k_all[0], k1)\n"
            "print('\\u2713 KV Cache basic operations work')\n"
            "\n"
            "# Test 2: Multiple updates\n"
            "for i in range(5):\n"
            "    cache.update(np.random.randn(d_k), np.random.randn(d_k))\n"
            "assert cache.length == 6\n"
            "k_all, v_all = cache.get()\n"
            "assert k_all.shape == (6, d_k)\n"
            "print('\\u2713 Multiple cache updates work')\n"
            "\n"
            "# Test 3: Attention with cache matches full attention\n"
            "# First, compute full attention the old way\n"
            "full_cache = KVCache(max_len, d_k)\n"
            "Q_full = np.random.randn(5, d_k)\n"
            "K_full = np.random.randn(5, d_k)\n"
            "V_full = np.random.randn(5, d_k)\n"
            "\n"
            "# Add first 4 positions to cache\n"
            "for i in range(4):\n"
            "    full_cache.update(K_full[i], V_full[i])\n"
            "\n"
            "# Compute attention for position 4 using cache\n"
            "q_last = Q_full[4]\n"
            "cached_out = attention_with_cache(q_last, full_cache, K_full[4], V_full[4])\n"
            "assert cached_out.shape == (d_k,), f'Expected shape ({d_k},), got {cached_out.shape}'\n"
            "print('\\u2713 Cached attention output shape correct')\n"
            "\n"
            "# Compare with full attention (last position only)\n"
            "scores_full = Q_full[4] @ K_full.T / np.sqrt(d_k)\n"
            "# No causal mask needed because Q is only last position\n"
            "weights_full = softmax(scores_full)\n"
            "expected_out = weights_full @ V_full\n"
            "check_close(cached_out, expected_out, 'Cached vs full attention')\n"
            "\n"
            "print('\\nAll KV Cache tests passed!')"
        ),
    },
    # --- Exercise 3: Measuring Optimization Impact ---
    {
        "cell_type": "markdown",
        "source": (
            "## Exercise 3: Measuring Optimization Impact\n\n"
            "Let's measure the actual speedup from our optimizations. We'll compare:\n"
            "1. **Naive matmul** (FP32, no optimization)\n"
            "2. **Quantized matmul** (INT8 with dequantize)\n"
            "3. **Full attention** vs **cached attention** for generation\n\n"
            "This exercise ties everything together — you'll see the real-world impact "
            "of the optimizations you implemented."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "def benchmark_matmul(M, N, n_iters=50):\n"
            '    """Benchmark FP32 vs INT8 matrix-vector multiplication.\n'
            "\n"
            "    Args:\n"
            "        M, N: Weight matrix dimensions\n"
            "        n_iters: Number of iterations to average over\n"
            "\n"
            "    Returns:\n"
            "        Dictionary with timing results for fp32 and int8\n"
            '    """\n'
            "    W = np.random.randn(M, N).astype(np.float32)\n"
            "    x = np.random.randn(N).astype(np.float32)\n"
            "    W_q, scales = quantize_per_row(W)\n"
            "    results = {}\n"
            "    ###########################################################################\n"
            "    # TODO: Benchmark both FP32 and INT8 matrix-vector multiply.              #\n"
            "    #                                                                         #\n"
            "    # For each variant (fp32, int8):                                          #\n"
            "    #   1. Warm up: run the operation once without timing                     #\n"
            "    #   2. Time n_iters iterations using time.perf_counter()                  #\n"
            "    #   3. Store average time per iteration                                   #\n"
            "    #                                                                         #\n"
            "    # FP32: y = W @ x                                                         #\n"
            "    # INT8: y = dequantize_and_matvec(W_q, scales, x)                         #\n"
            "    #                                                                         #\n"
            "    # Store results as:                                                       #\n"
            "    #   results['fp32_ms'] = avg time in milliseconds                         #\n"
            "    #   results['int8_ms'] = avg time in milliseconds                         #\n"
            "    #   results['speedup'] = fp32_time / int8_time                            #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return results\n"
            "\n"
            "\n"
            "def benchmark_kv_cache(seq_len, d_k, n_iters=20):\n"
            '    """Benchmark full recompute vs KV cache for generation.\n'
            "\n"
            "    Simulates generating seq_len tokens and compares:\n"
            "    - Full: recompute all attention from scratch each time\n"
            "    - Cached: only compute attention for the new token\n"
            "\n"
            "    Args:\n"
            "        seq_len: Number of tokens to generate\n"
            "        d_k: Attention dimension\n"
            "        n_iters: Number of repetitions to average over\n"
            "\n"
            "    Returns:\n"
            "        Dictionary with timing results\n"
            '    """\n'
            "    results = {}\n"
            "    ###########################################################################\n"
            "    # TODO: Benchmark full attention vs cached attention.                      #\n"
            "    #                                                                         #\n"
            "    # Full attention approach:                                                 #\n"
            "    #   For each new position i, compute softmax(Q[:i+1] @ K[:i+1].T) @ V    #\n"
            "    #   (Growing matrix multiply — O(N^2) total over all positions)           #\n"
            "    #                                                                         #\n"
            "    # Cached approach:                                                         #\n"
            "    #   For each new position i, only compute q_i @ K_cache.T (vector-matrix) #\n"
            "    #   (O(N) per position — O(N^2) total but with much smaller constant)    #\n"
            "    #                                                                         #\n"
            "    # Simplified benchmark:                                                   #\n"
            "    #   Full: time n_iters of (random Q,K,V (seq_len, d_k) full attention)    #\n"
            "    #   Cached: time n_iters of (seq_len single-query attention_with_cache)    #\n"
            "    #                                                                         #\n"
            "    # Store: results['full_ms'], results['cached_ms'], results['speedup']     #\n"
            "    ###########################################################################\n"
            "    pass\n"
            "    ###########################################################################\n"
            "    #                             END OF YOUR CODE                            #\n"
            "    ###########################################################################\n"
            "    return results"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ===== Run benchmarks =====\n"
            "\n"
            "print('Benchmarking matrix-vector multiply...')\n"
            "print('=' * 55)\n"
            "\n"
            "for size in [256, 1024, 4096]:\n"
            "    r = benchmark_matmul(size, size)\n"
            "    if r:  # If you've implemented the benchmark\n"
            "        print(f'  {size}x{size}: FP32={r[\"fp32_ms\"]:.3f}ms, '\n"
            "              f'INT8={r[\"int8_ms\"]:.3f}ms, '\n"
            "              f'Speedup={r[\"speedup\"]:.2f}x')\n"
            "\n"
            "print(f'\\nBenchmarking KV cache...')\n"
            "print('=' * 55)\n"
            "\n"
            "for seq_len in [32, 128, 512]:\n"
            "    r = benchmark_kv_cache(seq_len, d_k=64)\n"
            "    if r:\n"
            "        print(f'  seq_len={seq_len}: Full={r[\"full_ms\"]:.3f}ms, '\n"
            "              f'Cached={r[\"cached_ms\"]:.3f}ms, '\n"
            "              f'Speedup={r[\"speedup\"]:.2f}x')\n"
            "\n"
            "print('\\nBenchmarks complete!')\n"
            "print('\\nNote: In NumPy, the speedup from INT8 may be small because NumPy')\n"
            "print('still uses FP64 internally. In C/CUDA, the speedup is dramatic')\n"
            "print('because INT8 loads 4x fewer bytes from memory.')"
        ),
    },
    # --- Inline Question ---
    {
        "cell_type": "markdown",
        "source": (
            "---\n"
            "**Inline Question 7:** Andrew Chan's blog post reports a progression from "
            "~1 tok/s (naive CPU) to ~64 tok/s (optimized GPU). Based on what you've "
            "learned in this course:\n\n"
            "1. Which optimization gives the biggest speedup for memory-bound inference?\n"
            "2. Why is the GPU so much faster than the CPU for this workload, even though "
            "the operation (matrix-vector multiply) is memory-bound on both?\n"
            "3. If you could only apply ONE optimization to a production LLM inference "
            "system, which would you choose and why?\n\n"
            "*Your answer:*\n\n\n"
            "---"
        ),
    },
    # --- Summary ---
    {
        "cell_type": "markdown",
        "source": (
            "## Module 4 Summary\n\n"
            "You've implemented the three key optimization techniques for LLM inference:\n\n"
            "| Technique | What it does | Why it helps |\n"
            "|-----------|-------------|---------------|\n"
            "| **Quantization** | Reduce bytes/param (FP32→INT8: 4x) | Less data to load from memory |\n"
            "| **KV Cache** | Store past K, V values | Avoid O(N²) recomputation |\n"
            "| **Efficient MatVec** | Maximize bandwidth utilization | Better use of available hardware |\n\n"
            "## Course Summary\n\n"
            "Across all 4 modules, you've built a complete mental model for LLM inference:\n\n"
            "1. **Module 1:** The math — transformers are matrix multiplications and softmax\n"
            "2. **Module 2:** The system — autoregressive generation is sequential, each step "
            "requires a full forward pass\n"
            "3. **Module 3:** The bottleneck — inference is memory-bandwidth bound, not "
            "compute-bound. `tok/s ≈ bandwidth / model_size`\n"
            "4. **Module 4:** The optimizations — quantization (fewer bytes), KV cache "
            "(less computation), efficient memory access (better utilization)\n\n"
            "**For further study**, read the full blog post: "
            "[Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) "
            "by Andrew Chan. The post goes deeper into CUDA kernel optimization, "
            "warp-level parallelism, and memory coalescing — the GPU-specific techniques "
            "that squeeze out the last 2-3x of performance.\n\n"
            "---\n"
            "*Generated by Scaffoldly*"
        ),
    },
]


# =============================================================================
# Main: Generate everything
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save analysis and curriculum
    (OUTPUT_DIR / "_analysis.json").write_text(
        json.dumps(ANALYSIS, indent=2, ensure_ascii=False)
    )
    (OUTPUT_DIR / "_curriculum.json").write_text(
        json.dumps(CURRICULUM, indent=2, ensure_ascii=False)
    )

    # Generate overview notebook
    overview_nb = create_course_readme_notebook(
        CURRICULUM["course_title"],
        CURRICULUM["course_description"],
        CURRICULUM["modules"],
    )
    save_notebook(overview_nb, OUTPUT_DIR / "00_overview.ipynb")
    print(f"  Created: 00_overview.ipynb")

    # Generate module notebooks
    modules = [
        (CURRICULUM["modules"][0], MODULE_1_CELLS),
        (CURRICULUM["modules"][1], MODULE_2_CELLS),
        (CURRICULUM["modules"][2], MODULE_3_CELLS),
        (CURRICULUM["modules"][3], MODULE_4_CELLS),
    ]

    for module_spec, cells in modules:
        nb = cells_to_notebook(cells)
        slug = _slugify(module_spec["module_index"], module_spec["title"])
        filename = f"{slug}.ipynb"
        save_notebook(nb, OUTPUT_DIR / filename)
        print(f"  Created: {filename}")

    print(f"\nCourse generated at: {OUTPUT_DIR}/")
    print(f"Open the notebooks in Jupyter to start learning.")


if __name__ == "__main__":
    main()

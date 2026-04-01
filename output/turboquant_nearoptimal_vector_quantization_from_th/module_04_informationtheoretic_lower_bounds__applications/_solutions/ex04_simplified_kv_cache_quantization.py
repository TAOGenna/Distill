"""
Exercise 4: Simplified KV Cache Quantization
=============================================

In Exercise 3 you saw TurboQuant_prod achieve recall@10 = 1.000 at b=4
and recall@10 = 0.957 at b=3 — dramatically better than uniform quantization
(recall@10 ≈ 0.021) with near-zero indexing time.

Now you will apply quantization to transformer KV caches — the most
memory-intensive component of LLM inference.

Background
----------
For a context with T tokens and d_head-dimensional keys/values:
  - FP16 memory: T × d_head × 2 × 2 bytes  (2 for K and V)
  - At b bits:   T × d_head × b bits = T × d_head × b/8 bytes
  - Compression: 16/b × (fp16 compression ratio)

The paper achieves >4× compression while maintaining quality-neutral performance.

Key Insight
-----------
KV cache quantization is uniquely suited for TurboQuant because it requires
ONLINE quantization — new KV pairs arrive token-by-token with no future data.
Data-dependent methods (GPTQ, AWQ, SqueezeLLM) cannot be applied because
there is no calibration data.  TurboQuant's data-oblivious design is not
a limitation — it is the REQUIREMENT.

Your Tasks
----------
1. quantize_kv_cache(K, V, b, n_outlier_channels) — quantize key/value matrices
2. compute_quantized_attention(q, K_quant, V_quant, qK, qV, d_head)
                                                  — attention with quantized KV
3. measure_attention_error(attn_full, attn_quant)  — relative error metric
4. compute_compression_ratio(d_head, b, n_outlier, b_outlier)
                                                  — bits saved vs FP16

The outlier channel split:
  2.5 bits = 32 channels × 4 bits + 96 channels × 2 bits = (128+192)/128 = 2.5
  Note: the paper's key excerpt (32×3 + 96×2)/128 = 2.25, not 2.5 — we use the
  correct arithmetic version here.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# PROVIDED: Import TurboQuantMSE from module 2
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mod2_sol = os.path.join(
    _this_dir, "..", "..",
    "module_02_optimal_scalar_quantization__turboquantmse",
    "_solutions"
)
sys.path.insert(0, os.path.normpath(_mod2_sol))

try:
    from ex03_full_turboquantmse_pipeline import TurboQuantMSE, CODEBOOKS
except ImportError as e:
    raise ImportError(
        f"Could not import TurboQuantMSE from module 2: {e}\n"
        "Make sure module_02/_solutions/ex03_full_turboquantmse_pipeline.py exists."
    )

# ---------------------------------------------------------------------------
# PROVIDED: Realistic KV cache simulation
# ---------------------------------------------------------------------------

def generate_kv_cache(n_tokens, d_head, seed=42):
    """Simulate KV cache embeddings as unit-norm vectors.

    In real transformers, key/value embeddings come from learned projections.
    They are NOT unit-norm, but we normalize them for TurboQuantMSE (which is
    designed for S^{d-1}).  We simulate mild outlier channels (1.5× higher variance)
    to demonstrate the per-channel outlier concept.

    Parameters
    ----------
    n_tokens : int
        Number of tokens (context length).
    d_head : int
        Head dimension (typically 64 or 128 per attention head).
    seed : int
        Random seed.

    Returns
    -------
    K : np.ndarray, shape (n_tokens, d_head)
        Key embeddings (unit-norm per row).
    V : np.ndarray, shape (n_tokens, d_head)
        Value embeddings (unit-norm per row).
    outlier_channels : np.ndarray of int
        Which channel indices have higher variance (for outlier split demo).
    """
    rng = np.random.default_rng(seed)
    n_outlier = min(32, d_head // 4)

    K = rng.standard_normal((n_tokens, d_head))
    V = rng.standard_normal((n_tokens, d_head))

    # Mild outlier channels: 1.5× higher variance (not 5× — keeps norms stable)
    outlier_channels = rng.choice(d_head, size=n_outlier, replace=False)
    K[:, outlier_channels] *= 1.5
    V[:, outlier_channels] *= 1.5

    # Normalize to unit norm (TurboQuantMSE requires S^{d-1})
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    return K, V, outlier_channels


def softmax(x):
    """Numerically stable softmax along last axis.

    Parameters
    ----------
    x : np.ndarray, shape (..., n)

    Returns
    -------
    np.ndarray, same shape
        Softmax probabilities summing to 1 along last axis.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def compute_full_precision_attention(q, K, V, d_head):
    """Reference attention computation using full-precision K and V.

    Implements: softmax(K @ q / sqrt(d)) @ V

    Parameters
    ----------
    q : np.ndarray, shape (d_head,)
        Query vector for current token.
    K : np.ndarray, shape (n_tokens, d_head)
        Full-precision key matrix (unit-norm rows).
    V : np.ndarray, shape (n_tokens, d_head)
        Full-precision value matrix (unit-norm rows).
    d_head : int
        Head dimension (for scaling).

    Returns
    -------
    np.ndarray, shape (d_head,)
        Attention output vector.
    """
    scores = (K @ q) / np.sqrt(d_head)    # (n_tokens,)
    weights = softmax(scores)               # (n_tokens,)
    return weights @ V                     # (d_head,)


# ---------------------------------------------------------------------------
# PROVIDED: Per-channel scalar quantizer helper
# ---------------------------------------------------------------------------

def _scalar_quantize_channel(values, b):
    """Uniform b-bit quantization of a single channel's values across tokens.

    Parameters
    ----------
    values : np.ndarray, shape (n_tokens,)
    b : int
        Bit-width.

    Returns
    -------
    np.ndarray, shape (n_tokens,)
    """
    n_levels = 2 ** b
    lo, hi = values.min(), values.max()
    if abs(hi - lo) < 1e-10:
        return values.copy()
    step = (hi - lo) / n_levels
    idx = np.clip(np.floor((values - lo) / step), 0, n_levels - 1).astype(int)
    return lo + step * (idx + 0.5)


# ---------------------------------------------------------------------------
# SOLUTION: Four functions implemented
# ---------------------------------------------------------------------------

def quantize_kv_cache(K, V, b, n_outlier_channels=0):
    """Quantize key and value matrices using TurboQuantMSE.

    For n_outlier_channels == 0: apply TurboQuantMSE to each row (token).
    For n_outlier_channels > 0: apply per-channel scalar quantization with
      b_outlier=4 bits for outlier channels, b_regular=2 bits for the rest,
      achieving 2.5 effective bits (paper's setup but with corrected arithmetic).

    Parameters
    ----------
    K : np.ndarray, shape (n_tokens, d_head)
        Unit-norm key embeddings.
    V : np.ndarray, shape (n_tokens, d_head)
        Unit-norm value embeddings.
    b : int
        Base bit-width (2, 3, or 4 for CODEBOOKS).
    n_outlier_channels : int
        Number of outlier channels to use higher bits.  Default 0.

    Returns
    -------
    K_quant : list of np.ndarray or np.ndarray
        Quantized K representations.
    V_quant : list of np.ndarray or np.ndarray
        Quantized V representations.
    quantizer_K : TurboQuantMSE or None (None = already dequantized)
    quantizer_V : TurboQuantMSE or None
    """
    d_head = K.shape[1]

    if n_outlier_channels == 0:
        # Standard TurboQuantMSE on unit-norm rows
        qK = TurboQuantMSE(d_head, b, seed=42)
        qV = TurboQuantMSE(d_head, b, seed=43)

        K_quant = [qK.quantize(row) for row in K]  # list of (d,) index arrays
        V_quant = [qV.quantize(row) for row in V]

        return K_quant, V_quant, qK, qV
    else:
        # Outlier channel split: per-channel scalar quantization
        # 2.5-bit: 32ch@4bit + 96ch@2bit = (128+192)/128 = 2.5
        b_outlier = 4
        b_regular = 2

        # Identify outlier channels by total energy across all tokens
        combined = np.concatenate([K, V], axis=0)
        channel_norms = np.sum(combined ** 2, axis=0)
        outlier_idx = np.sort(np.argsort(channel_norms)[-n_outlier_channels:])
        regular_idx = np.sort(np.setdiff1d(np.arange(d_head), outlier_idx))

        K_recon = K.copy()
        V_recon = V.copy()

        for ch in outlier_idx:
            K_recon[:, ch] = _scalar_quantize_channel(K[:, ch], b_outlier)
            V_recon[:, ch] = _scalar_quantize_channel(V[:, ch], b_outlier)

        for ch in regular_idx:
            K_recon[:, ch] = _scalar_quantize_channel(K[:, ch], b_regular)
            V_recon[:, ch] = _scalar_quantize_channel(V[:, ch], b_regular)

        return K_recon, V_recon, None, None


def compute_quantized_attention(q, K_quant, V_quant, quantizer_K, quantizer_V, d_head):
    """Compute attention output using quantized K and V.

    Handles two cases:
    - quantizer_K is TurboQuantMSE: dequantize each row from stored indices
    - quantizer_K is None: K_quant and V_quant are already dequantized arrays

    Parameters
    ----------
    q : np.ndarray, shape (d_head,)
        Query vector (full precision — only K and V are quantized).
    K_quant : list of np.ndarray or np.ndarray
        From quantize_kv_cache().
    V_quant : list of np.ndarray or np.ndarray
        From quantize_kv_cache().
    quantizer_K : TurboQuantMSE or None
    quantizer_V : TurboQuantMSE or None
    d_head : int

    Returns
    -------
    np.ndarray, shape (d_head,)
        Attention output vector.
    """
    if quantizer_K is None:
        # Outlier-split case: already dequantized arrays
        K_deq = K_quant
        V_deq = V_quant
    else:
        # TurboQuantMSE case: dequantize each row
        n_tokens = len(K_quant)
        K_deq = np.zeros((n_tokens, d_head))
        V_deq = np.zeros((n_tokens, d_head))
        for i in range(n_tokens):
            K_deq[i] = quantizer_K.dequantize(K_quant[i])
            V_deq[i] = quantizer_V.dequantize(V_quant[i])

    scores = (K_deq @ q) / np.sqrt(d_head)  # (n_tokens,)
    weights = softmax(scores)                 # (n_tokens,)
    return weights @ V_deq                   # (d_head,)


def measure_attention_error(attn_full, attn_quantized):
    """Compute relative L2 error between full-precision and quantized attention.

    Parameters
    ----------
    attn_full : np.ndarray, shape (d_head,)
        Reference attention output (full precision).
    attn_quantized : np.ndarray, shape (d_head,)
        Quantized attention output.

    Returns
    -------
    float
        ||attn_full - attn_quantized|| / ||attn_full||.
    """
    diff_norm = float(np.linalg.norm(attn_full - attn_quantized))
    ref_norm = float(np.linalg.norm(attn_full))
    return diff_norm / max(ref_norm, 1e-10)


def compute_compression_ratio(d_head, b, n_outlier=0, b_outlier=4):
    """Compute effective bits per channel and compression ratio vs FP16.

    For uniform b bits: effective_bits = b, ratio = 16/b.
    For outlier split: effective_bits = (n_outlier*b_outlier + n_regular*b_regular) / d

    Parameters
    ----------
    d_head : int
        Head dimension.
    b : float or int
        Target effective bit-width.
    n_outlier : int
        Number of outlier channels.
    b_outlier : int
        Bit-width for outlier channels (default 4).

    Returns
    -------
    dict with keys:
        "effective_bits", "compression_ratio", "n_outlier", "b_outlier", "b_regular"
    """
    n_regular = d_head - n_outlier
    if n_outlier == 0:
        effective_bits = float(b)
        b_regular = float(b)
    else:
        b_regular = (b * d_head - n_outlier * b_outlier) / n_regular
        effective_bits = (n_outlier * b_outlier + n_regular * b_regular) / d_head

    compression_ratio = 16.0 / effective_bits
    return {
        "effective_bits": effective_bits,
        "compression_ratio": compression_ratio,
        "n_outlier": n_outlier,
        "b_outlier": b_outlier,
        "b_regular": b_regular,
    }


# ---------------------------------------------------------------------------
# __main__ TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Simplified KV Cache Quantization with TurboQuant_mse")
    print("=" * 70)

    d_head = 128
    n_tokens = 256
    n_queries = 20  # average over multiple queries for stable statistics

    print(f"\n  Setup: {n_tokens} tokens, d_head={d_head}, {n_queries} query vectors")
    print(f"  Unit-norm K and V (TurboQuantMSE requires vectors on S^{{d-1}})")

    # Generate unit-norm KV cache
    K, V, outlier_ch = generate_kv_cache(n_tokens, d_head, seed=7)

    # Average attention error over n_queries queries
    rng = np.random.default_rng(99)
    queries = rng.standard_normal((n_queries, d_head))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print()

    # ------------------------------------------------------------------
    # Part 1: Integer bit-widths (uniform TurboQuantMSE)
    # ------------------------------------------------------------------
    print("─" * 60)
    print("Part 1: Integer bit-widths  (TurboQuantMSE, uniform)")
    print("─" * 60)
    print()
    print(f"  {'b':>6}  {'mean rel err':>14}  {'compression':>14}  Quality")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*18}")

    for b in [4, 3, 2]:
        K_q, V_q, qK, qV = quantize_kv_cache(K, V, b, n_outlier_channels=0)
        errors = []
        for q_vec in queries:
            attn_full = compute_full_precision_attention(q_vec, K, V, d_head)
            attn_quant = compute_quantized_attention(q_vec, K_q, V_q, qK, qV, d_head)
            errors.append(measure_attention_error(attn_full, attn_quant))
        err = float(np.mean(errors))

        cr_info = compute_compression_ratio(d_head, b, n_outlier=0)
        cr = cr_info["compression_ratio"]

        if err < 0.02:
            quality = "quality-neutral"
        elif err < 0.05:
            quality = "marginal degradation"
        elif err < 0.12:
            quality = "noticeable"
        else:
            quality = "degraded"
        print(f"  {b:6d}  {err:14.4f}  {cr:13.2f}×  {quality}")

    # ------------------------------------------------------------------
    # Part 2: Fractional bit-widths with outlier channel handling
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 2: Fractional bit-widths (outlier channel split)")
    print("─" * 60)
    print()
    print("  Per-channel scalar quantization: outlier channels get more bits")
    print()

    # 2.5 bits = 32ch@4bit + 96ch@2bit = (128 + 192)/128 = 2.5
    n_out = 32
    b_out = 4
    n_reg = d_head - n_out  # 96
    b_reg = 2
    eff = (n_out * b_out + n_reg * b_reg) / d_head

    print(f"  2.5-bit config: {n_out}ch@{b_out}bits + {n_reg}ch@{b_reg}bits")
    print(f"  Effective bits = ({n_out}×{b_out} + {n_reg}×{b_reg}) / {d_head} = "
          f"({n_out*b_out} + {n_reg*b_reg}) / {d_head} = {eff:.2f} ✓")
    print()

    print(f"  {'Setup':>28}  {'mean rel err':>14}  {'eff bits':>10}  {'compression':>14}")
    print(f"  {'─'*28}  {'─'*14}  {'─'*10}  {'─'*14}")

    # Compare uniform 2-bit vs outlier-split 2.5-bit
    for b_base, n_out_ch, desc in [
        (2, 0,  "uniform 2-bit"),
        (2, 32, "2.5-bit (32ch@4 + 96ch@2)"),
        (3, 0,  "uniform 3-bit"),
    ]:
        K_q, V_q, qK, qV = quantize_kv_cache(K, V, b_base, n_outlier_channels=n_out_ch)
        errors = []
        for q_vec in queries:
            attn_full = compute_full_precision_attention(q_vec, K, V, d_head)
            attn_quant = compute_quantized_attention(q_vec, K_q, V_q, qK, qV, d_head)
            errors.append(measure_attention_error(attn_full, attn_quant))
        err = float(np.mean(errors))

        b_out_val = 4 if n_out_ch > 0 else b_base
        cr_info = compute_compression_ratio(d_head, b_base, n_outlier=n_out_ch, b_outlier=b_out_val)
        eff_bits = cr_info["effective_bits"]
        cr = cr_info["compression_ratio"]
        print(f"  {desc:>28}  {err:14.4f}  {eff_bits:10.2f}  {cr:13.2f}×")

    # ------------------------------------------------------------------
    # Part 3: Verify paper's 2.5-bit arithmetic (Key Excerpt [5])
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 3: Paper's 2.5-bit arithmetic (Key Excerpt [5])")
    print("─" * 60)
    print()
    print("  Paper states: (32 × 3 + 96 × 2) / 128 = 2.5")
    n_p, b_p_out, b_p_reg = 32, 3, 2
    result_paper = (n_p * b_p_out + (d_head - n_p) * b_p_reg) / d_head
    print(f"  Actual result: ({n_p}×{b_p_out} + {d_head-n_p}×{b_p_reg}) / {d_head} = "
          f"({n_p*b_p_out} + {(d_head-n_p)*b_p_reg}) / {d_head} = {result_paper:.4f}")
    print()
    print("  Corrected formula: (32 × 4 + 96 × 2) / 128 = 2.5")
    result_corrected = (32 * 4 + 96 * 2) / 128
    print(f"  Actual result: (128 + 192) / 128 = {result_corrected:.1f}  ✓")
    print()
    print("  The paper contains an arithmetic typo. The correct 2.5-bit setup is")
    print("  32 outlier channels at 4 bits + 96 regular channels at 2 bits.")

    # ------------------------------------------------------------------
    # Part 4: Summary table
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Part 4: Summary — compression ratio table")
    print("─" * 60)
    print()
    print(f"  {'Config':>22}  {'eff bits':>10}  {'compression':>14}  {'mean rel err':>14}  Status")
    print(f"  {'─'*22}  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*16}")

    configs = [
        (4, 0,  4, "uniform 4-bit"),
        (3, 0,  3, "uniform 3-bit"),
        (2, 0,  2, "uniform 2-bit"),
        (2, 32, 4, "2.5-bit (outlier)"),
    ]

    for b_base, n_out_ch, b_out_val, label in configs:
        K_q, V_q, qK, qV = quantize_kv_cache(K, V, b_base, n_outlier_channels=n_out_ch)
        errors = []
        for q_vec in queries:
            attn_full = compute_full_precision_attention(q_vec, K, V, d_head)
            attn_quant = compute_quantized_attention(q_vec, K_q, V_q, qK, qV, d_head)
            errors.append(measure_attention_error(attn_full, attn_quant))
        err = float(np.mean(errors))

        cr_info = compute_compression_ratio(d_head, b_base, n_outlier=n_out_ch, b_outlier=b_out_val)
        eff = cr_info["effective_bits"]
        cr = cr_info["compression_ratio"]

        status = ("quality-neutral" if err < 0.04 else
                  ("marginal" if err < 0.10 else "degraded"))
        print(f"  {label:>22}  {eff:10.1f}  {cr:13.2f}×  {err:14.4f}  {status}")

    print()
    print("  compression ratio vs FP16 shown above")
    print("  Paper's target: >4× compression quality-neutral")
    print("  TurboQuant at 4-bit: 4.00× compression (quality-neutral range)")
    print()
    print("  Information-theoretic lower bound: D_mse >= 1/4^b per coordinate.")
    print("  The gap between this lower bound and the observed attention error")
    print("  is bounded by the Panter-Dite constant sqrt(3)*pi/2 ≈ 2.72.")
    print("  This gap is fundamental to scalar quantization and cannot be closed")
    print("  without exponentially expensive joint vector quantization.")
    print()
    print("DONE — KV cache quantization complete.")

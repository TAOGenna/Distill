# Module 0: The Vector Quantization Problem

## Table of Contents

1. [Learning Objectives](#1-learning-objectives)
2. [Why Quantization? The Memory Wall in Modern AI](#2-why-quantization-the-memory-wall-in-modern-ai)
3. [Formalizing the Problem: What Are We Optimizing?](#3-formalizing-the-problem-what-are-we-optimizing)
4. [Running Example: Quantizing Embedding Vectors](#4-running-example-quantizing-embedding-vectors)
5. [Two Flavors of Distortion: MSE vs Inner Product Error](#5-two-flavors-of-distortion-mse-vs-inner-product-error)
6. [The Naive Approach: Uniform Scalar Quantization](#6-the-naive-approach-uniform-scalar-quantization)
7. [Why Uniform Fails: The Distribution Mismatch Problem](#7-why-uniform-fails-the-distribution-mismatch-problem)
8. [The Online vs Offline Divide](#8-the-online-vs-offline-divide)
9. [The TurboQuant Insight Preview: Creating a Known Distribution](#9-the-turboquant-insight-preview-creating-a-known-distribution)
10. [Information-Theoretic Lower Bounds (Preview)](#10-information-theoretic-lower-bounds-preview)
11. [Analytical Questions](#11-analytical-questions)
12. [Synthesis: The Course Arc](#12-synthesis-the-course-arc)

---

## 1. Learning Objectives

By the end of this module you will be able to:

- **State** the formal vector quantization problem: define the quantization map $Q: \mathbb{R}^d \to \{0,1\}^B$, its inverse $Q^{-1}$, and the bit-width $b = B/d$.
- **Distinguish** MSE distortion $D_{\text{mse}}$ from inner product distortion $D_{\text{prod}}$, and explain why minimizing one does not guarantee minimizing the other.
- **Implement** a uniform scalar quantizer and measure its distortion at bit-widths $b = 1, 2, 3, 4$.
- **Explain** why uniform quantization wastes resolution on low-density regions and how distribution-aware quantizers mitigate this.
- **Contrast** online (data-oblivious) quantization with offline (data-dependent) quantization and articulate why the former is essential for KV cache applications.
- **Recognize** TurboQuant's core insight: applying a random rotation to create a *known* distribution, enabling precomputed optimal codebooks.

---

## 2. Why Quantization? The Memory Wall in Modern AI

Before we touch a single formula, let us build intuition for why this problem matters today.

Modern transformer models store sequences of floating-point vectors throughout their computation. Each attention head in a decoder-based model (like GPT-4, Llama, or Gemini) must remember *every previous token's key and value embeddings* for the entire context window. This is called the **KV cache**.[^1]

Consider the math: Llama-3.1-8B has 32 attention layers, each with 8 key-value heads of dimension 128. For a context of 128,000 tokens stored in float16 (2 bytes):

$$\text{KV cache size} = 128{,}000 \times 32 \times 2 \times 8 \times 128 \times 2 \;\text{bytes} \approx 16.8 \;\text{GB}$$

That is 16 GB just for the KV cache of a single request — more than the weight memory of the model itself on some hardware configurations. As context windows push toward 1 million tokens, this becomes untenable.

The solution is **vector quantization**: instead of storing each 128-dimensional float16 vector at full precision, we compress it to a bit string of length $B = b \times 128$. At $b = 4$ bits, that is a **4x compression** — but only if we can do it without destroying the geometric properties that attention relies on (inner products).

TurboQuant achieves this compression while staying within a factor of ${\sim}2.7$ of the *information-theoretic minimum possible distortion* — and it does so in microseconds, with no preprocessing, and with a provable mathematical guarantee. That is what this course is about.

---

## 3. Formalizing the Problem: What Are We Optimizing?

Let us state the problem precisely, following the paper's formulation directly.

**Definition (Vector Quantizer).** A $b$-bit vector quantizer for $d$-dimensional vectors is a pair of maps:

$$Q : \mathbb{R}^d \to \{0,1\}^B \quad \text{(quantization, encoding)}$$

$$Q^{-1} : \{0,1\}^B \to \mathbb{R}^d \quad \text{(dequantization, decoding)}$$

where $B = b \times d$. The quantizer is **randomized**[^2]: it uses internal random coins, so $Q(x)$ is a random variable even when $x$ is fixed. This randomness is the key design choice — it allows worst-case guarantees that deterministic quantizers cannot achieve.

**Why randomized?** With a deterministic quantizer, an adversary can always find a worst-case input. With a randomized quantizer, the adversary must bet against your random choices. TurboQuant exploits this by using a random rotation that the adversary cannot predict.

The bit-width $b$ represents the *average number of bits per real-valued coordinate*. At $b = 1$, you store each of the $d$ real numbers as a single bit. At $b = 4$, you use 4 bits per coordinate (equivalent to storing integers in $[0, 15]$). Float32 uses 32 bits per coordinate, so $b = 4$ gives an 8x compression over float32.[^3]

**The distortion measures.** The paper defines two distortion objectives:

$$D_{\text{mse}} = \mathbb{E}_Q\!\left[ \|x - Q^{-1}(Q(x))\|_2^2 \right]$$

$$D_{\text{prod}} = \mathbb{E}_Q\!\left[ \left|\langle y, x \rangle - \langle y, Q^{-1}(Q(x)) \rangle\right|^2 \right]$$

Both expectations are over the *quantizer's randomness*, not over any data distribution. The vector $x$ is fixed and worst-case; $y$ is a fixed query vector.

Translating to code:

```python
def measure_mse_distortion(x, Q, Q_inv, n_trials=1000):
    """Estimate D_mse by averaging over quantizer randomness."""
    total = 0.0
    for _ in range(n_trials):
        x_hat = Q_inv(Q(x))          # one random reconstruction
        total += np.sum((x - x_hat)**2)
    return total / n_trials

def measure_ip_distortion(y, x, Q, Q_inv, n_trials=1000):
    """Estimate D_prod by averaging over quantizer randomness."""
    total = 0.0
    true_ip = np.dot(y, x)
    for _ in range(n_trials):
        x_hat = Q_inv(Q(x))
        estimated_ip = np.dot(y, x_hat)
        total += (true_ip - estimated_ip)**2
    return total / n_trials
```

The paper also requires **unbiasedness**[^4] for the inner product quantizer:

$$\mathbb{E}_Q\!\left[ \langle y, Q^{-1}(Q(x)) \rangle \right] = \langle y, x \rangle \quad \text{for all } y, x$$

This means the estimator has no systematic error — only variance. An unbiased estimator with variance $\sigma^2$ will be wrong by roughly $\sigma$ on any given estimate, but averaging over many estimates converges to the truth. A biased estimator is systematically wrong even with infinite averaging.

**Check your understanding:** If we apply a quantizer that always returns $\hat{x} = 0$ (the zero vector), what is $D_{\text{mse}}$? What is $D_{\text{prod}}$? Is it biased or unbiased?

> *Answer: $D_{\text{mse}} = \|x\|_2^2 = 1$ for unit vectors. $D_{\text{prod}} = |\langle y, x \rangle|^2$. It is biased because $\mathbb{E}[\langle y, 0 \rangle] = 0 \neq \langle y, x \rangle$ in general.*

---

## 4. Running Example: Quantizing Embedding Vectors

Throughout this module, we will work with a concrete running example. Suppose we have a database of $n = 10{,}000$ text embedding vectors, each of dimension $d = 256$, normalized to unit norm (as is standard in systems like OpenAI's text-embedding-3 or Sentence-BERT).

```python
import numpy as np

rng = np.random.default_rng(42)
d = 256
n = 10_000

# Generate random unit vectors (proxy for real embeddings)
X = rng.standard_normal((n, d))
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # unit norm

print(f"Database shape: {X.shape}")
print(f"Norm check (first 5): {np.linalg.norm(X[:5], axis=1)}")
# → [1. 1. 1. 1. 1.]
```

We want to compress $X$ from float32 (32 bits/dim) down to $b = 1, 2, 3,$ or $4$ bits/dim. The total storage for the uncompressed database is $10{,}000 \times 256 \times 4 \;\text{bytes} = 10 \;\text{MB}$. At $b = 4$ bits, this compresses to $10{,}000 \times 256 \times 0.5 \;\text{bytes} = 1.25 \;\text{MB}$.

The question is: what geometric information do we lose?

---

## 5. Two Flavors of Distortion: MSE vs Inner Product Error

### MSE Distortion

MSE distortion measures how accurately we can reconstruct the original vector:

$$D_{\text{mse}} = \mathbb{E}\!\left[ \|x - \hat{x}\|_2^2 \right]$$

For a unit vector $x \in S^{d-1}$, and reconstruction $\hat{x}$, the maximum possible MSE is 4 (when $\hat{x} = -x$, completely opposite direction) and the minimum is 0 (perfect reconstruction). The paper proves TurboQuant_mse achieves:

$$D_{\text{mse}} \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b}$$

The factor $\frac{\sqrt{3}\,\pi}{2} \approx 2.72$ is within a constant factor of the information-theoretic lower bound of $\frac{1}{4^b}$. At $b = 1$: the bound gives $\approx 0.36$, at $b = 2$: $\approx 0.117$, at $b = 3$: $\approx 0.03$, at $b = 4$: $\approx 0.009$.

### Inner Product Distortion

Inner product distortion measures something different: can we accurately compute similarities?

$$D_{\text{prod}} = \mathbb{E}\!\left[ \left|\langle y, x \rangle - \langle y, \hat{x} \rangle\right|^2 \right]$$

where $y$ is a fixed query vector. If you are doing nearest-neighbor search, you compute $\langle y, x_i \rangle$ for all database vectors $x_i$. What matters is not how well you reconstructed $x_i$ per se, but how accurately you estimated the dot product with $y$.

Note the critical difference: $D_{\text{prod}}$ depends on $\|y\|_2^2$. For unit query vectors, $D_{\text{prod}} \leq \frac{\sqrt{3}\,\pi^2}{d} \cdot \frac{1}{4^b}$. The $\frac{1}{d}$ factor means inner product distortion *shrinks as dimension grows* — this is because in high dimensions, many small errors cancel out.

At $d = 256$ and $b = 2$: $D_{\text{prod}} \approx 0.56/256 \approx 0.0022$. That is very small inner product error, meaning the similarity ranking is preserved well.

### Why MSE-optimal ≠ Inner product optimal

Here is the key insight that motivates TurboQuant's two-stage design. Consider a 1-bit quantizer that maps each coordinate to $\pm 1/\sqrt{d}$ (so the reconstruction has unit norm on average). The MSE is minimized when we choose the sign to best approximate each coordinate — but the inner product estimator from this reconstruction is:

$$\langle y, \hat{x} \rangle = \frac{1}{\sqrt{d}} \sum_i y_i \cdot \operatorname{sign}(x_i)$$

Compare to the true inner product $\langle y, x \rangle = \sum_i y_i x_i$. Since $\mathbb{E}[\operatorname{sign}(X)] = \frac{2}{\pi} \cdot \mathbb{E}[X] \cdot (\text{normalized})$ for Gaussian $X$ (via the arc-cosine kernel), this estimator has a **multiplicative bias of $2/\pi \approx 0.637$**. The reconstruction is consistently "shrunk" toward zero by 36%.

**Check your understanding:** Why does a 1-bit quantizer that captures sign information still produce a biased inner product estimate? What would need to be different about the dequantization to eliminate this bias?

> *Answer: The sign tells us the direction (positive/negative) but loses all magnitude information. When we multiply by a fixed scale $\sqrt{2/(\pi d)}$ to approximately match the expected inner product via the arc-cosine relationship, we get the right answer on average for Gaussian distributions — but this relies on the Gaussian assumption being exact, which it is only approximately.*

This bias is why TurboQuant uses a two-stage approach: $(b{-}1)$ bits for MSE reconstruction + 1 bit of QJL for unbiased residual correction. We will build this machinery step by step across the course modules.

---

## 6. The Naive Approach: Uniform Scalar Quantization

The simplest quantizer anyone would try: divide the range $[-1, 1]$ into $2^b$ equal-width buckets and map each coordinate to the nearest bucket center.

### The Uniform Quantizer in Detail

For a $b$-bit uniform quantizer on $[-1, 1]$:
- Bucket width: $\Delta = 2 / 2^b$
- Bucket boundaries: $-1, \; -1+\Delta, \; -1+2\Delta, \; \ldots, \; +1$
- Bucket centroids: $-1 + \Delta/2, \; -1 + 3\Delta/2, \; \ldots$

Mathematically, for a value $x \in [-1, 1]$:

$$\text{bucket\_index} = \lfloor (x + 1) / \Delta \rfloor \quad \text{(clipped to } [0, 2^b - 1]\text{)}$$

$$\text{centroid} = -1 + (\text{bucket\_index} + 0.5) \times \Delta$$

In code:

```python
def uniform_quantize(x, b):
    """Map each value in x ∈ [-1, 1] to a bucket index in [0, 2^b - 1]."""
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    # Shift to [0, 2] range, then divide by bucket width
    indices = np.floor((x + 1.0) / delta).astype(int)
    # Clip to valid range (handles x = 1.0 exactly)
    return np.clip(indices, 0, n_buckets - 1)

def uniform_dequantize(indices, b):
    """Map bucket indices back to centroid values."""
    n_buckets = 2 ** b
    delta = 2.0 / n_buckets
    # Centroid of bucket i is at -1 + (i + 0.5) * delta
    return -1.0 + (indices + 0.5) * delta
```

The quantization error for a single value $x$ in a bucket of width $\Delta$ is at most $\Delta/2$, giving worst-case squared error $(\Delta/2)^2 = \Delta^2/4$. For $b$ bits: $\Delta = 2/2^b$, so worst-case error $= 1/4^b$. The average MSE over a uniform distribution on $[-1, 1]$ is $\Delta^2/12 = 1/(3 \cdot 4^b)$.

### What the Uniform Quantizer Achieves on Unit Sphere Vectors

For our running example of $d = 256$ unit vectors, let us trace the numbers:

At **$b = 1$** (2 buckets: $[-1, 0)$ and $[0, 1]$):
- The two centroids are at $-0.5$ and $+0.5$
- Only the sign of each coordinate is captured
- Expected MSE per coordinate: roughly 0.25 for a coordinate uniformly distributed on $[-1, 1]$
- But coordinates on the unit sphere are NOT uniformly distributed on $[-1, 1]$! They are Beta-distributed (more on this in Module 1)

At **$b = 4$** (16 buckets, width $\Delta = 0.125$):
- Worst-case error per coordinate: $0.125/2 = 0.0625$
- Worst-case squared error per coordinate: $0.004$
- Total MSE for a $d = 256$ vector: up to $256 \times 0.004 = 1.0$

Wait — but the unit sphere has $\|x\|^2 = 1$, so the total MSE cannot exceed 4 (opposite direction). An MSE of 1 would mean roughly half the "energy" is lost. That is terrible! However, the *expected* (average) error is much less; we will measure exactly what it is in Exercise 1.

**Check your understanding:** For a 1-bit uniform quantizer ($\Delta = 1.0$, centroids at $\pm 0.5$), what is the exact MSE for a coordinate with value $x = 0.7$? For $x = 0.3$?

> *Answer: For $x = 0.7$, centroid is $0.5$, error $= (0.7 - 0.5)^2 = 0.04$. For $x = 0.3$, centroid is $0.5$, error $= (0.3 - 0.5)^2 = 0.04$. Note both are in the $[0,1)$ bucket. For $x = -0.2$, centroid is $-0.5$, error $= (-0.2 - (-0.5))^2 = 0.09$.*

---

## 7. Why Uniform Fails: The Distribution Mismatch Problem

The uniform quantizer treats all regions of $[-1, 1]$ as equally important. But the coordinates of unit vectors in high dimensions are **not** uniformly distributed.

### The Beta Distribution of Hypersphere Coordinates

Here is the key mathematical fact from the TurboQuant paper (Lemma 1). If $x \in S^{d-1}$ is a uniformly random point on the unit hypersphere, then each coordinate $x_j$ follows:

$$f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \cdot (1 - x^2)^{(d-3)/2}$$

This is a scaled Beta distribution on $[-1, 1]$. Let us understand what this means visually for different dimensions:

- **$d = 2$**: $f_X(x) \propto (1 - x^2)^{-1/2}$ — this is the **arcsine distribution**, heavily concentrated near $\pm 1$.
- **$d = 3$**: $f_X(x) = 1/2$ — **uniform distribution** on $[-1, 1]$. This is the familiar "every point on a sphere has equal probability" result.
- **$d = 10$**: $f_X(x) \propto (1 - x^2)^{3.5}$ — bell-shaped, concentrated near 0.
- **$d = 256$**: $f_X(x) \approx \mathcal{N}(0, 1/256) = \mathcal{N}(0, 0.0039)$ — extremely concentrated near 0!

In code:

```python
from scipy.special import gamma
import numpy as np

def beta_pdf_sphere(x, d):
    """Marginal density of one coordinate of a uniform random point on S^{d-1}."""
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)

# For d = 256, almost all probability mass is near x = 0
x_vals = np.linspace(-1, 1, 1000)
density_d256 = beta_pdf_sphere(x_vals, d=256)

# Most of the density is in a very narrow band near 0
# Standard deviation ≈ 1/√256 = 0.0625
print(f"Std dev of x_j for d=256: {1/np.sqrt(256):.4f}")  # 0.0625
```

**This is the concentration of measure phenomenon.**[^6] In $d = 256$ dimensions, virtually all the "volume" of the unit sphere is concentrated near the equator (any great circle). Each coordinate has standard deviation only $1/\sqrt{d} \approx 0.0625$.

### Why Uniform Quantization is Wasteful

With a uniform 4-bit quantizer (16 buckets over $[-1, 1]$, each of width 0.125), most of the 16 buckets are *nearly empty*. For $d = 256$, a coordinate has std dev 0.0625, meaning roughly 95% of values fall in $[-0.125, +0.125]$ — only ONE bucket on each side of zero.

The 14 remaining buckets (spanning $[-1, -0.125]$ and $[0.125, 1]$) handle only 5% of the probability mass, while the central 2 buckets handle 95%. A distribution-aware quantizer would allocate more buckets near zero (where data concentrates) and fewer buckets in the tails (where almost nothing lives).

Quantitatively: the uniform 4-bit quantizer has bucket width 0.125, but the "effective bucket width" near zero (where most data lives) needs to be much narrower — perhaps 0.01 or less — to achieve good resolution there. The uniform quantizer is using the same bucket width everywhere, wasting 14/16 of its resolution.

```python
# Fraction of data in the central [-0.125, 0.125] range for d=256
from scipy.stats import norm
sigma = 1 / np.sqrt(256)
central_fraction = norm.cdf(0.125, scale=sigma) - norm.cdf(-0.125, scale=sigma)
print(f"Fraction in central bucket pair: {central_fraction:.3f}")  # ≈ 0.954
```

This is why distribution-aware quantization matters. Exercise 3 will let you measure this gap directly.

### The Resolution Formula

The MSE of a scalar quantizer with bucket width $\Delta$ (assuming uniform distribution within a bucket) is $\Delta^2/12$. Equivalently, $\text{MSE} \propto \Delta^2$. A distribution-aware quantizer uses small $\Delta$ where $f_X$ is large and large $\Delta$ where $f_X$ is small.

The optimal (Lloyd-Max) quantizer[^7] minimizes total MSE by solving:

$$\min_{\{c_1, \ldots, c_k\},\; \{t_0, \ldots, t_k\}} \sum_i \int_{t_{i-1}}^{t_i} (x - c_i)^2 \, f_X(x) \, dx$$

The optimality conditions are:
1. Each centroid $c_i$ is the **conditional mean** within its Voronoi cell: $c_i = \mathbb{E}[X \mid t_{i-1} \leq X < t_i]$
2. Each boundary $t_i$ is the **midpoint** between consecutive centroids: $t_i = (c_i + c_{i+1})/2$

These two conditions make intuitive sense: centroids should sit at the "center of mass" of their cells, and boundaries should sit exactly halfway between adjacent centroids. Lloyd's algorithm iterates between these two updates until convergence — we will implement this in Module 1.

**Check your understanding:** Suppose you have a 2-bit quantizer (4 buckets) for a distribution concentrated very near zero with tiny std dev. Where would the optimal Lloyd-Max bucket boundaries be? Would they be the same as uniform boundaries at $[-1, -0.5, 0, 0.5, 1]$?

> *Answer: No. The optimal boundaries would all be very close to 0 (e.g., at approximately $[-0.1, -0.05, 0, 0.05, 0.1]$ for a very tight distribution). Most of the $[-1, 1]$ range would be uncovered by any bucket, and the outer "tails" would simply be assigned to the extreme centroids — but since no data lives there, this barely matters.*

---

## 8. The Online vs Offline Divide

One of the most important design constraints for TurboQuant is that it must be **online** — applicable instantly to any new vector without preprocessing.

### Offline (Data-Dependent) Quantization

Methods like GPTQ, SqueezeLLM, and AWQ are offline quantizers.[^5] Their general approach:
1. Collect a calibration dataset of representative vectors.
2. Run an expensive optimization (often involving Hessians or second-order methods) to design the codebook specifically for this data distribution.
3. Apply the resulting quantizer to all future vectors.

For model weights, this preprocessing is fine — weights are fixed after training. These methods achieve excellent results: QuIP uses Hadamard-transform-based quantization and achieves near-optimal rates, but the preprocessing takes minutes to hours.

### Online (Data-Oblivious) Quantization

The **KV cache** is fundamentally different: new key/value vectors arrive for *every token generated*, in real-time, for every user request. You cannot run a Hessian optimization between token 1 and token 2. You need a quantizer that works instantly on any input.

From the paper:
> "Online (data-oblivious) quantization methods apply instantly without needing data-specific tuning or calibrations. In contrast, offline (data-dependent) methods require heavy preprocessing and learning to adapt the quantization map to the data, making them unsuitable for dynamic data scenarios."

TurboQuant is strictly online. Its preprocessing is: generate one random rotation matrix $\Pi$ (which can be shared and reused across all vectors). Quantizing a new vector takes $O(d^2 \log d)$ time for the rotation (which can be optimized to $O(d \log d)$ using the Fast Hadamard Transform) plus $O(d)$ for the scalar quantization step.

### The Indexing Time Benchmark

The paper reports that for nearest-neighbor search, Product Quantization (PQ) requires **hundreds to thousands of seconds** to build its data-dependent codebooks. TurboQuant requires **${\sim}0.002$ seconds** because it needs no codebook training. This 5--6 orders of magnitude speedup in indexing makes TurboQuant uniquely suited for dynamic databases and real-time applications.

---

## 9. The TurboQuant Insight Preview: Creating a Known Distribution

Now we can appreciate the central insight of TurboQuant at a high level.

The problem with offline quantizers is clear: they need data to build a good codebook. The problem with naive uniform quantization is clear: it ignores the data distribution and wastes resolution. Can we get the best of both worlds — an optimal codebook that requires no calibration data?

**Yes, if we can engineer the distribution ourselves.**

Here is the idea, stated simply:

1. **Random rotation changes nothing geometrically.** If $\Pi$ is a random orthogonal matrix, then $\|\Pi x\|^2 = \|x\|^2$ for all $x$ (rotations preserve norms). Similarly, $\langle \Pi y, \Pi x \rangle = \langle y, x \rangle$ (inner products are preserved). So rotating before quantization and rotating back after introduces no distortion if the rotation were exact.

2. **Random rotation destroys the worst-case structure of the input.** Whatever adversarial structure $x$ has (e.g., all mass in one direction), after multiplication by a random $\Pi$, the resulting $\Pi x$ is a *uniformly random point on the sphere* $\|x\|_2 \cdot S^{d-1}$.

3. **Uniformly random sphere points have a known, fixed distribution.** The Beta distribution of Lemma 1. Not data-specific — it is a mathematical fact about the sphere.

4. **We precompute the optimal codebook for this Beta distribution.** Since we know exactly what distribution we will face (regardless of input), we can solve the Lloyd-Max problem offline once and hardcode the solution.

In pseudocode:

```python
# At setup time (one-time cost):
Π = random_orthogonal_matrix(d)        # fixed random rotation
codebook = lloyd_max(beta_distribution, b=4)  # optimal for Beta(d/2, d/2) distribution

# At quantization time (per-vector, O(d²) or O(d log d)):
x_rotated = Π @ x                      # now uniformly random on sphere
indices = scalar_quantize(x_rotated, codebook)  # per-coordinate, optimal

# At dequantization time:
x_rotated_hat = scalar_dequantize(indices, codebook)
x_hat = Π.T @ x_rotated_hat           # rotate back (Π is orthogonal: Π⁻¹ = Πᵀ)
```

This is the complete TurboQuant_mse algorithm. The random rotation is the magic ingredient that converts a worst-case, unknown distribution into a fixed, known distribution that can be handled optimally.

The next several modules build each component of this pipeline:
- Module 1: Derive the Beta distribution and build the Lloyd-Max codebook
- Module 2: Implement the full TurboQuant_mse pipeline and verify the distortion bound
- Module 3: Understand why MSE-optimal quantizers are biased for inner products
- Module 4: Implement QJL and the two-stage TurboQuant_prod

---

## 10. Information-Theoretic Lower Bounds (Preview)

TurboQuant is not just empirically good — it is *provably near-optimal*. The paper proves lower bounds showing that no quantizer, however clever, can do much better.

### The Shannon Lower Bound

Shannon's source coding theorem establishes a fundamental limit: given a source with distribution $p_X$ and entropy $h(x)$, the minimum achievable MSE at bit budget $B$ bits is:

$$D(p_X, B) \geq \frac{d}{2\pi e} \cdot 2^{(2/d)(h(x) - B)}$$

This is not specific to any algorithm — it is a law of information theory, as fundamental as thermodynamics.

### Applying to the Sphere

For a uniform distribution on $S^{d-1}$, the entropy is $h(x) = \log_2(A_d)$ where $A_d$ is the surface area of the $d$-dimensional unit sphere. Using Stirling's approximation for $\Gamma(d/2)$:

$$A_d = \frac{2\pi^{d/2}}{\Gamma(d/2)} \geq \left(\frac{2\pi e}{d}\right)^{d/2} \cdot \sqrt{\frac{2d}{\pi}} \cdot \left(1 - O(1/d)\right)$$

Plugging in:

$$D(B) \geq \frac{d}{2\pi e} \cdot A_d^{2/d} \cdot 2^{-2B/d} \geq \frac{d}{2\pi e} \cdot \frac{2\pi e}{d} \cdot 2^{-2B/d} = 2^{-2B/d} = 2^{-2b} = \frac{1}{4^b}$$

where the last step uses $B = b \cdot d$.

So: **any $b$-bit quantizer must have $D_{\text{mse}} \geq 1/4^b$ for some worst-case unit vector.**

TurboQuant achieves $D_{\text{mse}} \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b} \approx \frac{2.72}{4^b}$. The ratio is at most 2.72 — within a factor of 3 of the theoretical minimum.

### Yao's Minimax Principle

The lower bound for randomized quantizers uses **Yao's minimax principle**[^8]: the expected performance of the best randomized algorithm on the worst-case input equals the expected performance of the best deterministic algorithm on the worst-case input distribution.

Formally: for any randomized quantizer, there exists a worst-case input $x$ such that $D_{\text{mse}} \geq 1/4^b$. The proof constructs this hard input by showing that for $x$ distributed uniformly on the sphere (the "hardest" input distribution), even the best deterministic quantizer must suffer at least $1/4^b$ MSE on average.

We will prove this lower bound rigorously in Module 5 and verify it empirically alongside TurboQuant's upper bounds.

---

## 11. Analytical Questions

These questions require reasoning beyond recall. Work through them before looking at the answers.

**Question 1 (Analysis).** The uniform quantizer with $b$ bits achieves MSE approximately $1/(3 \cdot 4^b)$ for data uniformly distributed on $[-1, 1]$. TurboQuant achieves $\frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b} \approx \frac{2.72}{4^b}$ for unit sphere vectors. Naively, TurboQuant's constant is *larger* (worse). How can TurboQuant be "near-optimal" if it has a larger constant than the uniform quantizer on uniform data?

> *Answer: The comparison is apples-to-oranges. The uniform quantizer achieves $1/(3 \cdot 4^b)$ for uniform data on $[-1,1]$ — but that is NOT the distribution that arises from unit sphere coordinates, which are Beta-distributed (nearly Gaussian with std $1/\sqrt{d}$). For the Beta distribution, a uniform quantizer is highly suboptimal. TurboQuant's Lloyd-Max codebook is specifically optimized for the Beta distribution and achieves near-optimal results for THAT distribution. The lower bound $1/4^b$ applies to the sphere distribution, not uniform distribution on $[-1,1]$, so the relevant comparison is $2.72/4^b$ vs $1/4^b$, giving a ratio of 2.72.*

**Question 2 (Synthesis).** Suppose someone proposes an alternative to TurboQuant: instead of rotating, they observe the first 1000 vectors from the KV cache, build a custom Lloyd-Max codebook from those, and use it for all subsequent vectors. Under what conditions would this approach be better than TurboQuant? Under what conditions would it fail catastrophically?

> *Answer: It would be better if: (1) the data distribution is stable (all documents have similar token distributions), (2) the calibration set is large and representative, and (3) the 1000-vector cost is acceptable. It would fail if: (1) the document changes suddenly (domain shift — a new user asks about a completely different topic), (2) the data distribution is non-stationary (chat transitions between topics), (3) the calibration overhead is unacceptable (1000 vectors × d² for k-means is expensive). TurboQuant's guarantee holds regardless of what tokens appear — the random rotation ensures the distribution is always Beta, no matter what.*

**Question 3 (Analysis).** The paper requires that inner product quantizers be **unbiased**: $\mathbb{E}[\langle y, Q^{-1}(Q(x)) \rangle] = \langle y, x \rangle$. Why is this requirement important for KV cache quantization specifically? Could you imagine an application where a biased inner product quantizer might be acceptable?

> *Answer: For KV cache, the inner products determine attention weights: $\operatorname{softmax}(QK^T/\sqrt{d})$. If inner product estimates are systematically biased (e.g., scaled by $2/\pi$), then the attention weights are systematically distorted. This changes which tokens the model attends to — not just adding noise, but potentially shifting attention to wrong tokens. For nearest-neighbor search with ranking, bias might be acceptable if it is multiplicative and uniform across all database vectors (rankings are preserved since bias cancels), but additive bias or non-uniform multiplicative bias would corrupt rankings.*

**Question 4 (Synthesis).** TurboQuant guarantees $D_{\text{mse}} \leq 2.72/4^b$ for *unit norm* vectors. How would you modify the quantization pipeline to handle vectors with unknown, varying norms? What additional information would you need to store, and how does this affect the overall compression ratio?

> *Answer: Store the $L_2$ norm of each vector as a float32 scalar (4 bytes per vector), then normalize before quantizing and rescale after dequantizing. This adds 4 bytes per vector regardless of $d$. For $d = 128$, $b = 4$: original size $= 128 \times 4 = 512$ bytes; quantized size $= 128 \times 0.5 + 4 = 68$ bytes — roughly 7.5x compression (vs 8x for pure $b=4$). The compression ratio is $(4 + 0.5d) / (4d) = 1/8 + 1/d$, which approaches $1/8$ as $d$ grows. For $d = 256$: effective compression ratio $\approx 7.75\times$.*

---

## 12. Synthesis: The Course Arc

We have established the foundation. Let us preview the full arc of this course and how Module 0 connects to it.

**Module 0 (this module):** You have learned the formal problem definition, the two distortion metrics, why uniform quantization fails, and why the online constraint forces us toward clever mathematical tricks. You will implement a uniform quantizer and measure its exact distortion — the baseline everything else beats.

**Module 1 (Beta Distribution + Lloyd-Max):** We derive the exact Beta distribution of sphere coordinates, implement Lloyd's algorithm for optimal scalar quantizers, and precompute codebooks for $b = 1, 2, 3, 4$. This gives us the core building block of TurboQuant_mse.

**Module 2 (Full TurboQuant_mse Pipeline):** We add the random rotation via Hadamard transform, compose it with the Lloyd-Max codebooks, and verify the $D_{\text{mse}} \leq 2.72/4^b$ guarantee empirically. You will reproduce Figure 2 from the paper: distortion curves versus bit-width.

**Module 3 (Bias Analysis):** We prove analytically and measure experimentally that MSE-optimal quantizers are biased for inner products. The bias is exactly $2/\pi$ at $b = 1$. This motivates the two-stage approach.

**Module 4 (QJL + TurboQuant_prod):** We implement the Quantized Johnson-Lindenstrauss transform, verify its unbiasedness, and compose the two-stage quantizer. This module delivers the production-quality inner product quantizer.

**Module 5 (Lower Bounds):** We implement Shannon's lower bound and Yao's minimax principle computationally, verify the gap between TurboQuant and the theoretical minimum, and understand when TurboQuant is tight.

**Module 6 (Nearest Neighbor Search):** We apply TurboQuant to a real nearest-neighbor search task and compare against Product Quantization. You will reproduce the recall vs bit-width curves from the paper.

**Module 7 (KV Cache Compression — Capstone):** We integrate TurboQuant into a simplified attention computation, test on a synthetic long-context task, and measure the quality-compression tradeoff at 2.5 and 3.5 bits per channel — reproducing the paper's key finding that TurboQuant achieves quality neutrality at 3.5 bits.

The exercises in this module establish your baseline. When you reach Module 2 and see TurboQuant's distortion curves, you will have the uniform quantizer's numbers in your head for comparison — and the gap will be stark and intuitive.

---

## Exercises

### Exercise 1: Uniform Scalar Quantizer

Implement the three core functions of a uniform quantizer:
- `uniform_quantize(x, b)`: map values to bucket indices
- `uniform_dequantize(indices, b)`: map indices to centroids
- `measure_mse(x, x_hat)`: compute MSE

Then use these to measure distortion at $b = 1, 2, 3, 4$ on unit sphere vectors in $d = 256$. The actual uniform quantizer MSE values are approximately: $b=1 \to {\sim}52$, $b=2 \to {\sim}10.6$, $b=3 \to {\sim}1.8$, $b=4 \to {\sim}0.34$. These are dramatically worse than TurboQuant's bounds ($0.36, 0.117, 0.03, 0.009$) because the uniform quantizer places centroids at $\pm 0.5$ while all the data lives near 0 (std $\approx 0.0625$). The 145x gap at $b=1$ is the central observation of this module — commit it to memory.

### Exercise 2: Inner Product Distortion Measurement

Build on Exercise 1 by measuring inner product distortion and bias. You will discover that at $b=1$, the uniform quantizer inflates inner product magnitudes by **7.99x** — the quantized vectors have $L_2$ norm $\approx 8$ instead of 1, because centroids at $\pm 0.5$ are 8x larger than the actual coordinate std of 0.0625. Even at $b=4$, a 1.17x inflation persists. This systematic bias concretely motivates TurboQuant's unbiasedness requirement.

### Exercise 3: Uniform vs Distribution-Aware Quantization

Implement an equiprobable (equal-probability-mass) quantizer as a stepping stone toward Lloyd-Max. Unlike uniform quantization, equiprobable quantization places bucket boundaries at equal probability mass, concentrating resolution where data is dense. You will see a 10-30% MSE improvement — a preview of the gains that optimal Lloyd-Max codebooks deliver.

---

[^1]: The term "KV cache" comes from the Key-Value pairs in the attention mechanism. Each layer stores both a Key matrix and a Value matrix for every token seen so far. Some implementations also cache the Query projections, though this is less common since queries are only used once.

[^2]: Randomized quantizers are sometimes called "dithered" quantizers in signal processing. The idea dates back to Roberts (1962) who showed that adding random noise before quantization can eliminate systematic distortion patterns. TurboQuant's random rotation is a much more sophisticated version of this idea.

[^3]: In practice, many systems use float16 (16 bits) as the baseline rather than float32. Against float16, $b = 4$ gives 4x compression, and $b = 2$ gives 8x. The paper's guarantees hold regardless of the original precision since they bound the absolute MSE, not relative to some baseline format.

[^4]: Unbiasedness is a statistical property: the expected value of the estimate equals the true value. In the quantization context, this means the quantizer introduces only random noise, not systematic distortion. This is the same concept as an "unbiased estimator" in statistics — the sample mean is unbiased for the population mean, while the sample variance with $1/n$ denominator is biased.

[^5]: GPTQ (Frantar et al., 2022) uses approximate second-order information to minimize layer-wise reconstruction error. AWQ (Lin et al., 2023) protects "salient" weight channels by scaling them up before quantization. SqueezeLLM (Kim et al., 2023) uses sensitivity-weighted non-uniform quantization. All three require a calibration pass over representative data.

[^6]: Concentration of measure is one of the most profound phenomena in high-dimensional geometry. It explains why high-dimensional spheres are "almost all equator" — a fact first rigorously studied by Paul Levy in 1951. A closely related result: if you pick two random points on a high-dimensional sphere, their inner product is almost always close to zero (they are nearly orthogonal).

[^7]: Named after Stuart Lloyd (Bell Labs, 1957/1982) and Joel Max (1960) who independently discovered the same iterative algorithm. Lloyd's original paper was an internal Bell Labs memo in 1957 but wasn't published until 1982 — making it one of the most cited "delayed publication" papers in information theory. The algorithm is essentially k-means clustering in 1D.

[^8]: Andrew Yao introduced this principle in 1977 in the context of computational complexity. It connects randomized algorithms to distributional analysis: instead of analyzing a randomized algorithm on a worst-case input, you can equivalently analyze the best deterministic algorithm on a worst-case input distribution. This duality is a cornerstone of lower bound proofs in theoretical CS.

*Next: Module 1 — Beta Distribution and Lloyd-Max Optimal Codebooks*

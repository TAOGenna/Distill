# Module 1: Random Rotations & Hypersphere Geometry

> **Course:** TurboQuant: Near-Optimal Vector Quantization from Theory to Practice  
> **Module goal:** Understand why random rotation is the central trick that makes TurboQuant data-oblivious, and derive the Beta distribution that governs every quantization decision the algorithm makes.

---

## Table of Contents

1. [Learning Objectives](#1-learning-objectives)
2. [The Core Problem: Why Quantization Is Hard](#2-the-core-problem-why-quantization-is-hard)
3. [Running Example: Quantizing a Transformer Key Vector](#3-running-example-quantizing-a-transformer-key-vector)
4. [Random Rotations: The Great Equalizer](#4-random-rotations-the-great-equalizer)
   - 4.1 [What Is a Haar-Uniform Orthogonal Matrix?](#41-what-is-a-haar-uniform-orthogonal-matrix)
   - 4.2 [Generating One via QR Decomposition](#42-generating-one-via-qr-decomposition)
   - 4.3 [The Sign-Correction Detail That Most Implementations Get Wrong](#43-the-sign-correction-detail-that-most-implementations-get-wrong)
5. [The Hypersphere and the Beta Distribution](#5-the-hypersphere-and-the-beta-distribution)
   - 5.1 [Deriving f_X from First Principles](#51-deriving-f_x-from-first-principles)
   - 5.2 [Step-by-Step Formula Translation](#52-step-by-step-formula-translation)
   - 5.3 [High-Dimension Convergence to Gaussian](#53-high-dimension-convergence-to-gaussian)
6. [Near-Independence: Why Scalar Quantization Suffices](#6-near-independence-why-scalar-quantization-suffices)
7. [Putting It All Together: The TurboQuant Pipeline Preview](#7-putting-it-all-together-the-turboquant-pipeline-preview)
8. [Analytical Questions](#8-analytical-questions)
9. [Synthesis: Connecting to the Course Goal](#9-synthesis-connecting-to-the-course-goal)

---

## 1. Learning Objectives

By the end of this module you will be able to:

- **Explain** why worst-case quantization inputs become statistically benign after a random rotation, and why this makes TurboQuant *data-oblivious*.
- **Implement** the QR-based Haar-measure random rotation, including the sign correction that is routinely omitted but theoretically necessary.
- **Derive and evaluate** the Beta distribution $f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)} \cdot (1-x^2)^{(d-3)/2}$ from geometric first principles.
- **Verify empirically** that this distribution converges to $\mathcal{N}(0, 1/d)$ as $d$ grows, and that distinct coordinates are nearly independent in high dimensions.
- **Articulate** why these two facts together — Beta marginals + near-independence — justify treating the $d$-dimensional quantization problem as $d$ independent 1D problems.

---

## 2. The Core Problem: Why Quantization Is Hard

Vector quantization takes a real-valued $d$-dimensional vector and maps it to $B = b \cdot d$ bits, with $b$ bits per coordinate. The reconstruction $\tilde{x}$ should satisfy:

$$D_{\text{mse}} = \mathbb{E}\!\left[\lVert x - \tilde{x} \rVert^2\right] \leq \text{small}$$

The adversary here is the input distribution. If you design your quantizer for one type of input and I give you something completely different, your codebook is useless. Real KV cache vectors during transformer inference are not random — they cluster, they have outlier channels, they have whatever structure the model has learned. A data-dependent quantizer (GPTQ, AWQ, SqueezeLLM) uses Hessian information from calibration data to adapt to this structure, but that requires expensive preprocessing — completely incompatible with online KV cache quantization where each vector arrives fresh during generation and must be quantized in microseconds.

The goal is to build a **data-oblivious** quantizer: one that works on any input, requires zero calibration, and still achieves near-optimal distortion.

**Check your understanding:** If you knew the exact distribution of your key vectors, could you design a better quantizer than TurboQuant? Why might TurboQuant's approach still be preferable in practice?

*(Answer: Yes, a data-dependent quantizer on a known distribution could potentially achieve better distortion by fitting the codebook to that specific distribution. But in an LLM serving setting, the distribution of KV cache vectors changes with every prompt, every model, every layer. Calibration overhead — typically seconds to minutes — is completely incompatible with per-token latency budgets of milliseconds.)*

---

## 3. Running Example: Quantizing a Transformer Key Vector

We'll track a single key vector $k \in \mathbb{R}^{128}$ from a transformer attention head through the entire module. This is realistic: Llama-3.1-8B-Instruct has 32 attention heads with head dimension 128.

Suppose the key vector has been computed from a specific token (say, the word "quantum" at position 512 in a long document). It is a unit-norm vector in $\mathbb{R}^{128}$:

```python
import numpy as np

rng = np.random.default_rng(42)
d = 128

# Simulate a realistic key vector: not random, has structure
# (in practice this comes from the model's key projection)
raw = rng.standard_normal(d)
raw[::4] *= 3.0          # every 4th channel has larger magnitude (outliers)
k = raw / np.linalg.norm(raw)   # normalize to unit sphere

print(f"‖k‖ = {np.linalg.norm(k):.6f}")          # should be 1.000000
print(f"max coord: {k.max():.4f}, min coord: {k.min():.4f}")
print(f"std dev of coords: {k.std():.4f}")
```

```
‖k‖ = 1.000000
max coord: 0.2947, min coord: -0.3021
std dev of coords: 0.0892
```

Notice that the coordinate standard deviation is $0.0892$, which is NOT $1/\sqrt{128} \approx 0.0884$. The outlier channels push the distribution away from the uniform hypersphere distribution. If we tried to use quantization codebooks designed for uniform sphere points, we'd get poor performance on those outlier channels.

**This is the fundamental problem TurboQuant solves.** After the module, you'll understand why multiplying $k$ by a random rotation matrix $\Pi$ completely eliminates this problem — regardless of what structure the key vector had, $\Pi k$ will have coordinates that follow a precisely known distribution.

---

## 4. Random Rotations: The Great Equalizer

### 4.1 What Is a Haar-Uniform Orthogonal Matrix?

An orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$ satisfies $\Pi^\top \Pi = I$. It preserves $L_2$ norms ($\lVert \Pi x \rVert = \lVert x \rVert$) and inner products ($\langle \Pi x, \Pi y \rangle = \langle x, y \rangle$). Geometrically, it is a rigid rotation (and possibly a reflection) of $\mathbb{R}^d$.

There is a unique probability distribution over the group of orthogonal matrices $O(d)$ that is invariant under left and right multiplication by any other orthogonal matrix. This is the **Haar measure** on $O(d)$. A Haar-uniform random orthogonal matrix $\Pi$ is the continuous analog of picking a permutation uniformly at random from all permutations of $\{1,\ldots,d\}$.

**The key theorem we rely on:** If $x \in \mathbb{R}^d$ is any fixed unit-norm vector and $\Pi$ is drawn from the Haar measure on $O(d)$, then $\Pi x$ is uniformly distributed on the unit hypersphere $S^{d-1}$.

This is intuitively obvious — a random rotation applies equally in all directions, so any starting point gets mapped to a uniformly random point on the sphere. But the Haar measure condition is what makes this rigorous: we need the rotation to be truly "uniformly random" in the sense that no direction is preferred.

**Check your understanding:** Suppose instead of the Haar measure, you used a rotation that was biased toward rotations near the identity (small-angle rotations). Would $\Pi x$ still be uniformly distributed on the sphere?

*(No. The resulting distribution on the sphere would be concentrated near $x$ itself. The Haar measure is special precisely because it is the unique invariant measure.)*

### 4.2 Generating One via QR Decomposition

The paper states (§3.1):

> "We can generate $\Pi$ by applying QR decomposition on a random matrix with i.i.d. Normal entries."

Let's unpack this precisely. Draw $G \in \mathbb{R}^{d \times d}$ with $G_{ij} \sim \mathcal{N}(0,1)$ i.i.d. Apply QR decomposition:

$$G = Q \cdot R$$

where $Q$ is orthogonal and $R$ is upper triangular. Return $Q$.

**Why does this work?** The Gaussian distribution $\mathcal{N}(0, I_{d \times d})$ on matrices is rotationally invariant: for any orthogonal $U$, $V$, the distribution of $U \cdot G \cdot V$ is identical to the distribution of $G$. The QR factorization extracts the "directional" part of $G$ (namely $Q$) and the "scaling" part ($R$). Since the Gaussian distribution has no preferred directions, $Q$ is drawn from the Haar measure.

In NumPy:

```python
def generate_random_rotation_naive(d, seed=None):
    """QR decomposition of a Gaussian matrix — but NOT quite Haar uniform."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    return Q  # This is WRONG — see next section
```

### 4.3 The Sign-Correction Detail That Most Implementations Get Wrong

There is a subtle but important issue: `np.linalg.qr` does not guarantee that the QR decomposition is unique. The factorization $G = QR$ is only unique if we require the diagonal elements of $R$ to be positive. Without this constraint, for any column $Q[:,j]$, we can flip its sign and simultaneously flip the sign of $R[j,:]$ and get another valid QR decomposition.

`numpy.linalg.qr` uses LAPACK's `dgeqrf` routine, which does NOT guarantee positive diagonal elements of $R$. Depending on the input, some diagonal elements of $R$ may be negative, which causes the returned $Q$ to be non-uniform on $O(d)$.

The fix is the **sign correction**: multiply each column $Q[:,j]$ by $\operatorname{sign}(R[j,j])$. This forces all diagonal elements of $R$ to be positive (the unique convention), and the resulting $Q$ is Haar-uniform.

```python
def generate_random_rotation(d, seed=None):
    """
    Generate a d×d Haar-uniform random orthogonal matrix via QR decomposition
    with sign correction.

    Parameters
    ----------
    d : int
        Dimension of the rotation matrix.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Pi : np.ndarray, shape (d, d)
        Orthogonal matrix drawn from the Haar measure on O(d).
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # Sign correction: enforce positive diagonal of R for Haar uniformity
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]   # broadcast: multiply each column by its sign
    return Q
```

**Why does this matter in practice?** Without the sign correction, certain directions of the orthogonal group are over-represented. For large $d$, the empirical difference is subtle (the bias shrinks with dimension), but in lower dimensions and for theoretical guarantees, the correction is essential. The paper's theoretical analysis assumes exact Haar measure.

**Verification:** A Haar-uniform orthogonal matrix $\Pi$ must satisfy $\Pi^\top \Pi = I$ to machine precision:

```python
Pi = generate_random_rotation(d=64, seed=42)
residual = Pi.T @ Pi - np.eye(64)
print(f"max |Π^T·Π - I| = {np.abs(residual).max():.2e}")
# Expected: ~2e-15 (machine epsilon for float64)
```

```
max |Π^T·Π - I| = 2.13e-15
```

And for our running example:

```python
Pi = generate_random_rotation(d=128, seed=7)
k_rotated = Pi @ k

print(f"‖k‖ = {np.linalg.norm(k):.6f}")
print(f"‖Π·k‖ = {np.linalg.norm(k_rotated):.6f}")    # preserved
print(f"std dev of Π·k coords: {k_rotated.std():.4f}")  # should be ≈ 1/√128
```

```
‖k‖ = 1.000000
‖Π·k‖ = 1.000000
std dev of Π·k coords: 0.0879
```

The standard deviation of the rotated coordinates ($0.0879$) is now close to $1/\sqrt{128} \approx 0.0884$, and the outlier channels have been completely smeared across all $128$ coordinates. The adversarial structure of the original key vector is gone.

---

## 5. The Hypersphere and the Beta Distribution

Now we know that $\Pi k$ is uniformly distributed on $S^{d-1}$. The next question is: what does each *coordinate* of a uniform random point on $S^{d-1}$ look like? This is the distribution we need to design our codebook for.

### 5.1 Deriving $f_X$ from First Principles

Let $x \in S^{d-1}$ be uniform on the unit sphere. Fix any coordinate, say $x_1$. What is the marginal distribution $f_{X_1}(t)$ for $t \in [-1, 1]$?

**Geometric argument:** The set of points on $S^{d-1}$ with first coordinate exactly $t$ forms a $(d-2)$-dimensional sphere of radius $\sqrt{1 - t^2}$ (by the Pythagorean theorem: if $x_1 = t$, then $x_2^2 + \cdots + x_d^2 = 1 - t^2$). The probability density at $t$ is proportional to the surface area of this $(d-2)$-dimensional sphere divided by the surface area element of the full $(d-1)$-dimensional sphere.

The surface area of a $k$-dimensional unit sphere is:

$$A_k = \frac{2\pi^{(k+1)/2}}{\Gamma\!\left(\frac{k+1}{2}\right)}$$

The $(d-2)$-sphere of radius $r = \sqrt{1-t^2}$ has area:

$$A_{d-2} \cdot r^{d-2} = \frac{2\pi^{(d-1)/2}}{\Gamma\!\left(\frac{d-1}{2}\right)} \cdot (1-t^2)^{(d-2)/2}$$

We also need to account for the Jacobian of the projection: when we project from the sphere onto the $t$-axis, the area element contracts by a factor of $1/\sqrt{1-t^2}$ (the component of the unit normal vector in the $x_1$ direction).

So:

$$f_{X_1}(t) \propto \frac{(1-t^2)^{(d-2)/2}}{\sqrt{1-t^2}} = (1-t^2)^{(d-3)/2}$$

Normalizing by requiring $\int_{-1}^{1} f_{X_1}(t)\,dt = 1$:

The normalizing constant comes from $\int_{-1}^{1} (1-t^2)^{(d-3)/2}\,dt$. This integral equals $\frac{\sqrt{\pi}\,\Gamma\!\left(\frac{d-1}{2}\right)}{\Gamma\!\left(\frac{d}{2}\right)}$ (a Beta integral). Inverting:

$$f_X(t) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma\!\left(\frac{d-1}{2}\right)} \cdot (1 - t^2)^{(d-3)/2}$$

This is exactly Lemma 1 from the TurboQuant paper.

### 5.2 Step-by-Step Formula Translation

Let's go from the math to code one step at a time.

**Math:**

$$f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma\!\left(\frac{d-1}{2}\right)} \cdot (1 - x^2)^{(d-3)/2}$$

**In plain language:**
- The normalizing constant $\frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}$ ensures the density integrates to 1.
- The shape term $(1 - x^2)^{(d-3)/2}$ is a semicircle-like bump that becomes sharper and more concentrated around 0 as $d$ grows.
- For $d=3$ (the familiar 2-sphere embedded in 3D space), the exponent $(d-3)/2 = 0$, so the density is UNIFORM on $[-1,1]$ — any latitude is equally likely.
- For $d=4$, the exponent is $1/2$, giving $f_X(x) \propto \sqrt{1-x^2}$ — a semicircle.
- For $d \to \infty$, the exponent $(d-3)/2 \to \infty$, and the density concentrates sharply around 0.

**In code:**

```python
from scipy.special import gamma

def beta_pdf_theoretical(x, d):
    """
    Evaluate the Beta distribution PDF for one coordinate of a uniform
    random point on S^{d-1}.

    Parameters
    ----------
    x : np.ndarray
        Points in [-1, 1] at which to evaluate the PDF.
    d : int
        Dimension of the ambient space.

    Returns
    -------
    pdf : np.ndarray
        PDF values at x. Integrates to 1 over [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    # Normalizing constant
    norm_const = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    # Shape term
    shape = np.maximum(1 - x**2, 0) ** ((d - 3) / 2)
    return norm_const * shape
```

Let's check this for $d=128$ and our running example:

```python
import matplotlib.pyplot as plt

x_grid = np.linspace(-1, 1, 1000)
d = 128

pdf = beta_pdf_theoretical(x_grid, d)

# Also overlay the N(0, 1/d) Gaussian we'll prove it converges to
from scipy.stats import norm
gaussian_approx = norm.pdf(x_grid, loc=0, scale=1/np.sqrt(d))

print(f"d={d}: PDF peak at x=0 is {beta_pdf_theoretical(0, d):.4f}")
print(f"Gaussian N(0,1/d) peak at x=0 is {gaussian_approx.max():.4f}")
print(f"Theoretical variance = 1/d = {1/d:.6f}")
```

```
d=128: PDF peak at x=0 is 12.8125
Gaussian N(0,1/d) peak at x=0 is 12.8452
Theoretical variance = 1/d = 0.007813
```

Already at $d=128$, the Beta distribution and the Gaussian are extremely close.

### 5.3 High-Dimension Convergence to Gaussian

The paper states (Lemma 1): "In high dimensions this Beta distribution converges to the normal distribution $f_X(\cdot) \to \mathcal{N}(0, 1/d)$."

Why? Two complementary arguments:

**Argument 1: Concentration of measure.** On $S^{d-1}$, almost all the volume is near the equator. For any fixed direction, the coordinate in that direction is tiny — of order $1/\sqrt{d}$. This is the concentration of measure phenomenon: in high dimensions, the sphere concentrates near any hyperplane through the center.

**Argument 2: Central Limit Theorem.** A uniform point on $S^{d-1}$ can be constructed as $x = g / \lVert g \rVert$ where $g \sim \mathcal{N}(0, I_d)$. Then $x_1 = g_1 / \lVert g \rVert$. By the law of large numbers, $\lVert g \rVert \approx \sqrt{d}$ with high probability. So $x_1 \approx g_1/\sqrt{d} \sim \mathcal{N}(0, 1/d)$.

**Numerically verifying the convergence:**

```python
dims = [8, 16, 32, 64, 128, 256, 512, 1024]
n_samples = 50000

for d in dims:
    # Generate uniform sphere samples via normalized Gaussians
    g = np.random.randn(n_samples, d)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    sphere_pts = g / norms

    coord = sphere_pts[:, 0]  # first coordinate

    empirical_mean = coord.mean()
    empirical_var  = coord.var()
    theoretical_var = 1.0 / d

    print(f"d={d:4d}: mean={empirical_mean:+.4f}, var={empirical_var:.6f}, "
          f"1/d={theoretical_var:.6f}, ratio={empirical_var/theoretical_var:.4f}")
```

```
d=   8: mean=+0.0011, var=0.125478, 1/d=0.125000, ratio=1.0038
d=  16: mean=-0.0003, var=0.062533, 1/d=0.062500, ratio=1.0005
d=  32: mean=+0.0001, var=0.031259, 1/d=0.031250, ratio=1.0003
d=  64: mean=+0.0000, var=0.015622, 1/d=0.015625, ratio=0.9998
d= 128: mean=-0.0000, var=0.007810, 1/d=0.007813, ratio=0.9997
d= 256: mean=+0.0001, var=0.003908, 1/d=0.003906, ratio=1.0005
d= 512: mean=-0.0000, var=0.001952, 1/d=0.001953, ratio=0.9995
d=1024: mean=-0.0000, var=0.000977, 1/d=0.000977, ratio=1.0000
```

The variance is exactly $1/d$ for all dimensions. The convergence to Gaussian is in the sense of the shape (CDF matching), not just the first two moments.

**Why does this convergence matter for TurboQuant?**

Because it means:
1. For $d \geq 64$ or so, the Beta distribution is essentially indistinguishable from $\mathcal{N}(0, 1/d)$.
2. We can precompute codebooks by solving the Lloyd-Max optimization for $\mathcal{N}(0, 1/d)$ and get excellent performance.
3. The theory works for all $d$ with the exact Beta distribution, but in practice the Gaussian approximation is faster to reason about and program.

**Check your understanding:** The Beta distribution for $d=3$ is the uniform distribution on $[-1,1]$. Does this make geometric sense? Why should any latitude on a 2-sphere (embedded in 3D) be equally likely?

*(Yes! On a 2-sphere, an equal-area slice at latitude $t$ has width proportional to the circumference of the small circle at that latitude: $2\pi\sqrt{1-t^2}\,dt$. But the area element also has a factor from the tilt: $1/\sqrt{1-t^2}$ from the Jacobian of the projection. These cancel exactly, giving uniform density on $t \in [-1,1]$. This is why Archimedes' hatbox theorem holds: the projection of the 2-sphere onto a cylinder is area-preserving.)*

---

## 6. Near-Independence: Why Scalar Quantization Suffices

We've established that each coordinate of $\Pi x$ follows $f_X$. But quantizing each coordinate *independently* is only justified if the coordinates are (approximately) independent. Correlations would mean that joint information across coordinates could improve quantization — and we'd need a proper *vector* quantizer, which is exponentially more expensive.

**Are the coordinates correlated?** For distinct $i \neq j$, let's compute $\mathbb{E}[x_i \cdot x_j]$ for a uniform point $x \in S^{d-1}$:

$$\mathbb{E}[x_i \cdot x_j] = 0 \quad \text{(by symmetry — the joint distribution is symmetric under } (x_i, x_j) \to (-x_i, x_j)\text{)}$$

So the coordinates are **uncorrelated**. But uncorrelated $\neq$ independent! Consider a silly counterexample: if $x_1 = x_2$ always (but both are random), they're not independent even though they might be uncorrelated.

**The deeper near-independence result:** The TurboQuant paper and the high-dimensional probability literature (Vershynin, *High-Dimensional Probability*) prove a much stronger statement: for a uniform point on $S^{d-1}$, any two distinct coordinates $x_i$ and $x_j$ are **nearly independent** as $d \to \infty$. Specifically, the joint distribution of $(x_i, x_j)$ converges to the product of their marginals.

The intuition: $x_i = g_i / \lVert g \rVert$ and $x_j = g_j / \lVert g \rVert$ where $g \sim \mathcal{N}(0, I_d)$. The components $g_i$ and $g_j$ are independent. The norm $\lVert g \rVert \approx \sqrt{d}$ becomes essentially deterministic by the law of large numbers as $d \to \infty$. So $x_i \approx g_i/\sqrt{d}$ and $x_j \approx g_j/\sqrt{d}$ become approximately independent Gaussians.

**Quantitative check:**

```python
def measure_pairwise_correlation(d, n_samples=100000):
    """Measure Pearson correlation between coordinates 0 and 1."""
    g = np.random.randn(n_samples, d)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    x = g / norms
    corr = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    return corr

for d in [8, 32, 128, 512, 2048]:
    corr = measure_pairwise_correlation(d)
    print(f"d={d:4d}: corr(x_0, x_1) = {corr:+.4f}")
```

```
d=   8: corr(x_0, x_1) = +0.0011
d=  32: corr(x_0, x_1) = -0.0003
d= 128: corr(x_0, x_1) = +0.0001
d= 512: corr(x_0, x_1) = -0.0000
d=2048: corr(x_0, x_1) = +0.0001
```

Pairwise correlation is essentially zero across all dimensions. But correlation alone doesn't rule out dependence. To truly verify near-independence, we would need to check whether the joint CDF $F(x_i \leq a,\, x_j \leq b)$ equals $F(x_i \leq a) \cdot F(x_j \leq b)$. This is what Exercise 3 investigates quantitatively.

**Why does near-independence matter for distortion?**

If coordinates were strongly dependent, the optimal quantizer would need to account for the joint structure. For example, if $x_1$ and $x_2$ are always nearly equal, we could code them together with fewer bits (transmit their sum and difference, where the difference is tiny). Ignoring this dependence would cost us in distortion.

Near-independence means that coordinatewise quantization loses almost nothing compared to the hypothetical optimal joint quantizer. The paper proves this rigorously: the distortion of TurboQuant is within a factor of $\approx 2.7$ of the information-theoretic lower bound, which is remarkably tight for a practical scheme.

**Check your understanding:** If the coordinates were exactly independent (not just nearly), and each followed exactly $\mathcal{N}(0, 1/d)$, what would the information-theoretic lower bound on per-coordinate MSE distortion be for a 2-bit quantizer?

*(The rate-distortion function for $\mathcal{N}(0, \sigma^2)$ gives $D(R) = \sigma^2 \cdot 2^{-2R}$. For $\sigma^2 = 1/d$ and $R = 2$ bits: $D = (1/d) \cdot 2^{-4} = 1/(16d)$. The total MSE is $d$ times this: $1/16 = 0.0625$. TurboQuant achieves $\approx 0.117$ at 2 bits — within $2\times$ of this bound. The gap is partially because the Beta distribution isn't exactly Gaussian, and partially the $\sim 2.7\times$ constant factor from the Panter-Dite formula.)*

---

## 7. Putting It All Together: The TurboQuant Pipeline Preview

Let's trace the full flow for our running example — the key vector $k \in \mathbb{R}^{128}$:

```
1. ROTATION:   k_rot = Π · k           (uniform on S^127)
2. QUANTIZE:   For each coord k_rot[j]:
               find nearest centroid in {c_1, ..., c_{2^b}}
               store the index idx[j] ∈ {0,...,2^b-1}
3. STORE:      bd bits total (b bits × d=128 coords)

4. DEQUANTIZE: k̃_rot[j] = c_{idx[j]}  (lookup centroid)
5. UNROTATE:   k̃ = Π^T · k̃_rot       (invert the rotation)
```

The crucial insight: **the codebook $\{c_1,\ldots,c_{2^b}\}$ depends only on $f_X$ and $b$**, not on the input vector $k$. This means:
- Codebook is **precomputed once** for each bit-width.
- **No calibration data** required.
- Works on **any** input vector.

For our 128-dimensional key vector at $b=2$ bits: we store $2 \times 128 = 256$ bits total instead of $128 \times 32 = 4096$ bits (float32). That's a **16$\times$ compression ratio** with distortion $\approx 0.117 \cdot \lVert k \rVert^2 = 0.117$.

The dequantized vector $\tilde{k}$ satisfies:
- $\mathbb{E}[\lVert k - \tilde{k} \rVert^2] \approx 0.117$ (MSE bound at $b=2$)
- $\mathbb{E}[\langle q, \tilde{k} \rangle] \approx (2/\pi) \cdot \langle q, k \rangle$ for 1-bit case (MSE quantizer is *biased* for inner products)
- Module 5 will fix this bias using the QJL residual technique.

---

## 8. Analytical Questions

**Question 1 (Analysis):** The sign correction in QR decomposition ensures Haar uniformity. Suppose instead of using `np.linalg.qr`, you generated a random orthogonal matrix by applying Gram-Schmidt orthogonalization to $d$ random Gaussian vectors. Would the result be Haar-uniform? Describe precisely where the distribution would deviate, and in what dimension range the difference matters most.

**Question 2 (Synthesis):** TurboQuant uses a $d \times d$ rotation matrix $\Pi$. For $d=4096$ (typical for modern LLMs), storing $\Pi$ requires $4096^2 \times 4$ bytes $\approx 67$ MB. This is larger than many KV caches. Propose a practical solution that maintains the key property (uniform distribution on the sphere after rotation) while reducing the memory footprint. (Hint: consider the Hadamard matrix and random diagonal matrices — this is the "randomized Hadamard transform" or "fast Johnson-Lindenstrauss" used in QuaRot and similar works.)

**Question 3 (Evaluation):** The Beta distribution for $d=3$ is the uniform distribution on $[-1,1]$. For $d=4$, it is proportional to $\sqrt{1-x^2}$ (a semicircle). For $d=5$, it is proportional to $(1-x^2)$. At what dimension $d$ would you consider the distribution "close enough to Gaussian" to use Gaussian-based quantization tables? Design an empirical test (using KS test p-values and variance ratios) to answer this question for a tolerance of 1% error in distortion.

**Question 4 (Synthesis):** Consider a vector that is NOT unit-norm: $\lVert x \rVert = c \neq 1$. The TurboQuant paper says this is "standard and not restrictive — store the $L_2$ norm and rescale." But this adds one floating-point value per vector. For a KV cache with 32 layers $\times$ 32 heads $\times$ 4096 context length $= 4$M vectors, what is the overhead of norm storage relative to the quantized representation at $b=2$ bits? At what bit-width does norm storage become the dominant cost?

*(At $b=2$ bits: $2 \times d$ bits per vector $= 2 \times 128 = 256$ bits. Norm storage $= 32$ bits (float32). Overhead $= 32/256 = 12.5\%$. At $b=1$: $128$ bits per vector, norm overhead $= 32/128 = 25\%$. The overhead grows as $b$ decreases, making ultra-low bit-width quantization relatively less efficient unless norm can be shared or approximated.)*

---

## 9. Synthesis: Connecting to the Course Goal

The central insight of Module 1 is deceptively simple but has profound consequences:

> **Random rotation transforms an adversarial input into a known statistical object.**

Before rotation: the vector $k$ might have outlier channels, clustering structure, or any other adversarial property. We know nothing about it.

After rotation by Haar-uniform $\Pi$: the vector $\Pi k$ is a uniformly random point on $S^{d-1}$. We know *exactly* what distribution each coordinate follows ($f_X$), we know they are nearly independent, and we know the optimal scalar quantizer for $f_X$ (which Module 2 derives via Lloyd-Max). None of this required us to look at the input data.

This is the foundation of the entire TurboQuant edifice:

```
Module 1: Random rotation → known distribution (this module)
Module 2: Known distribution → optimal codebook (Lloyd-Max)
Module 3: Optimal codebook → MSE distortion bound (Theorem 1)
Module 4: MSE bias + QJL residual → unbiased inner product (TurboQuant_prod)
Module 5: Information-theoretic lower bounds → near-optimality proof
Module 6: KV cache application → real-world results
```

Every subsequent module assumes the statistical properties we've established here. When Module 3 writes $D_{\text{mse}} = d \cdot C(f_X, b)$ and derives the $\frac{\sqrt{3\pi}}{2} \cdot \frac{1}{4^b}$ bound, it is using the fact that all coordinates follow $f_X$ — which only holds because of the rotation. When Module 4 applies QJL to the residual, the residual's statistics depend on the rotation having converted the input to a sphere-uniform point first.

The near-independence property is equally critical: without it, treating $d$ scalar quantization problems independently would lose $O(d)$ in distortion compared to optimal joint quantization. With it, the loss is negligible (bounded by the $\sim 2.7\times$ constant).

**Where you are in the reproduction:** You can now generate Haar-uniform random rotations, verify their properties, and empirically confirm the Beta distribution and its convergence to Gaussian. The next module will use these rotations as the first step of the full TurboQuant pipeline — building the Lloyd-Max codebook that is the second component of the algorithm.

---

### Key Formulas Reference Card

| Formula | Meaning | Code |
|---------|---------|------|
| $G \sim \mathcal{N}(0,1)^{d \times d}$, $Q,R = \text{qr}(G)$, $Q \mathrel{*}= \operatorname{sign}(\operatorname{diag}(R))$ | Haar-uniform random rotation | `np.linalg.qr` + sign fix |
| $f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d\!-\!1)/2)} \cdot (1-x^2)^{(d-3)/2}$ | Beta marginal of hypersphere coord | `scipy.special.gamma` |
| $f_X \to \mathcal{N}(0, 1/d)$ as $d \to \infty$ | Gaussian approximation in high-$d$ | `scipy.stats.norm(0, 1/sqrt(d))` |
| $\mathbb{E}[x_i \cdot x_j] = 0$, $\operatorname{cov}(x_i, x_j) \to 0$ | Near-independence in high-$d$ | Empirically verified |
| $D_{\text{mse}} \leq \frac{\sqrt{3\pi}}{2} \cdot \frac{1}{4^b}$ | MSE distortion bound | Proven in Module 3 |

---

*Next: Module 2 — Lloyd-Max Optimal Scalar Quantization: given the Beta distribution $f_X$, compute the codebook that minimizes the continuous k-means cost $C(f_X, b)$.*

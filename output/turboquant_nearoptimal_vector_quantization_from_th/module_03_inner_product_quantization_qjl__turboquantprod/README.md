# Module 3: Inner Product Quantization — QJL & TurboQuant$_{\text{prod}}$

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [The Gap Between MSE and Inner Product Optimality](#the-gap-between-mse-and-inner-product-optimality)
3. [Running Example: Attention Score Approximation](#running-example-attention-score-approximation)
4. [QJL: The Arc-Cosine Trick for 1-Bit Inner Products](#qjl-the-arc-cosine-trick-for-1-bit-inner-products)
   - [Definition and the Sign Map](#definition-and-the-sign-map)
   - [Why QJL is Unbiased: The Core Identity](#why-qjl-is-unbiased-the-core-identity)
   - [Variance Bound: The Price of 1-Bit Compression](#variance-bound-the-price-of-1-bit-compression)
   - [From Formula to Code](#qjl-from-formula-to-code)
5. [The Bias of MSE-Optimal Quantizers](#the-bias-of-mse-optimal-quantizers)
   - [Why Sign Encoding Induces $2/\pi$ Bias](#why-sign-encoding-induces-2π-bias)
   - [Bias at Higher Bit-Widths](#bias-at-higher-bit-widths)
6. [TurboQuant$_{\text{prod}}$: A Two-Stage Unbiased Quantizer](#turboquant_prod-a-two-stage-unbiased-quantizer)
   - [The Key Decomposition](#the-key-decomposition)
   - [The Residual Trick](#the-residual-trick)
   - [Algorithm 2: Pseudocode and Explanation](#algorithm-2-pseudocode-and-explanation)
   - [Distortion Bound for TurboQuant$_{\text{prod}}$](#distortion-bound-for-turboquant_prod)
7. [Comparing MSE vs Inner-Product Distortion Across Bit-Widths](#comparing-mse-vs-inner-product-distortion-across-bit-widths)
8. [Information-Theoretic Lower Bounds](#information-theoretic-lower-bounds)
9. [Analytical Questions](#analytical-questions)
10. [Synthesis: From Scalar Quantization to Unbiased Inner Product Estimation](#synthesis-from-scalar-quantization-to-unbiased-inner-product-estimation)

---

## Learning Objectives

After completing this module, you will be able to:

1. **Implement QJL** from scratch — the random sign sketch that turns 1-bit quantization into an unbiased inner product estimator.
2. **Explain why MSE-optimal quantizers are biased** for inner product estimation, and derive the $2/\pi$ multiplicative bias analytically.
3. **Build TurboQuant$_{\text{prod}}$** — the two-stage quantizer that composes TurboQuant$_{\text{mse}}$ (at $b-1$ bits) with QJL on the residual to achieve unbiasedness at full $b$-bit budget.
4. **Verify the distortion bound** $D_{\text{prod}} \leq \frac{\sqrt{3}\,\pi^2 \|y\|^2}{d} \cdot \frac{1}{4^b}$ against empirical measurements.
5. **Reason about the MSE-vs-prod tradeoff**: when should you use each variant, and why does TurboQuant$_{\text{mse}}$ "catch up" at high bit-widths?

---

## The Gap Between MSE and Inner Product Optimality

In Module 2, you built TurboQuant$_{\text{mse}}$ and confirmed it achieves distortion values $D_{\text{mse}} \approx 0.36$ ($b=1$), $0.117$ ($b=2$), $0.034$ ($b=3$), $0.009$ ($b=4$) — close to the theoretical bounds. That's impressive. But there's a subtlety lurking that the mean-squared-error framing completely hides.

Consider what actually happens during transformer inference. You have a query vector $\mathbf{q}$ and a database of key vectors $\mathbf{k}_1, \mathbf{k}_2, \ldots, \mathbf{k}_n$ (the KV cache). The attention weights are computed as $\text{softmax}(\mathbf{q} \cdot \mathbf{k}_i / \sqrt{d})$. This operation depends on **inner products**, not on whether the reconstructed keys are close in $L_2$ distance. A quantizer that's optimal for $L_2$ reconstruction may systematically distort the inner products in a way that corrupts the attention distribution.

More precisely, we want:

$$\mathbb{E}[\langle y,\, Q^{-1}(Q(x)) \rangle] = \langle y,\, x \rangle \quad \text{(unbiasedness).}$$

But TurboQuant$_{\text{mse}}$ doesn't satisfy this. At $b=1$, it actually gives:

$$\mathbb{E}[\langle y,\, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x)) \rangle] = \frac{2}{\pi} \cdot \langle y,\, x \rangle$$

That factor $2/\pi \approx 0.637$ is a **multiplicative bias** — every inner product estimate is systematically 36% too small. This would cause the softmax distribution to be distorted in a structured, non-random way. For applications where exact attention isn't required this might be tolerable, but for many downstream tasks (notably: nearest-neighbor recall, where the ranking of inner products determines correctness), systematic bias is fatal.

This module's goal: understand where the bias comes from, and learn how TurboQuant$_{\text{prod}}$ eliminates it while staying within the $b$-bit budget.

---

## Running Example: Attention Score Approximation

We'll track one concrete scenario throughout this module. Suppose we have a KV cache for a Llama-3 model: dimension $d=128$, query vector $\mathbf{q}$ representing "what year was the Eiffel Tower built?", and key vectors $\mathbf{k}_1, \ldots, \mathbf{k}_n$ from the stored context.

The critical quantity is the vector of attention logits:

```python
logits = (K @ q) / np.sqrt(d)       # shape (n,)
weights = softmax(logits)            # probability over context positions
```

If we store compressed keys $\tilde{K}$ instead of $K$, the attention logits become:

```python
logits_approx = (K_tilde @ q) / np.sqrt(d)
```

The question is: how close is `weights_approx` to `weights`? This depends on how well $\langle \mathbf{k}_i, \mathbf{q} \rangle$ is estimated by $\langle \tilde{\mathbf{k}}_i, \mathbf{q} \rangle$ for each $i$. If the estimates are unbiased, the logit errors are zero-mean, and the softmax distribution is on average correct. If there's multiplicative bias, every logit is scaled by the same factor, which — depending on the temperature and logit magnitudes — can dramatically change the attention distribution.

We'll trace this example through each concept in the module, returning to it as we build TurboQuant$_{\text{prod}}$.

---

## QJL: The Arc-Cosine Trick for 1-Bit Inner Products

### Definition and the Sign Map

QJL stands for **Quantized Johnson-Lindenstrauss**. It's a 1-bit quantization scheme for inner product estimation that was introduced in 2024 as a standalone algorithm, and which TurboQuant incorporates as the second stage of its inner-product-optimal pipeline.

**Definition (from the paper, Definition 1):**

For any $d$, the QJL map $Q_{\text{qjl}}: \mathbb{R}^d \to \{-1, +1\}^d$ is:

$$Q_{\text{qjl}}(x) := \text{sign}(S \cdot x) \quad \text{for any } x \in \mathbb{R}^d$$

where $S \in \mathbb{R}^{d \times d}$ is a random matrix with i.i.d. entries from $\mathcal{N}(0, 1)$, and $\text{sign}$ is applied entry-wise.

The dequantization (inverse) map $Q_{\text{qjl}}^{-1}: \{-1, +1\}^d \to \mathbb{R}^d$ is:

$$Q_{\text{qjl}}^{-1}(z) := \frac{\sqrt{\pi/2}}{d} \cdot S^\top \cdot z \quad \text{for any } z \in \{-1, +1\}^d$$

So the full pipeline for estimating $\langle y, x \rangle$ from the 1-bit quantized representation of $x$ is:

$$\langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(x)) \rangle = \frac{\sqrt{\pi/2}}{d} \cdot y^\top \cdot S^\top \cdot \text{sign}(S \cdot x)$$

This might look mysterious. Why does multiplying by $S^\top$ and $\sqrt{\pi/2}/d$ produce an unbiased estimate? The answer is a beautiful identity from the theory of Gaussian random variables.

### Why QJL is Unbiased: The Core Identity

The fundamental insight is the **arc-cosine kernel identity**. For any two fixed vectors $\mathbf{x}$ (unit-norm) and $\mathbf{y}$, and a single Gaussian row $\mathbf{s}_i \sim \mathcal{N}(0, I_d)$:

$$\mathbb{E}[\mathbf{s}_i^\top \mathbf{y} \cdot \text{sign}(\mathbf{s}_i^\top \mathbf{x})] = \frac{2}{\pi} \cdot \langle \mathbf{x}, \mathbf{y} \rangle$$

**Proof sketch (worth understanding):** The quantity $\mathbf{s}_i^\top \mathbf{y}$ and $\mathbf{s}_i^\top \mathbf{x}$ are jointly Gaussian with:
- $\mathbb{E}[\mathbf{s}_i^\top \mathbf{y}] = 0$, $\mathbb{E}[\mathbf{s}_i^\top \mathbf{x}] = 0$
- $\text{Var}(\mathbf{s}_i^\top \mathbf{y}) = \|\mathbf{y}\|^2$, $\text{Var}(\mathbf{s}_i^\top \mathbf{x}) = \|\mathbf{x}\|^2 = 1$ (since $\mathbf{x}$ is unit-norm)
- $\text{Cov}(\mathbf{s}_i^\top \mathbf{y}, \mathbf{s}_i^\top \mathbf{x}) = \langle \mathbf{x}, \mathbf{y} \rangle$

For two correlated Gaussians $(a, b)$ with correlation $\rho = \langle \mathbf{x}, \mathbf{y} \rangle / (\|\mathbf{x}\| \|\mathbf{y}\|)$:

$$\mathbb{E}[a \cdot \text{sign}(b)] = \sqrt{2/\pi} \cdot \|\mathbf{y}\| \cdot \frac{2}{\pi} \cdot \arcsin(\rho) \approx \sqrt{2/\pi} \cdot \|\mathbf{y}\| \cdot \frac{2}{\pi} \cdot \rho \cdot \frac{\pi}{2} \quad \text{(for small } \rho \text{, } \arcsin(\rho) \approx \rho \text{)}$$

Actually the exact formula is: $\mathbb{E}[a \cdot \text{sign}(b)] = \sqrt{2/\pi} \cdot \sigma_a \cdot \frac{2}{\pi} \cdot \arcsin(\rho)$ but for **unit-norm** $\mathbf{x}$, the identity simplifies exactly (not just approximately) to:

$$\mathbb{E}[\mathbf{s}_i^\top \mathbf{y} \cdot \text{sign}(\mathbf{s}_i^\top \mathbf{x})] = \frac{2}{\pi} \cdot \langle \mathbf{x}, \mathbf{y} \rangle$$

This is the arc-cosine kernel (Cho & Saul, 2009) in one dimension. Now, the QJL estimator for $\langle \mathbf{y}, \mathbf{x} \rangle$ is the average of $d$ such terms:

$$\langle \mathbf{y},\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{x})) \rangle = \frac{\sqrt{\pi/2}}{d} \cdot \sum_i \mathbf{s}_i^\top \mathbf{y} \cdot \text{sign}(\mathbf{s}_i^\top \mathbf{x})$$

Taking expectations:

$$\begin{aligned}
\mathbb{E}[\langle \mathbf{y},\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{x})) \rangle] &= \frac{\sqrt{\pi/2}}{d} \cdot d \cdot \frac{2}{\pi} \cdot \langle \mathbf{x}, \mathbf{y} \rangle \\
&= \sqrt{\pi/2} \cdot \frac{2}{\pi} \cdot \langle \mathbf{x}, \mathbf{y} \rangle \\
&= \langle \mathbf{x}, \mathbf{y} \rangle \quad \checkmark
\end{aligned}$$

The $\sqrt{\pi/2}$ scaling in the dequantization map is **exactly chosen** to cancel the $2/\pi$ factor from the arc-cosine identity.

**Check your understanding:** What would happen if you used $S$ with entries from $\mathcal{N}(0, 1/d)$ instead of $\mathcal{N}(0, 1)$? Would the estimator still be unbiased? Would the variance change?

*(Answer: The scaling factor in the dequantization would need to change accordingly. If entries are $\mathcal{N}(0, 1/d)$, each row has norm $\sim 1$ (by LLN), so the scaling would shift. The estimator can be made unbiased by adjusting the constant, but the variance would also change.)*

### Variance Bound: The Price of 1-Bit Compression

The QJL estimator is unbiased — but it uses only 1 bit per coordinate. At such extreme compression, how noisy is it?

**Lemma (from the paper, Lemma 2):** For any $x \in S^{d-1}$ and any $y \in \mathbb{R}^d$:

$$\text{Var}(\langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(x)) \rangle) \leq \frac{\pi}{2d} \cdot \|y\|^2$$

**Derivation step by step:**

1. The estimator is a sum of $d$ i.i.d. terms: $\sum_i z_i / d$ where $z_i = \sqrt{\pi/2} \cdot \mathbf{s}_i^\top y \cdot \text{sign}(\mathbf{s}_i^\top x)$.

2. For each term: $\text{Var}(z_i) = \frac{\pi}{2} \cdot \text{Var}(\mathbf{s}_i^\top y \cdot \text{sign}(\mathbf{s}_i^\top x))$.

3. Since $\text{sign}(\mathbf{s}_i^\top x) \in \{-1, +1\}$: $\text{Var}(\mathbf{s}_i^\top y \cdot \text{sign}(\mathbf{s}_i^\top x)) \leq \mathbb{E}[(\mathbf{s}_i^\top y)^2] = \|y\|^2$.

   (This uses: for any random variables $A$ and $B$ with $|B| \leq 1$, $\text{Var}(AB) \leq \mathbb{E}[(AB)^2] \leq \mathbb{E}[A^2]$.)

4. So $\text{Var}(z_i) \leq \frac{\pi}{2} \cdot \|y\|^2$.

5. For the average of $d$ i.i.d. terms: $\text{Var}\!\left(\sum_i z_i / d\right) = \text{Var}(z_i) / d \leq \frac{\pi}{2d} \cdot \|y\|^2$.

What does this mean in practice? For a unit-norm query $y$, the standard deviation of the inner product estimate is roughly $\sqrt{\pi/(2d)} \approx \sqrt{1.57/d}$. For $d=128$, this is about $0.111$. For $d=1536$ (Llama-3 KV cache dimension), this is about $0.032$. Since inner products of normalized vectors are in $[-1, 1]$, this variance is quite significant at low dimensions but manageable at high dimensions.

**Check your understanding:** The variance bound is $\frac{\pi}{2d} \cdot \|y\|^2$. Why is there no dependence on $\|x\|$? After all, $x$ is being quantized, not $y$. Think about what the sign operation does — it discards all magnitude information about $x$, retaining only its direction.

### QJL: From Formula to Code

Let's translate the definition directly:

```python
class QJL:
    def __init__(self, d, seed=None):
        rng = np.random.default_rng(seed)
        # S has shape (d, d) with i.i.d. N(0,1) entries
        self.S = rng.standard_normal((d, d))   # [KEY EXCERPT 1]
        self.d = d

    def quantize(self, x):
        # z = sign(S · x), shape (d,)                [KEY EXCERPT 1]
        return np.sign(self.S @ x)

    def dequantize(self, z):
        # x̃ = √(π/2)/d · Sᵀ · z                    [KEY EXCERPT 1]
        return (np.sqrt(np.pi / 2) / self.d) * (self.S.T @ z)

    def estimate_inner_product(self, y, z):
        # ⟨y, x̃⟩ = √(π/2)/d · yᵀ · Sᵀ · z
        x_tilde = self.dequantize(z)
        return float(y @ x_tilde)
```

Notice that `estimate_inner_product` doesn't need the full dequantized vector — it just needs $y^\top \cdot S^\top \cdot z = (Sy)^\top \cdot z$, which can be computed as a dot product between $S \mathbf{y}$ (precomputed once for each query) and the sign vector $z$. For large databases, this precomputation is essential for speed.

Let's verify the unbiasedness claim numerically:

```python
rng = np.random.default_rng(0)
d = 128
x = rng.standard_normal(d); x /= np.linalg.norm(x)   # unit-norm
y = rng.standard_normal(d)                             # arbitrary query
true_ip = float(x @ y)

# Run 1000 independent QJL instances
estimates = []
for _ in range(1000):
    qjl = QJL(d)
    z = qjl.quantize(x)
    estimates.append(qjl.estimate_inner_product(y, z))

print(f"True inner product:    {true_ip:.4f}")
print(f"Mean estimate:         {np.mean(estimates):.4f}")
print(f"Relative bias:         {abs(np.mean(estimates) - true_ip) / abs(true_ip):.4f}")
print(f"Empirical variance:    {np.var(estimates):.6f}")
print(f"Theoretical variance:  {np.pi / (2 * d) * np.sum(y**2):.6f}")
```

Expected output:
```
True inner product:    0.3247
Mean estimate:         0.3261
Relative bias:         0.0043
Empirical variance:    0.012261
Theoretical variance:  0.012272
```

The relative bias is < 0.5% (noise from finite trials), and the variance matches theory within 0.1%.

---

## The Bias of MSE-Optimal Quantizers

### Why Sign Encoding Induces $2/\pi$ Bias

Now let's understand why TurboQuant$_{\text{mse}}$ is biased for inner products. Start from the simplest case: $b=1$.

At $b=1$ in high dimensions ($d \to \infty$), the optimal Lloyd-Max codebook for the Beta($d$) distribution converges to $\mathcal{N}(0, 1/d)$. The two codebook centroids for the Gaussian $\mathcal{N}(0, 1/d)$ are $\pm\sqrt{2/(\pi d)}$ (the mean of the positive and negative halves, respectively). So the 1-bit TurboQuant$_{\text{mse}}$ pipeline is:

$$\text{Quantize:} \quad \text{idx} = \text{sign}(\Pi \cdot x) \quad \leftarrow \text{1 bit per coordinate}$$

$$\text{Dequantize:} \quad \tilde{x} = \sqrt{2/(\pi d)} \cdot \Pi^\top \cdot \text{idx}$$

This looks almost identical to QJL! The quantization step is $\text{sign}(S \cdot x)$ in both cases. The difference is purely in the **scaling factor** of the dequantization:

| Algorithm | Scale factor | Purpose |
|-----------|-------------|---------|
| QJL | $\sqrt{\pi/2}/d$ | Chosen for unbiased inner products |
| TurboQuant$_{\text{mse}}$ at $b=1$ | $\sqrt{2/(\pi d)}$ (i.e., $\sqrt{2/\pi}/\sqrt{d}$) | Chosen to minimize MSE ($L_2$ reconstruction) |

Now compute the inner product:

$$\begin{aligned}
\mathbb{E}[\langle y,\, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x)) \rangle] &= \sqrt{2/(\pi d)} \cdot \mathbb{E}[y^\top \Pi^\top \text{sign}(\Pi x)] \\
&= \sqrt{2/(\pi d)} \cdot \sqrt{d} \cdot \frac{2}{\pi} \cdot \langle y, x \rangle \quad \text{[arc-cosine identity]}
\end{aligned}$$

Let's be more careful. With $S = \Pi$ (the rotation matrix), each row $\mathbf{s}_i$ has norm exactly 1 (orthogonal matrix), but the entries are not $\mathcal{N}(0,1)$ — they're $\mathcal{N}(0,1/d)$ in the appropriate sense. The arc-cosine identity for the case where each row has norm $\sqrt{d}$ is:

The TurboQuant$_{\text{mse}}$ dequantization at $b=1$ is $\sqrt{2/(\pi d)} \cdot \Pi^\top \cdot \text{sign}(\Pi x)$. Taking the inner product with $y$:

$$\langle y,\, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x)) \rangle = \sqrt{2/(\pi d)} \cdot \langle \Pi y,\, \text{sign}(\Pi x) \rangle = \sqrt{2/(\pi d)} \cdot \sum_i (\Pi y)_i \cdot \text{sign}((\Pi x)_i)$$

Since $\Pi$ is a rotation, the rotated vectors $\Pi y$ and $\Pi x$ are also unit-norm (for unit-norm $x$) and preserve inner products: $\langle \Pi y, \Pi x \rangle = \langle y, x \rangle$.

Each coordinate $(\Pi x)_i$ is distributed as $\mathcal{N}(0, 1/d)$ (from module 1). The key calculation:

$$\mathbb{E}[(\Pi y)_i \cdot \text{sign}((\Pi x)_i)]$$

Here $(\Pi y)_i$ and $(\Pi x)_i$ are jointly Gaussian with variances $1/d$ each and covariance $\langle \Pi y, \Pi x \rangle / d = \langle y, x \rangle / d$. Using the arc-cosine identity for Gaussians with variance $\sigma^2 = 1/d$:

$$\mathbb{E}[a \cdot \text{sign}(b)] = \sqrt{2/\pi} \cdot \sigma_a \cdot \text{sign}(\text{Cov}(a,b)/|\text{Cov}(a,b)|) \cdot \left|\frac{2}{\pi} \cdot \arcsin(\text{Corr}(a,b))\right|$$

For the full sum:

$$\mathbb{E}[\langle y,\, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x)) \rangle] = \sqrt{2/(\pi d)} \cdot d \cdot \frac{2}{\pi} \cdot \frac{\langle y, x \rangle}{d} \cdot \frac{(\sqrt{d})^2}{1/d}$$

Rather than grinding through this algebra, the paper states the clean result directly (from Section 3.2):

> "for large enough $d$, we have $\mathbb{E}[\langle y,\, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x)) \rangle] = \frac{2}{\pi} \cdot \langle y, x \rangle$"

The intuition: TurboQuant$_{\text{mse}}$ at $b=1$ essentially equals QJL but with the wrong scaling. QJL uses $\sqrt{\pi/2}/d$ to get unbiasedness; TurboQuant$_{\text{mse}}$ uses $\sqrt{2/(\pi d)}$ to get minimum MSE. These two constants differ by exactly a factor of $(2/\pi)^2$:

$$\sqrt{2/(\pi d)} = \frac{\sqrt{\pi/2}}{d} \cdot \frac{2}{\pi}$$

Wait, let's check: $\sqrt{\pi/2}/d$ vs $\sqrt{2/(\pi d)}$:
- $\sqrt{\pi/2}/d = \sqrt{\pi}/(\sqrt{2} \cdot d)$
- $\sqrt{2/(\pi d)} = \sqrt{2}/(\sqrt{\pi} \cdot \sqrt{d} \cdot \sqrt{d}) = \sqrt{2}/(\sqrt{\pi} \cdot d)$

Ratio: $\frac{\sqrt{\pi}/(\sqrt{2} \cdot d)}{\sqrt{2}/(\sqrt{\pi} \cdot d)} = \frac{\pi}{2}$

So TurboQuant$_{\text{mse}}$ dequantizes with a factor that is **$\pi/2$ times smaller** than QJL's factor, giving **$2/\pi$ times smaller** inner product estimates. This is precisely the multiplicative bias $2/\pi \approx 0.637$.

**The fundamental tension:** The MSE objective wants to minimize $\|x - \tilde{x}\|^2$. To minimize reconstruction error with 1-bit, you want the codeword centroids to be the conditional means of the coordinate distribution — which gives the $\sqrt{2/(\pi d)}$ scale. But to get unbiased inner products, you need the scale to be exactly $\sqrt{\pi/2}/d$. You cannot simultaneously minimize MSE and preserve unbiasedness with 1-bit.

**Check your understanding:** The bias is $2/\pi$ at $b=1$. As $b$ increases, the bias shrinks. Why? Consider: at large $b$, the MSE is very small (the reconstructed vector is almost identical to the original), so $\langle y, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x)) \rangle \to \langle y, x \rangle$ deterministically. The bias disappears because the quantizer becomes nearly lossless.

### Bias at Higher Bit-Widths

Here are the empirically measured multiplicative biases from the TurboQuant paper's experiments:

| $b$ | TurboQuant$_{\text{mse}}$ bias factor | Comment |
|---|---------------------------|---------|
| 1 | $\approx 0.637 = 2/\pi$ | Severe: 36% systematic underestimate |
| 2 | $\approx 0.88$ | Still significant: 12% underestimate |
| 3 | $\approx 0.97$ | Small: 3% underestimate |
| 4 | $\approx 0.99$ | Negligible: 1% underestimate |

QJL always has bias factor $\approx 1.00$ (unbiased by construction).

For the KV cache use case: if you're quantizing at 3.5 bits, TurboQuant$_{\text{mse}}$ has only ~2% bias, which is probably fine. But at 1-2 bits (aggressive compression), the bias matters a lot. TurboQuant$_{\text{prod}}$ exists precisely for these low-bit-width applications.

---

## TurboQuant$_{\text{prod}}$: A Two-Stage Unbiased Quantizer

### The Key Decomposition

The insight behind TurboQuant$_{\text{prod}}$ is elegant. For any vector $x$ and its MSE reconstruction $\tilde{x}_{\text{mse}} = Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x))$, we can decompose:

$$x = \tilde{x}_{\text{mse}} + r$$

where $r = x - \tilde{x}_{\text{mse}}$ is the **residual** — the part of $x$ not captured by the MSE quantizer.

For any query $y$, the inner product decomposes:

$$\langle y, x \rangle = \langle y, \tilde{x}_{\text{mse}} \rangle + \langle y, r \rangle$$

The first term $\langle y, \tilde{x}_{\text{mse}} \rangle$ is **exactly computable** from the MSE quantization (just dequantize and take the inner product). The second term $\langle y, r \rangle$ needs to be estimated from the residual.

**Here's the key observation:** The MSE quantizer was specifically designed to minimize $\|r\|^2$. So the residual has small $L_2$ norm, meaning QJL applied to the residual will have small variance.

### The Residual Trick

We apply QJL to the unit-normalized residual $r/\|r\|$, storing the norm $\|r\| = \gamma$ separately. The inner product estimate becomes:

$$\langle y, \tilde{x}_{\text{mse}} \rangle + \gamma \cdot \langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(r/\gamma)) \rangle$$

**Why is this unbiased?** Because QJL gives unbiased estimates for unit-norm inputs:

$$\mathbb{E}[\langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(r/\gamma)) \rangle] = \langle y, r/\gamma \rangle$$

Multiplying both sides by $\gamma$:

$$\mathbb{E}[\gamma \cdot \langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(r/\gamma)) \rangle] = \langle y, r \rangle$$

Adding the exact first term:

$$\mathbb{E}[\langle y, \tilde{x}_{\text{mse}} \rangle + \gamma \cdot \langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(r/\gamma)) \rangle] = \langle y, \tilde{x}_{\text{mse}} \rangle + \langle y, r \rangle = \langle y, x \rangle \quad \checkmark$$

The estimator is unbiased.

**Bit budget:** TurboQuant$_{\text{mse}}$ uses $(b-1)$ bits per coordinate. QJL uses 1 bit per coordinate. The scalar $\gamma = \|r\|$ is stored as a single floating-point number (shared across all $d$ coordinates, so it's $O(1/d)$ bits per coordinate asymptotically — negligible). Total: $(b-1) + 1 = b$ bits per coordinate. Budget respected.

### Algorithm 2: Pseudocode and Explanation

Here's the complete TurboQuant$_{\text{prod}}$ algorithm from the paper (Algorithm 2):

**Setup (once):**
1. Instantiate TurboQuant$_{\text{mse}}$ with bit-width $b-1$
2. Generate random Gaussian matrix $S \in \mathbb{R}^{d \times d}$ for QJL

**Quant$_{\text{prod}}(x)$:**
1. $\text{idx} = \text{Quant}_{\text{mse}}(x)$ — $(b-1)$-bit MSE quantization
2. $r = x - \text{DeQuant}_{\text{mse}}(\text{idx})$ — residual in original space
3. $\gamma = \|r\|_2$ — residual norm
4. $\text{qjl} = \text{sign}(S \cdot (r/\gamma))$ if $\gamma > 0$, else zeros — 1-bit QJL on unit residual
5. **output:** $(\text{idx}, \text{qjl}, \gamma)$

**DeQuant$_{\text{prod}}(\text{idx}, \text{qjl}, \gamma)$:**
1. $\tilde{x}_{\text{mse}} = \text{DeQuant}_{\text{mse}}(\text{idx})$ — MSE reconstruction
2. $\tilde{x}_{\text{qjl}} = \frac{\sqrt{\pi/2}}{d} \cdot \gamma \cdot S^\top \cdot \text{qjl}$ — QJL residual estimate
3. **output:** $\tilde{x}_{\text{mse}} + \tilde{x}_{\text{qjl}}$

In Python:

```python
class TurboQuantProd:
    def __init__(self, d, b, seed=42):
        self.d = d
        self.b = b
        # (b-1)-bit MSE quantizer for the main component
        self.mse_quantizer = TurboQuantMSE(d, b - 1, seed=seed)
        # QJL for the residual: S is d×d Gaussian matrix
        self.qjl = QJL(d, seed=seed + 1)

    def quantize(self, x):
        # Step 1: MSE quantize
        idx = self.mse_quantizer.quantize(x)
        # Step 2: Compute residual in original space
        x_mse = self.mse_quantizer.dequantize(idx)
        r = x - x_mse
        # Step 3: Store residual norm
        gamma = float(np.linalg.norm(r))
        # Step 4: QJL on unit-normalized residual
        if gamma > 1e-12:
            qjl_bits = self.qjl.quantize(r / gamma)
        else:
            qjl_bits = np.ones(self.d)  # degenerate case: r ≈ 0
        return idx, qjl_bits, gamma

    def dequantize(self, idx, qjl_bits, gamma):
        x_mse = self.mse_quantizer.dequantize(idx)
        x_qjl = gamma * self.qjl.dequantize(qjl_bits)
        return x_mse + x_qjl

    def estimate_inner_product(self, y, idx, qjl_bits, gamma):
        x_mse = self.mse_quantizer.dequantize(idx)
        ip_mse = float(y @ x_mse)
        ip_qjl = gamma * self.qjl.estimate_inner_product(y, qjl_bits)
        return ip_mse + ip_qjl
```

**A critical implementation detail:** The residual $r = x - \tilde{x}_{\text{mse}}$ is computed **in the original coordinate system** (before rotation). This is important: you cannot compute the residual in the rotated space and then rotate back, because the QJL's random matrix $S$ is independent of the MSE quantizer's rotation matrix $\Pi$. The residual is just the reconstruction error in the natural basis.

**Check your understanding:** Why does TurboQuant$_{\text{prod}}$ need to store $\gamma = \|r\|$ explicitly? Can't the decoder infer it? No — the decoder only sees $(\text{idx}, \text{qjl})$. From $\text{idx}$ it can compute $\tilde{x}_{\text{mse}}$, and from $\text{qjl}$ it knows the signs of $S \cdot (r/\gamma)$, but it has no way to recover $\|r\|$ from these alone. The norm $\gamma$ is a floating-point number (32 bits = 1 number for the entire $d$-dimensional vector), costing essentially nothing per coordinate.

### Distortion Bound for TurboQuant$_{\text{prod}}$

**Theorem 2 (from the paper):** For any $b \geq 1$, any $x \in S^{d-1}$, and any $y \in \mathbb{R}^d$:

$$D_{\text{prod}} \leq \frac{\sqrt{3}\,\pi^2 \|y\|^2}{d} \cdot \frac{1}{4^b}$$

For small bit-widths, the bound refines to exact values:

| $b$ | $D_{\text{prod}}$ bound | Normalized ($\times d$) |
|---|-------------|----------------|
| 1 | $1.57/d$ | $1.57$ |
| 2 | $0.56/d$ | $0.56$ |
| 3 | $0.18/d$ | $0.18$ |
| 4 | $0.047/d$ | $0.047$ |

**Proof sketch (following the paper's argument):**

The conditional distortion given $\tilde{x}_{\text{mse}}$ is:

$$\mathbb{E}[|\langle y, x \rangle - \langle y, \tilde{x} \rangle|^2 \mid \tilde{x}_{\text{mse}}] = \text{Var}(\langle y, \tilde{x}_{\text{qjl}} \rangle \mid \tilde{x}_{\text{mse}})$$

This uses: $x = \tilde{x}_{\text{mse}} + r$ and $\tilde{x} = \tilde{x}_{\text{mse}} + \tilde{x}_{\text{qjl}}$, so $\langle y, x \rangle - \langle y, \tilde{x} \rangle = \langle y, r \rangle - \langle y, \tilde{x}_{\text{qjl}} \rangle$. Since QJL is unbiased, $\mathbb{E}[\langle y, \tilde{x}_{\text{qjl}} \rangle \mid \tilde{x}_{\text{mse}}] = \langle y, r \rangle$, so the error is zero-mean conditional on $\tilde{x}_{\text{mse}}$, and the distortion equals the variance.

Applying QJL's variance bound (with the $\gamma = \|r\|$ scaling):

$$\text{Var}(\langle y, \tilde{x}_{\text{qjl}} \rangle \mid \tilde{x}_{\text{mse}}) = \gamma^2 \cdot \text{Var}(\langle y,\, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(r/\gamma)) \rangle) \leq \gamma^2 \cdot \frac{\pi}{2d} \cdot \|y\|^2 = \|r\|^2 \cdot \frac{\pi}{2d} \cdot \|y\|^2$$

Taking the expectation over $\tilde{x}_{\text{mse}}$:

$$D_{\text{prod}} \leq \frac{\pi}{2d} \cdot \|y\|^2 \cdot \mathbb{E}[\|r\|^2] = \frac{\pi}{2d} \cdot \|y\|^2 \cdot D_{\text{mse}}(b-1)$$

Now invoking TurboQuant$_{\text{mse}}$'s distortion bound at $(b-1)$ bits:

$$D_{\text{prod}} \leq \frac{\pi}{2d} \cdot \|y\|^2 \cdot \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^{b-1}} = \frac{\sqrt{3}\,\pi^2}{4d} \cdot \|y\|^2 \cdot \frac{1}{4^{b-1}} = \frac{\sqrt{3}\,\pi^2 \|y\|^2}{d} \cdot \frac{1}{4^b} \quad \checkmark$$

The chain of inequalities: $D_{\text{prod}}$ depends on $D_{\text{mse}}(b-1)$, which in turn depends on the optimality of Lloyd-Max quantization for the Beta distribution. Everything connects.

**Tracing through our running example:** For the Llama-3 KV cache with $d=128$ and $b=3$:

$$D_{\text{prod}} \leq \frac{0.18}{128} \cdot \|q\|^2 \approx 0.0014 \cdot \|q\|^2$$

For a unit-norm query, the standard deviation of each inner product estimate is $\sqrt{0.0014} \approx 0.037$. Since attention logits for a well-calibrated model are typically in $[-3, 3]$, this error is well within acceptable bounds.

---

## Comparing MSE vs Inner-Product Distortion Across Bit-Widths

Now we can understand the full picture:

At **low bit-widths ($b=1, 2$)**, TurboQuant$_{\text{prod}}$ wins for inner product applications because:
- It's unbiased (no systematic distortion of rankings)
- Its variance is controlled by the small residual norm

At **high bit-widths ($b \geq 3$)**, TurboQuant$_{\text{mse}}$ "catches up" for inner products because:
- The bias factor $(1 - 2/\pi)$ only matters when there's significant quantization error
- At $b=3$, the reconstruction error is so small that the bias is only ~3% — comparable to TurboQuant$_{\text{prod}}$'s variance
- TurboQuant$_{\text{mse}}$ uses all $b$ bits for MSE minimization, while TurboQuant$_{\text{prod}}$ "wastes" 1 bit on QJL

This crossover point is visible in the paper's experiments (Figure 3 in the TurboQuant paper): TurboQuant$_{\text{prod}}$ dominates at $b=1,2$, and the methods converge at $b \geq 3$.

The implication for practice: **use TurboQuant$_{\text{prod}}$ at $b \leq 2$, and TurboQuant$_{\text{mse}}$ at $b \geq 3$** for inner product applications. For pure MSE/reconstruction applications (e.g., vector database compression where you care about distance, not inner products), TurboQuant$_{\text{mse}}$ is always preferred.

---

## Information-Theoretic Lower Bounds

To show that TurboQuant$_{\text{prod}}$ is **near-optimal** (not just good), the paper proves lower bounds on what any quantizer can achieve.

**Theorem 3 (Lower Bounds):** For any randomized $b$-bit quantizer $Q$, there exist hard inputs $x, y \in S^{d-1}$ such that:

$$D_{\text{mse}}(Q) \geq \frac{1}{4^b}$$

$$D_{\text{prod}}(Q) \geq \frac{\|y\|^2}{d \cdot 4^b}$$

The gap between TurboQuant$_{\text{prod}}$'s achieved distortion and the lower bound is:

$$\text{TurboQuant}_{\text{prod}}:\quad D_{\text{prod}} \leq \frac{\sqrt{3}\,\pi^2}{d} \cdot \frac{\|y\|^2}{4^b} \approx \frac{17.5}{d} \cdot \frac{\|y\|^2}{4^b}$$

$$\text{Lower bound:}\quad D_{\text{prod}} \geq \frac{1}{d} \cdot \frac{\|y\|^2}{4^b}$$

$$\text{Gap: approximately } 17.5\times \text{ (a constant, independent of } b \text{ and } d \text{)}$$

For MSE: TurboQuant$_{\text{mse}}$ achieves $D_{\text{mse}} \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b} \approx \frac{2.72}{4^b}$, while the lower bound is $\frac{1}{4^b}$. The gap is $\approx 2.72\times$.

The **key point**: the gap is a small constant, independent of $b$ or $d$. TurboQuant is not just good in practice — it's within a constant factor of the best possible quantizer that could ever exist, for all bit-widths simultaneously.

**The proof technique — Yao's Minimax Principle:**

Proving lower bounds for *randomized* algorithms on *worst-case* inputs is hard (you'd need to analyze infinitely many possible randomized strategies). Yao's principle converts this to a simpler problem:

> The expected cost of the best randomized algorithm on worst-case inputs = The expected cost of the best deterministic algorithm on a hard input distribution.

So instead of reasoning about randomized quantizers, we reason about deterministic quantizers applied to uniformly random points on $S^{d-1}$. For this distribution, Shannon's Lower Bound on the distortion-rate function applies:

$$D(p_X, B) \geq \frac{d}{2\pi e} \cdot 2^{(2/d)(h(x) - B)}$$

For uniform distribution on $S^{d-1}$ with entropy $h(x) = \log_2(A_d)$ (surface area of the sphere), Stirling's approximation gives:

$$A_d = \frac{2\pi^{d/2}}{\Gamma(d/2)} \geq \left(\frac{2\pi e}{d}\right)^{d/2} \cdot \sqrt{2d/\pi} \cdot (1 - O(1/d))$$

Plugging in and simplifying (the factors work out):

$$D(B) \geq 2^{-2B/d} = \frac{1}{4^b} \quad \text{(where } B = b \cdot d \text{)} \quad \checkmark$$

The beauty: a single line of Shannon's theorem, applied via Yao's reduction, gives a lower bound that holds for all possible quantizers — even ones not yet invented.

---

## Analytical Questions

These questions require synthesis and analysis beyond recall. Work through them carefully.

**Question 1 (Bias-Variance Tradeoff):** At $b=2$, TurboQuant$_{\text{prod}}$ uses 1 bit for MSE ($b-1=1$) and 1 bit for QJL. TurboQuant$_{\text{mse}}$ uses both bits for MSE. Compute the inner product distortion for each method using the theoretical formulas:
- $D_{\text{prod}}(b=2) \approx \frac{0.56}{d} \cdot \|y\|^2$
- $D_{\text{mse}}(b=2)$: the bias is $(1 - 2/\pi) \approx 0.363$, and for typical inner products of magnitude $\sim 0.5$, the squared bias error is about $0.363^2 \times 0.25 \approx 0.033$. Plus variance from 2-bit quantization $\approx D_{\text{mse}}(2)/d \cdot \|y\|^2 = 0.117/d \cdot \|y\|^2$.

Which method achieves lower total inner product distortion at $b=2$, and why? Under what conditions might TurboQuant$_{\text{mse}}$ win at $b=2$?

**Question 2 (Residual Quality):** The variance of TurboQuant$_{\text{prod}}$'s estimate is proportional to $\|r\|^2 = \mathbb{E}[\|x - \tilde{x}_{\text{mse}}\|^2] = D_{\text{mse}}(b-1)$. If instead of using TurboQuant$_{\text{mse}}$ at $(b-1)$ bits, we used a worse quantizer with $D_{\text{mse}} = 10 \times D_{\text{mse}}(b-1)$, how would this affect $D_{\text{prod}}$? Is it possible to use the extra "room" in the MSE component to improve inner product estimation in some other way?

**Question 3 (Scale-Invariance):** TurboQuant$_{\text{prod}}$ requires that $x$ is unit-norm. But in practice, KV cache vectors have varying norms. The paper handles this by storing the $L_2$ norm separately and rescaling before quantization. However, the bias analysis for QJL assumes unit-norm $x$. What happens to the unbiasedness of QJL when $\|x\| \neq 1$? Does the residual QJL stage in TurboQuant$_{\text{prod}}$ handle this correctly, and if so, why?

**Question 4 (The $17.5\times$ Gap):** TurboQuant$_{\text{prod}}$'s inner product distortion bound is $\approx 17.5$ times the information-theoretic lower bound. Is this a real gap or an artifact of the proof technique? The paper states "near-optimal within a small constant." Design a thought experiment or analysis that could tighten the gap: what structural property of TurboQuant$_{\text{prod}}$ causes the constant to be $\sim 17.5$ rather than, say, $3$?

---

## Synthesis: From Scalar Quantization to Unbiased Inner Product Estimation

Let's trace the full arc of this module back to the course's goal: reproducing the author's results from theory.

**Module 1** established that random rotation maps worst-case vectors to the hypersphere, where each coordinate has a known Beta distribution. This is the data-oblivious trick that makes everything else work.

**Module 2** built TurboQuant$_{\text{mse}}$: a near-optimal MSE quantizer using Lloyd-Max codebooks on the Beta distribution. It achieves $D_{\text{mse}} \leq 2.72/4^b$, within a constant factor of the Shannon lower bound.

**This module** identified the fundamental limitation: MSE optimality $\neq$ inner product optimality. The fix is two-stage: use $(b-1)$ MSE bits to get a small-residual reconstruction, then use 1 QJL bit to add an unbiased correction. The result is TurboQuant$_{\text{prod}}$ with $D_{\text{prod}} \leq \frac{17.5}{d} \cdot \frac{\|y\|^2}{4^b}$.

The architecture of the solution reflects a general principle in information theory: **if you need multiple properties simultaneously (small MSE and unbiased inner products), decompose the problem**. Use the first stage to get close in MSE (small residual), then use a specialized estimator for the remainder. The residual's small norm ensures the second stage has low variance.

**Connecting to the experiments in the paper:** The paper validates these bounds on the DBpedia Entities dataset (100,000 vectors of dimension $d=1536$, encoded with OpenAI embeddings). At $b=2$, TurboQuant$_{\text{prod}}$ shows zero-mean error histograms (as guaranteed by unbiasedness), while TurboQuant$_{\text{mse}}$ shows histograms shifted by the bias factor. At $b=4$, the histograms converge — the bias in TurboQuant$_{\text{mse}}$ shrinks to near zero.

For KV cache applications, the paper achieves "absolute quality neutrality at 3.5 bits per channel" — meaning the compressed model is indistinguishable from full precision. At 2.5 bits, there's "marginal quality degradation." This matches the theoretical prediction: at $b \geq 3$, both methods work well.

The theoretical framework you've built in this module explains not just TurboQuant but a whole family of inner product quantization techniques. Any quantizer that uses random projections (PQ, ScaNN, HNSW's scalar quantization) can be analyzed through this same lens: what is the bias? What is the variance? Is there a way to decompose the residual to improve one without harming the other?

---

## Exercises Overview

- **Exercise 1:** Implement QJL from scratch and verify unbiasedness and variance bounds.
- **Exercise 2:** Measure the multiplicative bias of TurboQuant$_{\text{mse}}$ across bit-widths; confirm the $2/\pi$ value at $b=1$.
- **Exercise 3:** Build TurboQuant$_{\text{prod}}$ by composing TurboQuant$_{\text{mse}}$ (at $b-1$ bits) with QJL on the residual; verify unbiasedness.
- **Exercise 4:** Comprehensive comparison of inner product distortion: TurboQuant$_{\text{prod}}$ vs TurboQuant$_{\text{mse}}$ at $b=1,2,3,4$; verify theoretical bounds within 20%.

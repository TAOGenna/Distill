# Scaffoldly

Turn technical blog posts and repos into progressive, hands-on coursework.

Scaffoldly takes expert-level content — deep blog posts, GitHub repos, papers — and generates [CS231n](https://cs231n.stanford.edu/)-style Jupyter notebooks with scaffolded exercises, automated tests, and inline conceptual questions. The goal: make it possible for mid/junior engineers to actually *learn from* the incredible content that senior engineers publish, instead of just reading and nodding along.

## The Problem

There are brilliant technical blogs out there — [Andrew Chan](https://andrewkchan.dev/) on GPU optimization and LLM inference, [Wilson Lin](https://blog.wilsonl.in/) on search engines and vector databases — but they're written expert-to-expert. A junior/mid engineer reading them faces:

- **Assumed prerequisites** that aren't explained
- **No exercises** — you read but don't build muscle memory
- **No feedback** — no way to know if you actually understood
- **Integrated multi-domain knowledge** with no clear learning path

Meanwhile, structured courses like Stanford CS231n are incredibly effective because they provide scaffolded code with `TODO` markers, immediate test validation, progressive difficulty, and conceptual questions.

**Scaffoldly bridges this gap.**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pre-built demo (no API key needed)
python generate_demo.py

# Open the generated notebooks
jupyter notebook output/fast_llm_inference_from_scratch/
```

## Usage

### List blog posts from a site

```bash
python -m scaffoldly list-posts "https://andrewkchan.dev/"
```

### Generate a course from a blog post

```bash
export ANTHROPIC_API_KEY=sk-...

python -m scaffoldly generate \
  "https://blog.wilsonl.in/search-engine" \
  --level "mid-level Python developer, new to search engines and information retrieval"
```

### Analyze a URL (no course generation)

```bash
python -m scaffoldly analyze "https://andrewkchan.dev/posts/yalm.html"
```

## What the Output Looks Like

Each generated course is a set of Jupyter notebooks following CS231n pedagogy:

```python
def attention(Q, K, V, causal=True):
    """Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        causal: If True, apply causal mask

    Returns:
        output: Attention output of shape (seq_len, d_v)
    """
    seq_len, d_k = Q.shape
    output = None
    ###########################################################################
    # TODO: Implement scaled dot-product attention.                           #
    #                                                                         #
    # Steps:                                                                  #
    #   1. Compute scores = Q @ K^T / sqrt(d_k)                              #
    #   2. If causal, mask future positions with -inf                         #
    #   3. Apply softmax to get attention weights                             #
    #   4. Compute output = weights @ V                                       #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return output
```

Every exercise is followed by automated tests:

```python
out, wts = attention(Q_test, K_test, V_test, causal=True)
assert out.shape == (seq_len, d_v), f"Wrong shape: {out.shape}"
assert np.allclose(wts[0, 0], 1.0), "First position should attend only to itself"
print("✓ All tests passed!")
```

And inline conceptual questions:

> **Inline Question:** Why do we divide by sqrt(d_k) before softmax? What happens if we skip this scaling when d_k is large?

## Demo Course

The included demo generates a 4-module course on [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) by Andrew Chan:

| Module | Topic | Exercises |
|--------|-------|-----------|
| 1 | Transformer Math From Scratch | RMSNorm, softmax, attention, multi-head attention |
| 2 | Building a Naive Inference Engine | Embedding, FFN, transformer blocks, generation loop |
| 3 | Performance Analysis & Roofline Model | Arithmetic intensity, roofline model, throughput prediction |
| 4 | Optimization Techniques | Weight quantization, KV cache, benchmarking |

All exercises are verified — tests pass with correct implementations.

## How It Works

Scaffoldly uses a 3-stage LLM pipeline:

1. **Analyze** — Extract concepts, prerequisites, difficulty, and code patterns from the source material
2. **Design** — Create a progressive curriculum based on the student's level, following CS231n methodology
3. **Generate** — Produce Jupyter notebooks with scaffolded code, tests, and conceptual questions for each module

## Project Structure

```
scaffoldly/
├── __main__.py     # CLI entry point
├── ingest.py       # Fetch & parse blogs and repos
├── pipeline.py     # 3-stage LLM generation pipeline
├── prompts.py      # Prompt templates for each stage
└── notebook.py     # Jupyter notebook generation
```

## Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY` for course generation (not needed for the demo)

## Acknowledgments

Inspired by [karpathify](https://github.com/nuwandavek/karpathify) and Stanford's [CS231n](https://cs231n.stanford.edu/) assignments.

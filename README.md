# Scaffoldly

Turn technical blog posts, papers, and repos into hands-on coursework you can actually learn from.

Scaffoldly takes expert-level content — deep blog posts, arXiv papers, GitHub repos — and generates progressive courses with MIT-quality lesson documents, scaffolded exercises, and observable milestones. The student works through modules that build toward **reproducing the author's results as faithfully as possible**.

## Quick Start

```bash
uv pip install scaffoldly

scaffoldly
# → opens http://localhost:8420
```

1. Pick a provider (OpenAI, Anthropic, Google, Ollama, OpenRouter) and add your API key
2. Paste a URL, describe your background, hit generate
3. A course appears in your output directory — lesson documents + exercise files

## What Gets Generated

```
output/aliens_trick_from_exactk_dp_to_ioi_2016/
├── README.md                          # Course overview + learning path
├── requirements.txt
├── module_01_count_dimension/
│   ├── README.md                      # 3,000-10,000 word lesson document
│   ├── ex01_exact_k_baseline.py       # Scaffold (student works here)
│   ├── ex02_penalty_sweep.py
│   ├── ex03_binary_search.py
│   └── _solutions/                    # Working solutions (hidden)
│       ├── ex01_exact_k_baseline.py
│       ├── ex02_penalty_sweep.py
│       └── ex03_binary_search.py
├── module_02_calc_lambda/
│   └── ...
└── module_03_ioi_2016/
    └── ...
```

### Lesson Documents (not README summaries)

Each module's README is a **self-contained teaching document** — 3,000-10,000 words:

- Learning objectives and table of contents
- Running example that evolves through the lesson
- Inline code showing concept → code translation
- Embedded comprehension checks at points of friction
- Formula translation: math → plain language → code
- Analytical questions requiring tradeoff reasoning

### Exercise Files

Each exercise has a **scaffold** (student-facing) and a **solution**:

```python
def exact_k_dp(arr, k):
    """Compute maximum sum of exactly k disjoint subarrays.

    Algorithm:
    1. Track two states per position: 'inside' and 'outside' a subarray
    2. Transition rules enforce exactly k non-overlapping segments

    Parameters
    ----------
    arr : list[int], length n
    k : int, number of subarrays

    Returns
    -------
    max_sum : int
    """
    ###########################################################
    # YOUR CODE HERE - 12-16 lines                            #
    #                                                         #
    # Hint: Use dp[i][j][state] where state is 0 (outside)   #
    # or 1 (inside a subarray). Initialize impossible states  #
    # to -infinity.                                           #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################

if __name__ == "__main__":
    arr = [3, -1, 4, -1, 5, -9, 2, 6]
    for k in range(1, 5):
        result = exact_k_dp(arr, k)
        brute = brute_force_exact_k(arr, k)
        match = "OK" if result == brute else "MISMATCH"
        print(f"  k={k}: dp={result}, brute={brute}  [{match}]")
```

The `__main__` block is always fully provided — run the file, see if your implementation works.

## How It Works

### Phase 1: Blueprint (2 API calls, design model)

Reads the full source material and produces a **Blueprint** — a rich contract specifying:
- Curriculum structure with module dependencies
- Scaffold contracts per exercise (what's provided vs what student writes, with line counts)
- Key excerpts: verbatim formulas and algorithms from the source
- Validation criteria: what correct output looks like

### Phase 2: Generate (multi-turn conversation per module, parallel)

Each module gets its own conversation with the full source material:

1. **Write the lesson** — 3,000-10,000 word markdown document (free-form, not JSON)
2. **Write exercises one at a time** — scaffold, then solution, one turn each
3. **Execute solutions** — Python runs each solution, captures output
4. **Feed results forward** — exercise 2's prompt includes exercise 1's actual output

Modules generate in parallel. The lesson-first approach means the model deeply processes the source material before writing any code.

### Phase 3: Review (pre-flight + LLM review)

Python pre-flight checks (syntax, TODO markers, file length, output patterns) catch structural issues. LLM review checks pedagogical quality and contract compliance. Failed modules are re-generated.

## Providers

Works with any LLM provider via [LiteLLM](https://github.com/BerriAI/litellm):

| Provider | Design model default | Generate model default |
|---|---|---|
| OpenAI | gpt-5.4 | gpt-5.4 |
| Anthropic | claude-opus-4-6 | claude-sonnet-4-6 |
| Google | gemini-2.5-pro | gemini-2.5-flash |
| Ollama | llama3 | llama3 |
| OpenRouter | claude-opus-4-6 | claude-sonnet-4-6 |

## Development

```bash
git clone https://github.com/TAOGenna/scaffoldly.git
cd scaffoldly
uv sync
uv run scaffoldly
```

## Acknowledgments

Inspired by [karpathify](https://github.com/nuwandavek/karpathify), Stanford's [CS231n](https://cs231n.stanford.edu/) assignments, and [MIT 6.102](https://web.mit.edu/6.102/www/sp26/) course readings.

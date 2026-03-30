"""Mock LLM client for end-to-end testing without API calls.

Returns realistic pre-canned responses for each pipeline phase.
Tests the full pipeline plumbing: file writing, syntax validation,
code execution, conversation flow, review, web UI events.

Usage:
  Select "Mock (testing)" as provider in the web UI, or pass provider="mock".
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .llm import CompletionResult, Usage

# ── Mock response fixtures ───────────────────────────────────────────────────

_ANALYSIS_JSON = {
    "title": "Mock Course: Binary Search Optimization",
    "summary": "A technique for reducing 2D DP to 1D using penalty binary search.",
    "domain": "competitive programming",
    "content_type": "tutorial",
    "overall_difficulty": "intermediate",
    "key_concepts": [
        {
            "name": "Penalty Binary Search",
            "description": "Binary search on a penalty parameter to reduce exact-k DP to unconstrained DP.",
            "importance": "core",
            "difficulty": "intermediate",
            "source": "focus",
            "priority": "essential",
            "priority_rationale": "The central technique — everything else builds toward it.",
        },
        {
            "name": "Concavity of Cost Function",
            "description": "The cost function f(k) must be concave for penalty search to work.",
            "importance": "core",
            "difficulty": "intermediate",
            "source": "focus",
            "priority": "essential",
            "priority_rationale": "Without concavity, the search is not guaranteed correct.",
        },
        {
            "name": "Convex Hull Trick",
            "description": "Optimization for the inner DP to achieve O(n) per lambda evaluation.",
            "importance": "supporting",
            "difficulty": "advanced",
            "source": "focus",
            "priority": "supporting",
            "priority_rationale": "Speeds up the inner loop but the trick works without it.",
        },
    ],
    "prerequisites": [
        {
            "name": "Dynamic Programming",
            "why_needed": "The technique optimizes a DP recurrence.",
            "difficulty": "intermediate",
        }
    ],
    "code_patterns": [
        {
            "description": "DP with penalty parameter lambda",
            "language": "python",
            "concepts_demonstrated": ["Penalty Binary Search"],
        }
    ],
    "learning_goals": [
        "Implement exact-k DP baseline",
        "Understand how penalty lambda reduces dimensionality",
        "Verify correctness via concavity",
    ],
}

_CURRICULUM_JSON = {
    "curriculum": {
        "course_title": "Mock: Penalty Binary Search",
        "course_description": "Learn the penalty binary search technique for DP optimization.",
        "modules": [
            {
                "module_index": 1,
                "title": "Exact-K DP Baseline",
                "description": "Implement the naive exact-k DP to establish the baseline.",
                "learning_objectives": [
                    "Implement a 2D DP for exactly k disjoint subarrays",
                    "Understand why this is O(n^2 * k)",
                ],
                "concepts_covered": ["Penalty Binary Search"],
                "depends_on": [],
                "exercises": [
                    {
                        "title": "Exact-K Subarray Sum",
                        "type": "implement",
                        "description": "Implement exact-k disjoint subarray maximum sum.",
                        "scaffolding_level": "heavy",
                        "what_is_provided": "import block, prefix_sum helper, brute_force function, __main__ test harness",
                        "what_student_writes": "exact_k_dp(arr, k) — fill the DP recurrence (~10-12 lines)",
                        "milestone": "Prints table comparing DP vs brute force for k=1..4",
                        "key_insight": "The DP tracks inside/outside subarray state transitions",
                        "common_mistakes": "confusing exactly-k with at-most-k; wrong initialization",
                        "expected_output_pattern": "k=",
                    },
                    {
                        "title": "Penalty Sweep",
                        "type": "explore",
                        "description": "Sweep lambda values and observe how the optimal k changes.",
                        "scaffolding_level": "heavy",
                        "what_is_provided": "lambda grid, plotting code, __main__ harness",
                        "what_student_writes": "shifted_score(f_values, k, lam) ~3 lines; best_k(f_values, lam) ~5 lines",
                        "milestone": "Prints lambda → best_k table showing monotone count",
                        "key_insight": "Lambda tilts the answer curve until desired k becomes optimal",
                        "common_mistakes": "wrong sign for penalty; not handling ties",
                        "expected_output_pattern": "lambda",
                    },
                ],
                "inline_questions": [
                    {
                        "question": "When lambda increases, why does the optimal k decrease?",
                        "context": "Think about the penalty subtracted per subarray.",
                    }
                ],
                "visible_outcome": "A table showing exact-k values matching brute force.",
                "key_excerpts": [
                    "dp[i][j][s] = best value using first i elements with exactly j subarrays, s=0 outside, s=1 inside",
                    "The penalized objective is f(k) - lambda * k. Binary search on lambda.",
                ],
            },
            {
                "module_index": 2,
                "title": "Penalty Binary Search",
                "description": "Implement the full penalty binary search technique.",
                "learning_objectives": [
                    "Implement calc(lambda) returning (value, count)",
                    "Binary search on lambda to hit target k",
                ],
                "concepts_covered": ["Penalty Binary Search", "Concavity of Cost Function"],
                "depends_on": [1],
                "exercises": [
                    {
                        "title": "Calc Lambda",
                        "type": "implement",
                        "description": "Implement the relaxed solver that returns value and count.",
                        "scaffolding_level": "medium",
                        "what_is_provided": "Pair dataclass, binary search wrapper, __main__ harness",
                        "what_student_writes": "calc_lambda(arr, lam) — the O(n) penalized DP (~15-18 lines)",
                        "milestone": "Prints (value, count) for several lambda values, matching exact-k",
                        "key_insight": "The count is part of the DP state, not computed after the fact",
                        "common_mistakes": "tracking only value; wrong tie-breaking direction",
                        "expected_output_pattern": "value=",
                    },
                ],
                "inline_questions": [
                    {
                        "question": "What happens if the cost function is not concave?",
                        "context": "The binary search assumes monotone count.",
                    }
                ],
                "visible_outcome": "Binary search finds lambda that yields target k.",
                "key_excerpts": [
                    "calc(lambda) returns (penalized_value, count_used). Binary search adjusts lambda until count == k.",
                ],
            },
        ],
    },
    "shared_definitions": {
        "language": "python",
        "dependencies": ["numpy"],
        "naming_convention": "snake_case",
    },
    "root_readme": "# Mock: Penalty Binary Search\n\nA course on the penalty binary search technique.\n\n## Learning Path\n\n1. Exact-K DP Baseline\n2. Penalty Binary Search\n\n---\n_Generated from mock source on 2026-03-29 by scaffoldly._\n",
    "requirements": "numpy>=1.20",
}

_LESSON_CONTENT = """\
# Exact-K DP Baseline

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [The Problem: Maximum K Disjoint Subarrays](#the-problem)
3. [Building the DP Table](#building-the-dp-table)
4. [The Penalty Intuition](#the-penalty-intuition)
5. [Exercises](#exercises)
6. [Analytical Questions](#analytical-questions)

## Learning Objectives

After completing this module you will be able to:
- Implement a 2D DP for exactly k disjoint subarrays
- Understand why this is O(n^2 * k)
- See how a penalty parameter changes which k is optimal

## The Problem

Given an array of integers, find the maximum sum of exactly k non-overlapping
contiguous subarrays.

**Example:** `arr = [3, -1, 4, -1, 5, -9, 2, 6]`, k=2

The best two subarrays are `[3, -1, 4, -1, 5]` (sum=10) and `[2, 6]` (sum=8),
for a total of 18.

### Why this matters

This is the baseline problem that the penalty binary search technique optimizes.
Before we can speed it up, we need to solve it correctly.

## Building the DP Table

We define `dp[i][j][s]` where:
- `i` = position in the array (0 to n)
- `j` = number of subarrays used so far (0 to k)
- `s` = state: 0 = outside a subarray, 1 = inside

The transitions are:
- **Outside, stay outside:** `dp[i][j][0] = dp[i-1][j][0]`
- **Inside, end subarray:** `dp[i][j][0] = dp[i-1][j][1]`
- **Outside, start subarray:** `dp[i][j+1][1] = dp[i-1][j][0] + arr[i]`
- **Inside, extend:** `dp[i][j][1] = dp[i-1][j][1] + arr[i]`

```python
# Initialize impossible states to -infinity
dp = [[[-float('inf')] * 2 for _ in range(k + 1)] for _ in range(n + 1)]
dp[0][0][0] = 0  # base case: no elements, no subarrays, outside
```

**Check your understanding:** Why do we initialize to negative infinity?
What would go wrong if we initialized to zero?

## The Penalty Intuition

The exact-k DP is O(n * k) per state. For large k, this is expensive.

The key insight: instead of fixing k, we add a **penalty** lambda for each
subarray used. The penalized objective becomes:

```
f_penalized(k) = f(k) - lambda * k
```

When lambda = 0, the optimizer takes as many subarrays as profitable.
When lambda is large, it takes fewer. By binary searching on lambda,
we can find the value that makes exactly k subarrays optimal.

**Check your understanding:** If lambda = 0, how many subarrays does the
optimizer select? What about lambda = infinity?

## Exercises

1. **Exact-K Subarray Sum** — Implement the 2D DP described above
2. **Penalty Sweep** — Sweep lambda and observe how optimal k changes

## Analytical Questions

1. The DP has O(n * k) states. At what point does the penalty approach
   (O(n * log(max_value))) become faster? Express in terms of n and k.

2. What property of f(k) guarantees that binary search on lambda works?
   Can you construct an f(k) where it fails?
"""

_SCAFFOLD_EX01 = '''\
"""Exact-K Subarray Sum: Maximum sum of exactly k disjoint subarrays.

Algorithm:
1. Define dp[i][j][s] tracking position, subarrays used, inside/outside state
2. Transition between states: start, extend, end, or skip
3. Answer is dp[n][k][0] — used exactly k subarrays, ended outside

Parameters
----------
arr : list[int], length n
    The input array.
k : int
    Exact number of non-overlapping subarrays to select.

Returns
-------
max_sum : int
    Maximum possible sum using exactly k subarrays.
"""


def brute_force_exact_k(arr, k):
    """Brute force for small inputs — try all combinations."""
    n = len(arr)
    if k == 0:
        return 0
    best = -float('inf')

    def backtrack(start, remaining, current_sum):
        nonlocal best
        if remaining == 0:
            best = max(best, current_sum)
            return
        for i in range(start, n):
            sub_sum = 0
            for j in range(i, n):
                sub_sum += arr[j]
                backtrack(j + 1, remaining - 1, current_sum + sub_sum)

    backtrack(0, k, 0)
    return best


def exact_k_dp(arr, k):
    """Compute maximum sum of exactly k disjoint subarrays using DP.

    Parameters
    ----------
    arr : list[int], length n
    k : int

    Returns
    -------
    max_sum : int
    """
    n = len(arr)
    ###########################################################
    # YOUR CODE HERE - 10-12 lines                            #
    #                                                         #
    # Define dp[i][j][s] with dimensions (n+1) x (k+1) x 2   #
    # Initialize dp[0][0][0] = 0, everything else = -inf      #
    # Fill using the four transitions from the lesson.         #
    # Return dp[n][k][0]                                       #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


if __name__ == "__main__":
    arr = [3, -1, 4, -1, 5, -9, 2, 6]
    print(f"Array: {arr}")
    print(f"{'k':>3} | {'DP':>6} | {'Brute':>6} | {'Match':>5}")
    print("-" * 30)
    for k in range(1, 5):
        dp_val = exact_k_dp(arr, k)
        brute_val = brute_force_exact_k(arr, k)
        match = "OK" if dp_val == brute_val else "FAIL"
        print(f"k={k} | {dp_val:>6} | {brute_val:>6} | {match:>5}")
    print()
    print(">> If all rows show OK, your DP is correct.")
    print(">> Next: exercise 2 explores how a penalty parameter changes optimal k.")
'''

_SOLUTION_EX01 = '''\
"""Exact-K Subarray Sum: Maximum sum of exactly k disjoint subarrays."""


def brute_force_exact_k(arr, k):
    """Brute force for small inputs — try all combinations."""
    n = len(arr)
    if k == 0:
        return 0
    best = -float('inf')

    def backtrack(start, remaining, current_sum):
        nonlocal best
        if remaining == 0:
            best = max(best, current_sum)
            return
        for i in range(start, n):
            sub_sum = 0
            for j in range(i, n):
                sub_sum += arr[j]
                backtrack(j + 1, remaining - 1, current_sum + sub_sum)

    backtrack(0, k, 0)
    return best


def exact_k_dp(arr, k):
    """Compute maximum sum of exactly k disjoint subarrays using DP."""
    n = len(arr)
    NEG_INF = -float('inf')
    # dp[i][j][s]: best value using arr[:i], j subarrays, s=0 outside / s=1 inside
    dp = [[[NEG_INF] * 2 for _ in range(k + 1)] for _ in range(n + 1)]
    dp[0][0][0] = 0

    for i in range(1, n + 1):
        val = arr[i - 1]
        for j in range(k + 1):
            # Outside: stay outside or end a subarray
            dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1])
            # Inside: extend current subarray
            if dp[i - 1][j][1] != NEG_INF:
                dp[i][j][1] = max(dp[i][j][1], dp[i - 1][j][1] + val)
            # Start new subarray (from outside, costs one subarray slot)
            if j > 0 and dp[i - 1][j - 1][0] != NEG_INF:
                dp[i][j][1] = max(dp[i][j][1], dp[i - 1][j - 1][0] + val)

    return dp[n][k][0]


if __name__ == "__main__":
    arr = [3, -1, 4, -1, 5, -9, 2, 6]
    print(f"Array: {arr}")
    print(f"{'k':>3} | {'DP':>6} | {'Brute':>6} | {'Match':>5}")
    print("-" * 30)
    for k in range(1, 5):
        dp_val = exact_k_dp(arr, k)
        brute_val = brute_force_exact_k(arr, k)
        match = "OK" if dp_val == brute_val else "FAIL"
        print(f"k={k} | {dp_val:>6} | {brute_val:>6} | {match:>5}")
    print()
    print(">> If all rows show OK, your DP is correct.")
    print(">> Next: exercise 2 explores how a penalty parameter changes optimal k.")
'''

_SCAFFOLD_EX02 = '''\
"""Penalty Sweep: Observe how lambda changes the optimal k.

Given precomputed f(k) values, sweep lambda and find which k
maximizes f(k) - lambda * k for each lambda.
"""


def shifted_score(f_values, k, lam):
    """Compute f(k) - lambda * k.

    Parameters
    ----------
    f_values : list[int], indexed by k
    k : int
    lam : float

    Returns
    -------
    score : float
    """
    ###########################################################
    # YOUR CODE HERE - 2-3 lines                              #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def best_k_for_lambda(f_values, lam):
    """Find k that maximizes shifted_score.

    Parameters
    ----------
    f_values : list[int]
    lam : float

    Returns
    -------
    best_k : int
    """
    ###########################################################
    # YOUR CODE HERE - 5-7 lines                              #
    #                                                         #
    # Hint: iterate over all k, compute shifted_score,        #
    # track the argmax. Break ties by smallest k.             #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


if __name__ == "__main__":
    # Precomputed f(k) for arr = [3, -1, 4, -1, 5, -9, 2, 6]
    f_values = [0, 18, 18, 15, 14]  # f(0)=0, f(1)=10+8=18, etc.

    lambdas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    print(f"{'lambda':>8} | {'best_k':>6} | {'score':>8}")
    print("-" * 30)
    for lam in lambdas:
        k = best_k_for_lambda(f_values, lam)
        score = shifted_score(f_values, k, lam)
        print(f"lambda={lam:>4.1f} | k={k:>4} | {score:>8.1f}")
    print()
    print(">> Notice: as lambda increases, best_k decreases (monotone).")
    print(">> This monotonicity is what makes binary search on lambda work.")
'''

_SOLUTION_EX02 = '''\
"""Penalty Sweep: Observe how lambda changes the optimal k."""


def shifted_score(f_values, k, lam):
    """Compute f(k) - lambda * k."""
    return f_values[k] - lam * k


def best_k_for_lambda(f_values, lam):
    """Find k that maximizes shifted_score."""
    best_score = -float('inf')
    best_k = 0
    for k in range(len(f_values)):
        score = shifted_score(f_values, k, lam)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


if __name__ == "__main__":
    f_values = [0, 18, 18, 15, 14]

    lambdas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    print(f"{'lambda':>8} | {'best_k':>6} | {'score':>8}")
    print("-" * 30)
    for lam in lambdas:
        k = best_k_for_lambda(f_values, lam)
        score = shifted_score(f_values, k, lam)
        print(f"lambda={lam:>4.1f} | k={k:>4} | {score:>8.1f}")
    print()
    print(">> Notice: as lambda increases, best_k decreases (monotone).")
    print(">> This monotonicity is what makes binary search on lambda work.")
'''

_REVIEW_PASS = {"module_index": 1, "verdict": "pass", "issues": []}


# ── Mock LLM Client ─────────────────────────────────────────────────────────


class MockLLMClient:
    """Drop-in replacement for LLMClient that returns pre-canned responses.

    Tracks conversation turns to return the right fixture for each phase.
    Zero API calls, zero cost, instant responses.
    """

    def __init__(self, **kwargs):
        self.provider = "mock"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_calls = 0
        self._turn_counters: dict[str, int] = {}  # per-conversation turn tracking

    async def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        response_model: type | None = None,
        max_tokens: int = 16384,
        max_retries: int = 2,
        temperature: float | None = None,
    ) -> CompletionResult:
        self.total_calls += 1
        self.total_input_tokens += 100
        self.total_output_tokens += 50

        # Structured output (Phase 1 + Phase 3)
        if response_model is not None:
            return self._structured_response(response_model, messages)

        # Raw completion (Phase 2 conversational)
        return self._conversational_response(messages, system)

    def _structured_response(self, response_model: type, messages: list[dict]) -> CompletionResult:
        """Return pre-canned structured responses for Phase 1 and 3."""
        from .schemas import Analysis, CurriculumDesign, ModuleReview

        model_name = response_model.__name__ if hasattr(response_model, '__name__') else str(response_model)

        if response_model is Analysis or model_name == "Analysis":
            obj = Analysis(**_ANALYSIS_JSON)
            return CompletionResult(
                content=obj.model_dump_json(indent=2),
                structured=obj,
                usage=Usage(input_tokens=100, output_tokens=50),
            )
        elif response_model is CurriculumDesign or model_name == "CurriculumDesign":
            obj = CurriculumDesign(**_CURRICULUM_JSON)
            return CompletionResult(
                content=obj.model_dump_json(indent=2),
                structured=obj,
                usage=Usage(input_tokens=100, output_tokens=50),
            )
        elif response_model is ModuleReview or model_name == "ModuleReview":
            obj = ModuleReview(**_REVIEW_PASS)
            return CompletionResult(
                content=obj.model_dump_json(indent=2),
                structured=obj,
                usage=Usage(input_tokens=100, output_tokens=50),
            )
        else:
            return CompletionResult(content="{}", usage=Usage())

    def _conversational_response(self, messages: list[dict], system: str | None) -> CompletionResult:
        """Return pre-canned responses for Phase 2 conversation turns."""
        # Count user messages to determine which turn we're on
        user_msgs = [m for m in messages if m["role"] == "user"]
        turn = len(user_msgs)

        # Determine which module by checking for module index in first user message
        first_msg = user_msgs[0]["content"] if user_msgs else ""
        module_id = "1"  # default

        # Turn 1 = lesson, Turn 2 = scaffold ex1, Turn 3 = solution ex1, etc.
        if turn == 1:
            content = _LESSON_CONTENT
        elif turn == 2:
            content = _SCAFFOLD_EX01
        elif turn == 3:
            content = _SOLUTION_EX01
        elif turn == 4:
            # "Execution acknowledged" turn — the model says "ready"
            content = "Understood. Ready for the next exercise."
        elif turn == 5:
            content = _SCAFFOLD_EX02
        elif turn == 6:
            content = _SOLUTION_EX02
        else:
            content = "Understood."

        return CompletionResult(
            content=content,
            usage=Usage(input_tokens=100, output_tokens=50),
        )

    def get_totals(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": 0.0,
            "api_calls": self.total_calls,
        }

    def __repr__(self) -> str:
        return "MockLLMClient(provider='mock')"

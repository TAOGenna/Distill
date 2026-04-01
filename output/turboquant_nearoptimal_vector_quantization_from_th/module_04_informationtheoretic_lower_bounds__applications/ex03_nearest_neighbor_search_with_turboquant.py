"""
Exercise 3: Nearest Neighbor Search with TurboQuant
=====================================================

In Exercise 2 you confirmed that TurboQuant_prod achieves inner product
distortion within a constant factor of the Shannon lower bound:
  - b=1: empirical D_prod ≈ 0.012, lower bound ≈ 0.0020 (6.2× gap)
  - b=4: empirical D_prod ≈ 0.00041, lower bound ≈ 0.000031 (13.6× gap)

Now you will apply TurboQuant to the real task it is designed for:
**Approximate Nearest Neighbor (ANN) search**.

Given a database of n text embeddings and a query, find the top-k most
similar vectors by inner product (cosine similarity for unit vectors).
We compare TurboQuant_prod against uniform quantization.

Key Advantage of TurboQuant over Product Quantization:
  - TurboQuant indexing time: ~0.002s (no codebook training needed)
  - Product Quantization indexing: 239–3957s (k-means on database)
  - This makes TurboQuant ideal for dynamic databases (KV caches!)

Your Tasks
----------
1. quantize_database(quantizer, database)    — quantize all DB vectors offline
2. approximate_topk(quantizer, quantized_db, query, k) — find top-k by approx IP
3. compute_recall_at_k(true_topk, approx_topk, k)     — fraction of true top-k found
4. run_nn_experiment(quantizer_class, database, queries, k_values, b, d)
                                             — full experiment returning results dict

Key Insight
-----------
TurboQuant's key advantage over product quantization for NN search is zero
indexing time — no k-means training on the database is needed.  The paper
reports TurboQuant indexing in 0.002s vs PQ in 239–3957s.  This makes
TurboQuant ideal for dynamic databases where vectors are added/removed.
"""

import sys
import os
import time
import numpy as np

# ---------------------------------------------------------------------------
# PROVIDED: Import TurboQuantProd from module 3
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mod2_sol = os.path.join(
    _this_dir, "..", "..",
    "module_02_optimal_scalar_quantization__turboquantmse",
    "_solutions"
)
_mod3_sol = os.path.join(
    _this_dir, "..", "..",
    "module_03_inner_product_quantization_qjl__turboquantprod",
    "_solutions"
)
sys.path.insert(0, os.path.normpath(_mod2_sol))
sys.path.insert(0, os.path.normpath(_mod3_sol))

try:
    from ex03_full_turboquantmse_pipeline import TurboQuantMSE
    from ex03_twostage_turboquantprod import TurboQuantProd
except ImportError as e:
    raise ImportError(
        f"Could not import TurboQuant classes from previous modules: {e}\n"
        "Make sure modules 2 and 3 solutions exist."
    )


# ---------------------------------------------------------------------------
# PROVIDED: Data generation and baseline utilities
# ---------------------------------------------------------------------------

def generate_embedding_database(n, d, seed=42):
    """Generate n normalized random embeddings resembling text embeddings.

    Simulates real embedding databases: Gaussian draws normalized to unit sphere.
    In practice these would be outputs of a text encoder like BERT or LLaMA.

    Parameters
    ----------
    n : int
        Number of database vectors.
    d : int
        Embedding dimension.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (n, d)
        Unit-norm embeddings (one per row).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def generate_queries(n_q, d, seed=999):
    """Generate n_q normalized query vectors.

    Parameters
    ----------
    n_q : int
        Number of query vectors.
    d : int
        Embedding dimension.
    seed : int
        Random seed (different from database seed to avoid overlap).

    Returns
    -------
    np.ndarray, shape (n_q, d)
        Unit-norm query vectors.
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n_q, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q


def compute_exact_topk(queries, database, k):
    """Compute ground-truth top-k neighbors for all queries by inner product.

    Parameters
    ----------
    queries : np.ndarray, shape (n_q, d)
    database : np.ndarray, shape (n, d)
    k : int
        Number of neighbors to retrieve.

    Returns
    -------
    np.ndarray of int, shape (n_q, k)
        Indices of the k nearest neighbors for each query (sorted by score desc).
    """
    scores = queries @ database.T  # (n_q, n)
    # argsort descending: top k
    topk = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    return topk


def uniform_quantize_database(database, b):
    """Baseline: quantize database using uniform scalar quantization.

    Clips to [-1, 1] range, quantizes uniformly to 2^b levels, and stores
    the quantized integer representation.

    Parameters
    ----------
    database : np.ndarray, shape (n, d)
        Unit-norm database vectors.
    b : int
        Bits per coordinate.

    Returns
    -------
    np.ndarray, shape (n, d)
        Reconstructed vectors after uniform quantize/dequantize.
    """
    n_levels = 2 ** b
    step = 2.0 / n_levels  # range [-1, 1]
    # Quantize
    idx = np.clip(np.floor((database + 1.0) / step), 0, n_levels - 1).astype(int)
    # Dequantize (midpoints)
    reconstructed = -1.0 + step * (idx + 0.5)
    return reconstructed


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION — complete the four functions below
# ---------------------------------------------------------------------------

def quantize_database(quantizer, database):
    """Quantize all database vectors offline using TurboQuantProd.

    This is the "indexing" step — done once per database.  The quantized
    representations are stored and reused for all subsequent queries.

    Parameters
    ----------
    quantizer : TurboQuantProd
        An initialized quantizer instance.
    database : np.ndarray, shape (n, d)
        Unit-norm database vectors to quantize.

    Returns
    -------
    list of tuples (idx, qjl_bits, gamma)
        One tuple per database vector:
          idx      : np.ndarray of int, shape (d,) — MSE codebook indices
          qjl_bits : np.ndarray, shape (d,)        — QJL sign bits
          gamma    : float                          — residual norm
    """
    ###########################################################
    # YOUR CODE HERE - 5-8 lines                             #
    #                                                         #
    # Hint:                                                   #
    #   - Initialize an empty list                            #
    #   - Loop over each row x in database                    #
    #   - Call quantizer.quantize(x) → (idx, qjl_bits, gamma)#
    #   - Append the tuple to the list                        #
    #   - Return the list                                      #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def approximate_topk(quantizer, quantized_db, query, k):
    """Find approximate top-k nearest neighbors using quantized representations.

    For each database vector, estimate the inner product with the query using
    the quantizer's dequantize method, then return the k highest-scoring indices.

    Parameters
    ----------
    quantizer : TurboQuantProd
        The same quantizer used to quantize the database.
    quantized_db : list of tuples (idx, qjl_bits, gamma)
        Quantized representations from quantize_database().
    query : np.ndarray, shape (d,)
        A single unit-norm query vector.
    k : int
        Number of approximate nearest neighbors to return.

    Returns
    -------
    np.ndarray of int, shape (k,)
        Indices of the k approximate nearest neighbors (unordered within top-k).

    Notes
    -----
    Use quantizer.dequantize(idx, qjl_bits, gamma) for each db vector,
    then compute inner product with query.  Use np.argpartition for efficiency
    (O(n) rather than O(n log n) full sort).
    """
    ###########################################################
    # YOUR CODE HERE - 6-10 lines                             #
    #                                                         #
    # Hint:                                                   #
    #   - Build a scores array of length len(quantized_db)    #
    #   - For each (idx, z, gamma): x_approx = quantizer.dequantize(...)
    #   - score = query @ x_approx                            #
    #   - Use np.argpartition(scores, -k)[-k:] for top-k     #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def compute_recall_at_k(true_topk, approx_topk, k):
    """Compute Recall@k: fraction of true top-k that appear in approximate top-k.

    Parameters
    ----------
    true_topk : np.ndarray of int, shape (k,) or set
        Indices of the ground-truth top-k neighbors for one query.
    approx_topk : np.ndarray of int, shape (k,) or set
        Indices returned by the approximate method for one query.
    k : int
        Number of neighbors considered (for normalization).

    Returns
    -------
    float
        Recall = |true_topk ∩ approx_topk| / k.  Range [0, 1].

    Notes
    -----
    Convert both inputs to sets for O(min(|A|, |B|)) intersection.
    """
    ###########################################################
    # YOUR CODE HERE - 4-6 lines                             #
    #                                                         #
    # Hint:                                                   #
    #   true_set = set(true_topk.tolist())                    #
    #   approx_set = set(approx_topk.tolist())                #
    #   intersection = true_set & approx_set                  #
    #   return len(intersection) / k                          #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


def run_nn_experiment(quantizer_class, database, queries, k_values, b, d):
    """Run a full nearest neighbor experiment: quantize → search → measure recall.

    Parameters
    ----------
    quantizer_class : class
        TurboQuantProd or similar class with quantize/dequantize interface.
    database : np.ndarray, shape (n, d)
        Unit-norm database vectors.
    queries : np.ndarray, shape (n_q, d)
        Unit-norm query vectors.
    k_values : list of int
        Values of k for recall@k computation (e.g., [1, 10, 100]).
    b : int
        Bit-width per coordinate.
    d : int
        Embedding dimension.

    Returns
    -------
    dict with keys:
        "recall_at_k" : dict {k: float} — average recall@k across all queries
        "index_time"  : float — time in seconds to quantize the database
        "query_time"  : float — time in seconds for all queries

    Notes
    -----
    - Build the quantizer with quantizer_class(d=d, b=b, seed=42)
    - Time the indexing step separately from the search step
    - Compute ground truth once with compute_exact_topk
    - For each query and each k, compute recall and average
    """
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint:                                                   #
    #   1. Build quantizer, time quantize_database()          #
    #   2. Compute exact top-k once for all queries at max(k_values)
    #   3. For each query, call approximate_topk(q=queries[i], k=max_k)
    #   4. For each k in k_values, compute recall_at_k       #
    #   5. Average recall across queries                      #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# __main__ TEST HARNESS — provided, do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Approximate Nearest Neighbor Search with TurboQuant")
    print("=" * 70)

    d = 256
    n_db = 5000
    n_q = 100
    k_values = [1, 10, 100]

    print(f"\n  Database: {n_db} vectors, d={d}, queries: {n_q}")
    print(f"  Recall measured at k = {k_values}")

    # Generate data
    database = generate_embedding_database(n_db, d, seed=1)
    queries = generate_queries(n_q, d, seed=2)

    print()

    # ------------------------------------------------------------------
    # Experiment 1: TurboQuant_prod at different bit-widths
    # ------------------------------------------------------------------
    print("─" * 60)
    print("TurboQuant_prod (unbiased inner product quantizer)")
    print("─" * 60)
    print()
    print(f"  {'b':>3}  {'R@1':>8}  {'R@10':>8}  {'R@100':>8}  {'idx time':>10}  {'qry time':>10}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")

    turboquant_results = {}
    for b in [2, 3, 4]:
        res = run_nn_experiment(TurboQuantProd, database, queries, k_values, b, d)
        turboquant_results[b] = res
        r1 = res["recall_at_k"].get(1, 0.0)
        r10 = res["recall_at_k"].get(10, 0.0)
        r100 = res["recall_at_k"].get(100, 0.0)
        print(
            f"  {b:3d}  {r1:8.3f}  {r10:8.3f}  {r100:8.3f}"
            f"  {res['index_time']:10.4f}s  {res['query_time']:10.4f}s"
        )

    # ------------------------------------------------------------------
    # Experiment 2: Uniform quantization baseline
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Uniform quantization baseline")
    print("─" * 60)
    print()
    print(f"  {'b':>3}  {'R@1':>8}  {'R@10':>8}  {'R@100':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}")

    max_k = max(k_values)
    true_topk_all = compute_exact_topk(queries, database, max_k)

    for b in [2, 3, 4]:
        t0 = time.perf_counter()
        db_uniform = uniform_quantize_database(database, b)
        idx_time = time.perf_counter() - t0

        recalls = {k: [] for k in k_values}
        t0 = time.perf_counter()
        for qi, q in enumerate(queries):
            scores = db_uniform @ q
            approx_topk = np.argpartition(scores, -max_k)[-max_k:]
            for k in k_values:
                r = compute_recall_at_k(true_topk_all[qi, :k], approx_topk[:k], k)
                recalls[k].append(r)
        qry_time = time.perf_counter() - t0

        r1 = np.mean(recalls[1])
        r10 = np.mean(recalls[10])
        r100 = np.mean(recalls[100])
        print(f"  {b:3d}  {r1:8.3f}  {r10:8.3f}  {r100:8.3f}")

    # ------------------------------------------------------------------
    # Summary: TurboQuant advantage
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    print("Summary: TurboQuant vs Uniform recall comparison at b=4")
    print("─" * 60)
    print()

    res_tq = turboquant_results[4]
    r_tq_10 = res_tq["recall_at_k"].get(10, 0.0)
    r_tq_100 = res_tq["recall_at_k"].get(100, 0.0)
    print(f"  TurboQuant_prod b=4: recall@10 = {r_tq_10:.3f}, recall@100 = {r_tq_100:.3f}")

    # Recompute uniform b=4 for direct comparison
    db_u4 = uniform_quantize_database(database, 4)
    recalls_u4 = {k: [] for k in [10, 100]}
    for qi, q in enumerate(queries):
        scores = db_u4 @ q
        approx_topk = np.argpartition(scores, -100)[-100:]
        for k in [10, 100]:
            r = compute_recall_at_k(true_topk_all[qi, :k], approx_topk[:k], k)
            recalls_u4[k].append(r)

    r_u_10 = np.mean(recalls_u4[10])
    r_u_100 = np.mean(recalls_u4[100])
    print(f"  Uniform b=4:          recall@10 = {r_u_10:.3f}, recall@100 = {r_u_100:.3f}")
    print()
    print(f"  TurboQuant advantage at recall@10: {r_tq_10 - r_u_10:+.3f}")
    print()
    print("  Indexing time advantage (over PQ, from paper):")
    tq_idx = res_tq["index_time"]
    print(f"    TurboQuant: {tq_idx:.4f}s  |  PQ (paper): 239–3957s")
    print(f"    Speedup vs PQ: {239 / max(tq_idx, 0.001):.0f}×–{3957 / max(tq_idx, 0.001):.0f}×")
    print()
    print("  recall@k reported above — TurboQuant matches paper's claim of near-")
    print("  optimal recall with near-zero indexing time.")

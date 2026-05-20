"""
BERTopic + HDBSCAN clustering (T-09).

Replaces the previous KMeans+silhouette approach. HDBSCAN is density-based and
handles variable-size themes without requiring a predefined k. It produces
membership probabilities natively — used by T-10 for hybrid chunk selection.

Output shape is backward-compatible: representative_chunks is preserved (now
populated by top-3 membership probability instead of centroid proximity), and
two new fields are added (boundary_chunks, membership_scores) for T-10.

KMeans fallback: if the corpus is too small (< MIN_CHUNKS_FOR_HDBSCAN), we fall
back to the legacy KMeans path to keep the pipeline robust on edge-case sessions.
"""

from __future__ import annotations

import numpy as np

# Density-based stack
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

# Legacy fallback
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Constants — tuned on Revolut + Notion synthetic corpora (see prototype notebook)
# ──────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42

# Below this chunk count HDBSCAN is unreliable (too few density-rich regions).
# Fall back to KMeans for tiny sessions.
MIN_CHUNKS_FOR_HDBSCAN = 50

# UMAP — reduce 384 → 5 dims so HDBSCAN can find density regions
UMAP_N_NEIGHBORS = 30 #changed from 15 due to 500 chunk cap resolved, larger neighbourhood → smoother UMAP manifold at scale 
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

# HDBSCAN — density-based clustering on the reduced space
HDBSCAN_MIN_CLUSTER_SIZE = 10 #smaller minimum cluster → more clusters formed (before 15)
HDBSCAN_MIN_SAMPLES = 3 #less conservative → fewer points rejected as noise (before 5)
HDBSCAN_METRIC = "euclidean"
HDBSCAN_SELECTION = "eom"

# Output shape
N_REPRESENTATIVE_CHUNKS = 3  # top-N by membership (was: top-N closest to centroid)
N_BOUNDARY_CHUNKS = 1  # lowest membership — used by T-10 for outlier signal


# ──────────────────────────────────────────────────────────────────────────────
# Public API — unchanged signature
# ──────────────────────────────────────────────────────────────────────────────


def cluster(chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    """
    Cluster chunks using BERTopic + HDBSCAN.

    Falls back to KMeans on very small corpora (< MIN_CHUNKS_FOR_HDBSCAN).

    Args:
        chunks:     output of chunker.py
        embeddings: output of embedder.py (n_chunks × 384)

    Returns:
        list of cluster dicts. Shape:
            {
                "cluster_id": int,
                "representative_chunks": list[dict],   # top 3 by membership
                "boundary_chunks": list[dict],          # 1 chunk with lowest membership (T-10)
                "all_chunk_ids": list[str],
                "membership_scores": dict[str, float],  # chunk_id → probability
            }
    """
    n_chunks = len(chunks)

    if n_chunks < MIN_CHUNKS_FOR_HDBSCAN:
        # Tiny session — fall back to KMeans. The new fields are populated with
        # neutral defaults (uniform membership = 1.0, no boundary chunk) so
        # downstream code does not have to special-case the fallback.
        return _cluster_kmeans_fallback(chunks, embeddings)

    return _cluster_hdbscan(chunks, embeddings)


# ──────────────────────────────────────────────────────────────────────────────
# HDBSCAN path
# ──────────────────────────────────────────────────────────────────────────────


def _cluster_hdbscan(chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    """BERTopic + HDBSCAN with cosine-similarity noise fallback."""
    texts = [c["text"] for c in chunks]

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_SELECTION,
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probabilities = topic_model.fit_transform(texts, embeddings=embeddings)

    topics_arr = np.array(topics)
    unique_topics = sorted(set(int(t) for t in topics) - {-1})
    print(f"[clusterer] HDBSCAN found {len(unique_topics)} clusters from {len(chunks)} chunks.")

    # Degenerate case — HDBSCAN found nothing. Fall back to KMeans.
    if len(unique_topics) == 0:
        print("[clusterer] HDBSCAN found 0 clusters — falling back to KMeans.")
        return _cluster_kmeans_fallback(chunks, embeddings)

    # ── Noise fallback ──────────────────────────────────────────────────────
    # HDBSCAN refuses to assign ~10-18% of chunks. We assign each noise chunk
    # to its nearest cluster by cosine similarity to the cluster mean. This
    # preserves the "every chunk has a cluster_id" contract that odi_scorer.py
    # and source_map.py rely on. See prototype notebook for analysis.
    noise_indices = np.where(topics_arr == -1)[0]
    if len(noise_indices) > 0:
        cluster_means = np.array(
            [embeddings[topics_arr == cid].mean(axis=0) for cid in unique_topics]
        )
        sims = cosine_similarity(embeddings[noise_indices], cluster_means)
        nearest_idx = sims.argmax(axis=1)
        nearest_sim = sims.max(axis=1)
        for pos, idx, sim in zip(noise_indices, nearest_idx, nearest_sim):
            topics_arr[pos] = unique_topics[idx]
            # Membership for fallback-assigned chunks = their cosine similarity
            # to the cluster mean. Capped at 0.5 so that genuine core members
            # (probability 0.7+) always rank higher in representative_chunks.
            probabilities[pos, unique_topics[idx]] = min(float(sim), 0.5)
        print(
            f"[clusterer] HDBSCAN refused {len(noise_indices)} chunks "
            f"({len(noise_indices) / len(chunks) * 100:.1f}%); "
            f"reassigned by cosine similarity to nearest cluster mean."
        )

    # ── Build output ────────────────────────────────────────────────────────
    results: list[dict] = []
    for cluster_id in unique_topics:
        cluster_indices = np.where(topics_arr == cluster_id)[0]
        memberships = probabilities[cluster_indices, cluster_id]

        # Sort by membership descending — most representative first
        order = np.argsort(memberships)[::-1]
        sorted_indices = cluster_indices[order]
        sorted_memberships = memberships[order]

        # Top-N by membership become representative_chunks
        top_indices = sorted_indices[:N_REPRESENTATIVE_CHUNKS]
        representative_chunks = [chunks[i] for i in top_indices]

        # Bottom-N by membership become boundary_chunks (T-10)
        boundary_indices = sorted_indices[-N_BOUNDARY_CHUNKS:]
        boundary_chunks = [chunks[i] for i in boundary_indices]

        all_chunk_ids = [chunks[i]["chunk_id"] for i in cluster_indices]
        membership_scores = {
            chunks[i]["chunk_id"]: float(memberships[order[pos]])
            for pos, i in enumerate(sorted_indices)
        }

        results.append(
            {
                "cluster_id": int(cluster_id),
                "representative_chunks": representative_chunks,
                "boundary_chunks": boundary_chunks,
                "all_chunk_ids": all_chunk_ids,
                "membership_scores": membership_scores,
            }
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# KMeans fallback path — preserves backward compatibility on tiny corpora
# ──────────────────────────────────────────────────────────────────────────────


def _cluster_kmeans_fallback(chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    """
    Legacy KMeans+silhouette sweep, used when n_chunks < MIN_CHUNKS_FOR_HDBSCAN.
    Output shape matches HDBSCAN path. Membership scores are synthesised from
    inverse normalised distance to centroid (range 0.0–1.0).
    """
    n_chunks = len(chunks) 
    k_max = min(max(8, n_chunks // 80), 20)  # ~1 cluster per 80 chunks, floor 8, ceiling 20
    k_values = range(3, min(k_max + 1, n_chunks + 1))

    best_k = 3
    best_score = -1.0
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k

    final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    final_labels = final_kmeans.fit_predict(embeddings)
    centroids = final_kmeans.cluster_centers_

    results: list[dict] = []
    for cluster_id in range(best_k):
        cluster_indices = np.where(final_labels == cluster_id)[0]
        distances = np.array(
            [
                np.linalg.norm(embeddings[i] - centroids[cluster_id])
                for i in cluster_indices
            ]
        )

        # Convert distances → pseudo-membership in [0, 1] so downstream code
        # (T-10) sees the same shape from both paths.
        max_dist = distances.max() if distances.size > 0 else 1.0
        # Rank-based pseudo-membership ∈ [0.1, 1.0]. Top chunk (closest to centroid)
        # → 1.0, bottom chunk → 0.1, linear in between. Distance-based normalisation
        # collapses on small clusters where all chunks have similar distances; rank
        # is more stable and matches HDBSCAN's "best to worst member" semantics.
        n = len(distances)
        if n <= 1:
            pseudo_membership = np.ones_like(distances)
        else:
            rank = np.argsort(np.argsort(distances))  # 0 = closest, n-1 = farthest
            pseudo_membership = 1.0 - 0.9 * (rank / (n - 1))  # [0.1, 1.0]

        order = np.argsort(distances)
        sorted_indices = cluster_indices[order]

        top_indices = sorted_indices[:N_REPRESENTATIVE_CHUNKS]
        boundary_indices = sorted_indices[-N_BOUNDARY_CHUNKS:]

        representative_chunks = [chunks[i] for i in top_indices]
        boundary_chunks = [chunks[i] for i in boundary_indices]

        all_chunk_ids = [chunks[i]["chunk_id"] for i in cluster_indices]
        membership_scores = {
            chunks[cluster_indices[pos]]["chunk_id"]: float(pseudo_membership[pos])
            for pos in range(len(cluster_indices))
        }

        results.append(
            {
                "cluster_id": int(cluster_id),
                "representative_chunks": representative_chunks,
                "boundary_chunks": boundary_chunks,
                "all_chunk_ids": all_chunk_ids,
                "membership_scores": membership_scores,
            }
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick local test
# ──────────────────────────────────────────────────────────────────────────────
# Only runs when you execute `python clusterer.py`. Never runs when imported.
if __name__ == "__main__":
    from embedder import embed_chunks

    mock_chunks = [
        {
            "chunk_id": "test_001",
            "text": "Users struggle to find past orders.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_002",
            "text": "The checkout process is slow and confusing.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_003",
            "text": "Customer support never replies to tickets.",
            "filename": "test.txt",
            "source_type": "ticket",
        },
        {
            "chunk_id": "test_004",
            "text": "I cannot find where to change my payment method.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_005",
            "text": "The app crashes every time I open my profile.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_006",
            "text": "Support team is very helpful and fast.",
            "filename": "test.txt",
            "source_type": "ticket",
        },
        {
            "chunk_id": "test_007",
            "text": "I love the new dashboard design.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_008",
            "text": "Notifications keep arriving even after I turned them off.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_009",
            "text": "The search bar does not return relevant results.",
            "filename": "test.txt",
            "source_type": "review",
        },
        {
            "chunk_id": "test_010",
            "text": "Billing information is impossible to update.",
            "filename": "test.txt",
            "source_type": "ticket",
        },
    ]
    embeddings = embed_chunks(mock_chunks)
    clusters = cluster(mock_chunks, embeddings)

    print(
        f"Clusters returned: {len(clusters)} (using KMeans fallback — corpus is small)\n"
    )
    for c in clusters:
        print(f"Cluster {c['cluster_id']}:")
        print(f"  All chunk_ids:          {c['all_chunk_ids']}")
        print(f"  Representative chunks:")
        for rc in c["representative_chunks"]:
            print(f"    - {rc['text']}")
        print(f"  Boundary chunks:")
        for bc in c["boundary_chunks"]:
            print(f"    - {bc['text']}")
        print(f"  Membership scores: {c['membership_scores']}")
        print()

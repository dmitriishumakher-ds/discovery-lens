"""
odi_scorer.py
Computes deterministic priority signals per cluster.
No LLM. No external API. Pure sentiment scoring + cluster metadata.

Input:
    clusters       — output of clusterer.py
    chunks         — output of chunker.py (used to resolve text, source_type, sentiment)
    goal_embedding — 384-dim embedding of the validated goal string (np.ndarray)
                     Pass None to skip goal_relevance (scores as 1.0, no dampening).

Output:
    list of scored cluster dicts with four independent scores per cluster:

    odi_score           — classic ODI: importance x (1 - satisfaction)
                          "How underserved is this need?"

    evidence_robustness — cross-source corroboration signal:
                          (source_type_diversity x 0.65) + (importance x 0.35)
                          "How robustly is this theme evidenced across source types?"

    goal_relevance      — cosine similarity between goal embedding and mean cluster
                          chunk embeddings (membership-weighted where available).
                          "How directly does this cluster address the stated goal?"

    priority_score      — synthesis:
                          [(odi_score x 0.60) + (evidence_robustness x 0.40)]
                          x max(goal_relevance, 0.20)
                          "What should a PM act on first?"

    recommendation      — quadrant label derived from odi_score and evidence_robustness:
                          "Act" | "Validate" | "Monitor" | "Deprioritise"
                          Thresholds: ODI_THRESHOLD=0.10, EVIDENCE_THRESHOLD=0.40
                          PM sign-off: Lucas, May 14 2026 (D-02).

See docs/data_contracts.md for the full contract.

Sentiment model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
Replaced VADER May 13 2026 — benchmark in notebooks/sentiment_benchmark_lucas.ipynb


Changelog:
    May 19 2026 — _cluster_goal_relevance now uses membership-weighted mean
                  (D-03 gap closed). T-09 membership_scores passed from clusterer.
    May 14 2026 — Added goal_relevance (D-03), multiplicative priority_score dampening,
                  and recommendation labels (D-02). 
    May 13 2026 — Replaced VADER with lxyuan (T-08). 
    Apr 29 2026 — Replaced opportunity_score with three-score system.
"""

from __future__ import annotations

import numpy as np
from transformers import pipeline as hf_pipeline
from typing import Any

# Lazy-loaded sentiment model. @st.cache_resource keeps the loaded model in
# memory across Streamlit reruns — without it, every user interaction would
# reload the ~250MB lxyuan model from disk. Falls back to a plain singleton
# when Streamlit isn't available (e.g. when this module is used from a notebook).
# framework="tf" required on Apple Silicon (arm64) to avoid PyTorch bus error.
try:
    import streamlit as st
    @st.cache_resource
    def _get_sentiment_pipeline():
        """Load the lxyuan sentiment-analysis pipeline once per Streamlit session."""
        return hf_pipeline(
            "sentiment-analysis",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            framework="tf",
            truncation=True,
            max_length=512,
        )
except ImportError:
    _sentiment_singleton = None
    def _get_sentiment_pipeline():
        """Load the lxyuan sentiment-analysis pipeline once per Python process."""
        global _sentiment_singleton
        if _sentiment_singleton is None:
            _sentiment_singleton = hf_pipeline(
                "sentiment-analysis",
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                framework="tf",
                truncation=True,
                max_length=512,
            )
        return _sentiment_singleton

# ── Weights ────────────────────────────────────────────────────────────────────
# Confirmed stable by T-16 sensitivity analysis (May 14 2026).
# Grid: ±0.10 in 0.05 steps, 75 runs across 3 datasets.
# Min tau=0.9048, mean tau=0.9947. All variants above 0.90. PM sign-off: Lucas.
_DIVERSITY_WEIGHT = 0.65  # evidence_robustness: source_type_diversity component
_SIZE_WEIGHT = 0.35       # evidence_robustness: normalised cluster size component
_ODI_WEIGHT = 0.60        # priority_score: ODI component
_EVIDENCE_WEIGHT = 0.40   # priority_score: evidence robustness component

# goal_relevance multiplicative dampening floor (D-03, May 14 2026).
# Prevents a cluster from being zeroed out by low goal_relevance —
# it stays visible in the UI but ranked lower.
# Floor value subject to revalidation after T-09 (BERTopic + HDBSCAN) lands.
_GOAL_RELEVANCE_FLOOR = 0.20

# ── D-02 recommendation thresholds ────────────────────────────────────────────
# Calibrated against observed score distributions across Lidl, Asana, Revolut
# synthetic datasets (May 14 2026). PM sign-off: Lucas.
# Re-evaluate when real PM data is loaded — absolute ranges will shift.
ODI_THRESHOLD = 0.10       # odi_score cutoff for Act / Validate vs Monitor / Deprioritise
EVIDENCE_THRESHOLD = 0.40  # evidence_robustness cutoff for Act / Monitor vs Validate / Deprioritise

# ── Source-type enum ──────────────────────────────────────────────────────────
# Expanded May 13 2026: 4 → 6 types. Kept in sync with chunker.py / extractor.py.
# Used as FIXED denominator for source_type_diversity — see module docstring for why.
KNOWN_SOURCE_TYPES: tuple[str, ...] = (
    "interview",
    "review",
    "ticket",
    "usability",
    "social",
    "internal",
)
KNOWN_SOURCE_TYPES_COUNT: int = len(KNOWN_SOURCE_TYPES)


# ── Sentiment helpers ─────────────────────────────────────────────────────────

def _score_to_compound(result: dict) -> float:
    """
    Convert lxyuan output dict to a continuous score in [-1, 1].

    lxyuan returns {"label": "positive"|"negative"|"neutral", "score": float}.
    Mapping:
        positive → +score   (confidence as positive signal, range  0 to +1)
        negative → -score   (confidence as negative signal, range -1 to  0)
        neutral  →  0.0     (no directional signal)

    The resulting value is averaged across chunks in a cluster and rescaled
    to [0, 1] via (x + 1) / 2 to produce the satisfaction signal.
    """
    label = result["label"].lower()
    score = result["score"]
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:  # neutral
        return 0.0


def _batch_sentiment(texts: list[str], batch_size: int = 8) -> list[float]:
    """
    Run lxyuan on a list of texts in batches.
    Returns a list of compound scores in range -1…1.
    """
    compounds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = _get_sentiment_pipeline()(batch)
        compounds.extend(_score_to_compound(r) for r in results)
    return compounds


# ── Goal relevance helper ─────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.
    Returns 0.0 if either vector is zero-magnitude.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cluster_goal_relevance(
    chunk_ids: list[str],
    goal_embedding: np.ndarray,
    chunk_embedding_map: dict[str, np.ndarray],
    membership_scores: dict[str, float] | None = None,
) -> float:
    """
    Compute goal_relevance as membership-weighted mean cosine similarity
    between the goal embedding and each chunk's embedding in the cluster.

    Membership-weighted mean (D-03). T-09 merged May 14 — membership_scores
    are now available and used here. Falls back to unweighted mean if
    membership_scores is None or empty.

    Returns a value in range 0.0-1.0 (cosine similarity is clipped to 0
    for negative values, which indicate orthogonal/opposite direction).
    Returns 1.0 if goal_embedding is None (no dampening applied).
    """
    if goal_embedding is None:
        return 1.0

    sims = []
    weights = []
    for cid in chunk_ids:
        emb = chunk_embedding_map.get(cid)
        if emb is not None:
            sim = max(_cosine_similarity(goal_embedding, emb), 0.0)
            w = membership_scores.get(cid, 1.0) if membership_scores else 1.0
            sims.append(sim)
            weights.append(w)

    if not sims:
        return 1.0

    weights_arr = np.array(weights)
    total = weights_arr.sum()
    if total == 0:
        return float(np.mean(sims))
    return float(np.dot(sims, weights_arr) / total)


# ── Recommendation label ──────────────────────────────────────────────────────

def _recommendation_label(odi_score: float, evidence_robustness: float) -> str:
    """
    Assign a quadrant recommendation label based on D-02 thresholds.

    Quadrants (PM sign-off: Lucas, May 14 2026):
        Act          — high unmet need, well-evidenced → prioritise for roadmap
        Validate     — high unmet need, thin evidence  → run more research first
        Monitor      — low unmet need, well-evidenced  → keep on radar
        Deprioritise — low unmet need, thin evidence   → not worth roadmap space now

    Thresholds:
        ODI_THRESHOLD      = 0.10
        EVIDENCE_THRESHOLD = 0.40
    """
    high_odi      = odi_score >= ODI_THRESHOLD
    high_evidence = evidence_robustness >= EVIDENCE_THRESHOLD

    if high_odi and high_evidence:
        return "Act"
    elif high_odi and not high_evidence:
        return "Validate"
    elif not high_odi and high_evidence:
        return "Monitor"
    else:
        return "Deprioritise"


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_clusters(
    clusters: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    goal_embedding: np.ndarray | None = None,
    chunk_embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Score each cluster using four independent signals and assign a
    recommendation label.

    --- ODI score (unmet need signal) ---
    importance   = cluster_size / total_chunks          range 0-1
    satisfaction = (avg_sentiment_compound + 1) / 2    range 0-1
                   where avg_sentiment_compound = mean of _score_to_compound()
                   across all chunks in the cluster (lxyuan output, not VADER)
    odi_score    = importance x (1 - satisfaction)      range 0-1

    --- Evidence robustness (cross-source corroboration) ---
    source_type_diversity = unique source types in cluster / 6 (fixed denominator)
                            range 0-1
    evidence_robustness   = (diversity x 0.65) + (importance x 0.35)

    --- Goal relevance (on-topic signal) ---
    goal_relevance = mean cosine_sim(goal_embedding, chunk_embedding)
                     per chunk in cluster, clipped to [0, 1].
                     Returns 1.0 (no dampening) if goal_embedding is None.

    --- Priority score (synthesis) ---
    priority_score = [(odi_score x 0.60) + (evidence_robustness x 0.40)]
                     x max(goal_relevance, 0.20)

    --- Recommendation label (D-02) ---
    "Act" | "Validate" | "Monitor" | "Deprioritise"
    Based on ODI_THRESHOLD=0.10 and EVIDENCE_THRESHOLD=0.40.

    Args:
        clusters:         output of clusterer.py
        chunks:           output of chunker.py
        goal_embedding:   384-dim np.ndarray of the validated goal (optional).
                          If None, goal_relevance=1.0 and no dampening is applied.
        chunk_embeddings: np.ndarray of shape (n_chunks, 384), same index order
                          as chunks (optional, required for goal_relevance).
                          If None, goal_relevance defaults to 1.0.

    Returns:
        list of scored cluster dicts, sorted by priority_score descending.
    """
    if not chunks:
        raise ValueError("chunks list is empty — cannot compute scores")

    # Build lookups from chunk_id → text and chunk_id → source_type
    chunk_text_map: dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_source_map: dict[str, str] = {c["chunk_id"]: c["source_type"] for c in chunks}

    # Build chunk_id → embedding lookup if embeddings are provided
    chunk_embedding_map: dict[str, np.ndarray] = {}
    if chunk_embeddings is not None and goal_embedding is not None:
        for i, chunk in enumerate(chunks):
            if i < len(chunk_embeddings):
                chunk_embedding_map[chunk["chunk_id"]] = chunk_embeddings[i]

    total_chunks: int = len(chunks)
    scored: list[dict[str, Any]] = []

    for cluster in clusters:
        cluster_id: int = cluster["cluster_id"]
        chunk_ids: list[str] = cluster["all_chunk_ids"]
        cluster_size: int = len(chunk_ids)

        # ------------------------------------------------------------------ #
        # ODI SCORE                                                            #
        # ------------------------------------------------------------------ #

        importance: float = cluster_size / total_chunks

        texts_in_cluster = [
            chunk_text_map[cid] for cid in chunk_ids if cid in chunk_text_map
        ]

        if texts_in_cluster:
            compound_scores = _batch_sentiment(texts_in_cluster)
            avg_sentiment: float = sum(compound_scores) / len(compound_scores)
        else:
            avg_sentiment = 0.0  # neutral fallback

        satisfaction: float = (avg_sentiment + 1) / 2
        odi_score: float = importance * (1 - satisfaction)

        # ------------------------------------------------------------------ #
        # EVIDENCE ROBUSTNESS                                                  #
        # ------------------------------------------------------------------ #

        cluster_source_types: set[str] = {
            chunk_source_map[cid] for cid in chunk_ids if cid in chunk_source_map
        }
        source_type_diversity: float = len(cluster_source_types) / KNOWN_SOURCE_TYPES_COUNT

        evidence_robustness: float = (
            (source_type_diversity * _DIVERSITY_WEIGHT) +
            (importance * _SIZE_WEIGHT)
        )

        # ------------------------------------------------------------------ #
        # GOAL RELEVANCE                                                       #
        # ------------------------------------------------------------------ #

        goal_relevance: float = _cluster_goal_relevance(
            chunk_ids,
            goal_embedding,
            chunk_embedding_map,
            membership_scores=cluster.get("membership_scores"),
        )

        # ------------------------------------------------------------------ #
        # PRIORITY SCORE (with goal_relevance dampening)                      #
        # ------------------------------------------------------------------ #

        base_score: float = (odi_score * _ODI_WEIGHT) + (evidence_robustness * _EVIDENCE_WEIGHT)
        priority_score: float = base_score * max(goal_relevance, _GOAL_RELEVANCE_FLOOR)

        # ------------------------------------------------------------------ #
        # RECOMMENDATION LABEL (D-02)                                         #
        # ------------------------------------------------------------------ #

        recommendation: str = _recommendation_label(odi_score, evidence_robustness)

        scored.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                # Raw signals
                "importance": round(importance, 4),
                "avg_sentiment": round(avg_sentiment, 4),
                "satisfaction": round(satisfaction, 4),
                "source_type_diversity": round(source_type_diversity, 4),
                # Four scores
                "odi_score": round(odi_score, 4),
                "evidence_robustness": round(evidence_robustness, 4),
                "goal_relevance": round(goal_relevance, 4),
                "priority_score": round(priority_score, 4),
                # Recommendation label
                "recommendation": recommendation,
            }
        )

    # Sort highest priority first — convenient for results page
    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored
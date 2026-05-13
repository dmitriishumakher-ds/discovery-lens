"""
odi_scorer.py
Computes deterministic priority signals per cluster.
No LLM. No external API. Pure sentiment scoring + cluster metadata.

Input:
    clusters  — output of clusterer.py
    chunks    — output of chunker.py (used to resolve text, source_type, sentiment)

Output:
    list of scored cluster dicts with three independent scores per cluster:

    odi_score           — classic ODI: importance × (1 - satisfaction)
                          "How underserved is this need?"

    evidence_robustness — cross-source corroboration signal:
                          (source_type_diversity × 0.65) + (normalised_cluster_size × 0.35)
                          "How robustly is this theme evidenced across source types?"

    priority_score      — synthesis of both:
                          (odi_score × 0.6) + (evidence_robustness × 0.4)
                          "What should a PM act on first?"

See docs/data_contracts.md for the full contract.

Sentiment model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
Replaced VADER May 13 2026 — benchmark in notebooks/sentiment_benchmark_lucas.ipynb
PM sign-off: Lucas.
"""

from transformers import pipeline as hf_pipeline
from typing import Any

# Initialise once at module level — avoids reloading model weights on every call
# framework="tf" required on Apple Silicon (arm64) to avoid PyTorch bus error
_sentiment = hf_pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    framework="tf",
    truncation=True,
    max_length=512,
)

# Weights — defined as constants so they are easy to tune and review
_DIVERSITY_WEIGHT = 0.65       # evidence_robustness: source_type_diversity component
_SIZE_WEIGHT = 0.35            # evidence_robustness: normalised cluster size component
_ODI_WEIGHT = 0.60             # priority_score: ODI component
_EVIDENCE_WEIGHT = 0.40        # priority_score: evidence robustness component

# Fixed denominator — total recognised source types in the enum
# Updated May 13 2026: expanded from 4 to 6 source types
# interview, review, ticket, usability, social, internal
_TOTAL_SOURCE_TYPES = 6


def _score_to_compound(result: dict) -> float:
    """
    Convert lxyuan output to a -1…1 compound score analogous to VADER.
    positive → +score, negative → -score, neutral → 0
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
        results = _sentiment(batch)
        compounds.extend(_score_to_compound(r) for r in results)
    return compounds


def score_clusters(
    clusters: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Score each cluster using three independent signals.

    --- ODI score (unmet need signal) ---
    importance   = cluster_size / total_chunks          range 0–1
    satisfaction = (avg_sentiment_compound + 1) / 2    range 0–1
    odi_score    = importance × (1 - satisfaction)      range 0–1

    --- Evidence robustness (cross-source corroboration) ---
    source_type_diversity = unique source types in cluster / 6 (fixed denominator)
                            range 0–1
    evidence_robustness   = (diversity × 0.65) + (importance × 0.35)

    --- Priority score (synthesis) ---
    priority_score = (odi_score × 0.60) + (evidence_robustness × 0.40)

    Args:
        clusters: output of clusterer.py
        chunks:   output of chunker.py

    Returns:
        list of scored cluster dicts, sorted by priority_score descending
    """
    if not chunks:
        raise ValueError("chunks list is empty — cannot compute scores")

    # Build lookups from chunk_id → text and chunk_id → source_type
    chunk_text: dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_source: dict[str, str] = {c["chunk_id"]: c["source_type"] for c in chunks}

    total_chunks: int = len(chunks)

    scored: list[dict[str, Any]] = []

    for cluster in clusters:
        cluster_id: int = cluster["cluster_id"]
        chunk_ids: list[str] = cluster["all_chunk_ids"]
        cluster_size: int = len(chunk_ids)

        # ------------------------------------------------------------------ #
        # ODI SCORE                                                            #
        # ------------------------------------------------------------------ #

        # Importance — how large is this cluster relative to all chunks?
        importance: float = cluster_size / total_chunks

        # Collect texts for all chunks in this cluster
        texts_in_cluster = [
            chunk_text[cid] for cid in chunk_ids if cid in chunk_text
        ]

        # Sentiment — lxyuan compound score per chunk, then average
        if texts_in_cluster:
            compound_scores = _batch_sentiment(texts_in_cluster)
            avg_sentiment: float = sum(compound_scores) / len(compound_scores)
        else:
            avg_sentiment = 0.0  # neutral fallback if no text found

        # Satisfaction — normalise -1…1 → 0…1
        satisfaction: float = (avg_sentiment + 1) / 2

        # ODI score — high when need is large AND poorly satisfied
        odi_score: float = importance * (1 - satisfaction)

        # ------------------------------------------------------------------ #
        # EVIDENCE ROBUSTNESS                                                  #
        # ------------------------------------------------------------------ #

        # How many distinct source types appear in this cluster?
        cluster_source_types: set[str] = {
            chunk_source[cid] for cid in chunk_ids if cid in chunk_source
        }
        unique_in_cluster: int = len(cluster_source_types)

        # Fixed denominator of 6 — total recognised source types in the enum
        source_type_diversity: float = unique_in_cluster / _TOTAL_SOURCE_TYPES

        evidence_robustness: float = (
            (source_type_diversity * _DIVERSITY_WEIGHT)
            + (importance * _SIZE_WEIGHT)
        )

        # ------------------------------------------------------------------ #
        # PRIORITY SCORE                                                       #
        # ------------------------------------------------------------------ #

        priority_score: float = (
            (odi_score * _ODI_WEIGHT)
            + (evidence_robustness * _EVIDENCE_WEIGHT)
        )

        scored.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "importance": round(importance, 4),
                "avg_sentiment": round(avg_sentiment, 4),
                "satisfaction": round(satisfaction, 4),
                "source_type_diversity": round(source_type_diversity, 4),
                "odi_score": round(odi_score, 4),
                "evidence_robustness": round(evidence_robustness, 4),
                "priority_score": round(priority_score, 4),
            }
        )

    # Sort highest priority first — convenient for results page
    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored

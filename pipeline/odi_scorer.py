"""
odi_scorer.py
Computes deterministic ODI priority signals per cluster.
No LLM. No external API. Pure VADER + cluster metadata.

Input:
    clusters  — output of clusterer.py
    chunks    — output of chunker.py (used to resolve text + compute sentiment)

Output:
    list of scored cluster dicts (see data contract in docs/data_contracts.md)
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Any

# Initialise once at module level — avoids reloading the lexicon on every call
_vader = SentimentIntensityAnalyzer()


def score_clusters(
    clusters: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Score each cluster with three independent signals per the Apr 29 redesign.

    importance            = cluster_size / total_chunks
    satisfaction          = (avg_vader_compound + 1) / 2
    source_type_diversity = unique source types in cluster / total unique source types
    odi_score             = importance * (1 - satisfaction)
    evidence_robustness   = (source_type_diversity * 0.65) + (importance * 0.35)
    priority_score        = (odi_score * 0.60) + (evidence_robustness * 0.40)

    Args:
        clusters: output of clusterer.py
        chunks:   output of chunker.py

    Returns:
        list of scored cluster dicts sorted by priority_score descending
    """
    chunk_text_map: dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_source_map: dict[str, str] = {c["chunk_id"]: c["source_type"] for c in chunks}
    total_chunks: int = len(chunks)

    if total_chunks == 0:
        raise ValueError("chunks list is empty — cannot compute importance scores")

    total_source_types: int = len({c["source_type"] for c in chunks})

    scored: list[dict[str, Any]] = []

    for cluster in clusters:
        cluster_id: int = cluster["cluster_id"]
        chunk_ids: list[str] = cluster["all_chunk_ids"]
        cluster_size: int = len(chunk_ids)

        # --- Importance ---
        importance: float = cluster_size / total_chunks

        # --- Sentiment (VADER compound per chunk, then average) ---
        compound_scores: list[float] = []
        for cid in chunk_ids:
            text = chunk_text_map.get(cid)
            if text:
                compound_scores.append(_vader.polarity_scores(text)["compound"])

        avg_sentiment: float = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0

        # --- Satisfaction (normalise VADER -1…1 → 0…1) ---
        satisfaction: float = (avg_sentiment + 1) / 2

        # --- Source type diversity ---
        source_types_in_cluster = {
            chunk_source_map[cid] for cid in chunk_ids if cid in chunk_source_map
        }
        source_type_diversity: float = (
            len(source_types_in_cluster) / total_source_types
            if total_source_types > 0 else 0.0
        )

        # --- Three scores ---
        odi_score: float = importance * (1 - satisfaction)
        evidence_robustness: float = (source_type_diversity * 0.65) + (importance * 0.35)
        priority_score: float = (odi_score * 0.60) + (evidence_robustness * 0.40)

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

    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored

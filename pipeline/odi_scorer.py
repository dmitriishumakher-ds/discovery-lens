"""
odi_scorer.py
Computes deterministic priority signals per cluster.
No LLM. No external API. Pure VADER + cluster metadata.

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
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Any

# Initialise once at module level — avoids reloading the lexicon on every call
_vader = SentimentIntensityAnalyzer()

# Weights — defined as constants so they are easy to tune and review
_DIVERSITY_WEIGHT = 0.65  # evidence_robustness: source_type_diversity component
_SIZE_WEIGHT = 0.35  # evidence_robustness: normalised cluster size component
_ODI_WEIGHT = 0.60  # priority_score: ODI component
_EVIDENCE_WEIGHT = 0.40  # priority_score: evidence robustness component

# Source-type enum recognised by the system.
# Kept in sync with the source_type contract in chunker.py / extractor.py.
# Used as a FIXED denominator for source_type_diversity so the metric is:
#   - stable across sessions (does not shrink when the user uploads fewer source types)
#   - honest when only one source type is present (will not collapse to 1.0)
#   - comparable across runs and PMs
KNOWN_SOURCE_TYPES: tuple[str, ...] = ("interview", "review", "ticket", "usability")
KNOWN_SOURCE_TYPES_COUNT: int = len(KNOWN_SOURCE_TYPES)


def score_clusters(
    clusters: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Score each cluster using three independent signals.

    --- ODI score (unmet need signal) ---
    importance   = cluster_size / total_chunks          range 0–1
    satisfaction = (avg_vader_compound + 1) / 2        range 0–1
    odi_score    = importance × (1 - satisfaction)      range 0–1

    --- Evidence robustness (cross-source corroboration) ---
    source_type_diversity = unique source types in cluster / total unique source types
                            range 0–1
    normalised_size       = cluster_size / total_chunks (same as importance)
    evidence_robustness   = (diversity × 0.65) + (normalised_size × 0.35)

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

        # Sentiment — VADER compound per chunk, then average across cluster
        compound_scores: list[float] = []
        for cid in chunk_ids:
            text = chunk_text.get(cid)
            if text:
                compound_scores.append(_vader.polarity_scores(text)["compound"])

        avg_sentiment: float = (
            sum(compound_scores) / len(compound_scores)
            if compound_scores
            else 0.0  # neutral fallback if no text found
        )

        # Satisfaction — normalise VADER -1…1 → 0…1
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

        # Normalise against the FIXED count of source types the system recognises,
        # not against the unique types present in the current upload.
        # See KNOWN_SOURCE_TYPES_COUNT at the top of this module for why.
        source_type_diversity: float = unique_in_cluster / KNOWN_SOURCE_TYPES_COUNT

        # Normalised size is the same value as importance — explicit for clarity
        normalised_size: float = importance

        evidence_robustness: float = (source_type_diversity * _DIVERSITY_WEIGHT) + (
            normalised_size * _SIZE_WEIGHT
        )

        # ------------------------------------------------------------------ #
        # PRIORITY SCORE                                                       #
        # ------------------------------------------------------------------ #

        priority_score: float = (odi_score * _ODI_WEIGHT) + (
            evidence_robustness * _EVIDENCE_WEIGHT
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

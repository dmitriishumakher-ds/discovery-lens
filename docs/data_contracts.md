# Discovery Lens — Data Contracts

Source of truth for all pipeline module I/O. Last updated: Apr 29, 2026.
If a contract needs to change, open a GitHub Issue and tag Lucas first.

---

## Pipeline flow

```
UploadedFile → extractor.py → raw_text (str)
→ chunker.py → chunks (list[dict])
→ embedder.py → embeddings (np.ndarray, n_chunks × 384)
→ clusterer.py → clusters (list[dict])
→ source_map.py → source_map (dict) → st.session_state["source_map"]
→ odi_scorer.py → scored_clusters (list[dict]) → st.session_state["scored_clusters"]
→ llm.py → ost (dict) → st.session_state["ost"]
```

---

## extractor.py

```python
# Input
file: UploadedFile   # Streamlit UploadedFile object
source_type: str     # one of: "interview" | "review" | "ticket" | "usability" | "social" | "internal"

# Output
raw_text: str        # full extracted text, plain string
```

---

## chunker.py

```python
# Input
raw_text: str
filename: str
source_type: str     # same enum as extractor.py

# Output — list of dicts, one per chunk
[
  {
    "chunk_id": str,        # format: "{safe_filename}_{zero_padded_index}" e.g. "interview_01_001"
    "text": str,            # 2–4 sentences
    "filename": str,
    "source_type": str
  },
  ...
]
```

---

## embedder.py

```python
# Input
chunks: list[dict]   # output of chunker.py

# Output
embeddings: np.ndarray   # shape: (n_chunks, 384)
```

Notes: `chunks[i]` and `embeddings[i]` share the same index — never reorder independently.

---

## clusterer.py

```python
# Input
chunks: list[dict]        # output of chunker.py
embeddings: np.ndarray    # output of embedder.py

# Output — list of dicts, one per cluster
[
  {
    "cluster_id": int,
    "representative_chunks": list[dict],   # top 3 chunks closest to centroid (full chunk dicts)
    "all_chunk_ids": list[str]             # all chunk_ids belonging to this cluster
  },
  ...
]
```

---

## source_map.py

```python
# Input
chunks: list[dict]    # output of chunker.py
clusters: list[dict]  # output of clusterer.py

# Output — flat dict for chunk-level traceability
{
  "<chunk_id>": {
    "text": str,
    "filename": str,
    "source_type": str,
    "cluster_id": int | None   # None if chunk was not assigned to any cluster
  },
  ...
}
```

Notes:
- Must be called after clusterer.py and before llm.py.
- Stored in st.session_state["source_map"].
- Used by the results page to show source quotes per opportunity.
- No LLM, no external API. Pure dict construction.

---

## odi_scorer.py

```python
# Input
clusters: list[dict]   # output of clusterer.py
chunks: list[dict]     # output of chunker.py
# total_chunks and total_source_types are derived internally — no extra args needed

# Output — list of dicts, one per cluster, sorted by priority_score descending
[
  {
    "cluster_id": int,
    "cluster_size": int,
    # --- Raw signals (available for UI display and debugging) ---
    "importance": float,             # cluster_size / total_chunks, range 0.0–1.0
    "avg_sentiment": float,          # lxyuan compound mean (positive→+score, negative→-score, neutral→0), range -1.0 to 1.0
    "satisfaction": float,           # (avg_sentiment + 1) / 2, range 0.0–1.0
    "source_type_diversity": float,  # unique source types in cluster / KNOWN_SOURCE_TYPES_COUNT (=4), range 0.0–1.0
    # --- Three scores shown independently in UI ---
    "odi_score": float,              # importance * (1 - satisfaction) — unmet need signal, range 0.0–1.0
    "evidence_robustness": float,    # (source_type_diversity * 0.65) + (importance * 0.35), range 0.0–1.0
    "priority_score": float          # (odi_score * 0.60) + (evidence_robustness * 0.40), range 0.0–1.0
  },
  ...
]
```

Notes:
- Deterministic — no LLM, no external API. Uses lxyuan/distilbert-base-multilingual-cased-sentiments-student compound scores per chunk averaged per cluster. Replaced VADER May 13 2026.
- `opportunity_score` retired Apr 29 2026. Replaced by three independent scores. PM sign-off: Lucas.
- Sort key is `priority_score` descending.
- `source_type_diversity` uses a fixed denominator `KNOWN_SOURCE_TYPES_COUNT = 6` (the size of the source_type enum: interview, review, ticket, usability, social, internal — expanded May 13 2026, see docs/decisions.md). This makes the metric stable across single-source and multi-source sessions — a session with only one source type caps at 0.1667 (1/6), honestly reflecting weak cross-source evidence. PM sign-off: Lucas, May 13 2026.

### Score definitions

| Score | Formula | What it answers |
|-------|---------|-----------------|
| `odi_score` | `importance × (1 - satisfaction)` | How underserved is this need? |
| `evidence_robustness` | `(source_type_diversity × 0.65) + (importance × 0.35)` | How robustly evidenced across source types? |
| `priority_score` | `(odi_score × 0.60) + (evidence_robustness × 0.40)` | What should a PM act on first? |

---

## llm.py

```python
# Input
clusters: list[dict]          # output of clusterer.py
scored_clusters: list[dict]   # output of odi_scorer.py
goal: str                     # from st.session_state["goal"]

# LLM generates via Groq — JTBD and solutions only, no score fields
{
  "goal": str,
  "opportunities": [
    {
      "jtbd": str,                    # strictly: "When I [situation], I want to [motivation], so I can [outcome]."
      "job_type": str,                # "functional" | "emotional" | "social" — LLM-generated, not injected
      "jtbd_confidence": str,         # "high" | "medium" | "low" — LLM-generated; overridden to "low" for clusters with <= 3 chunks
      "jtbd_confidence_reason": str,  # one sentence — LLM-generated; deterministic for overridden clusters (states chunk count)
      "cluster_id": int,
      "solutions": [
        {
          "label": str,
          "assumptions": [
            {
              "text": str,
              "risk": str   # "low" | "medium" | "high"
            }
          ]
        }
      ]
    }
  ]
}

# After parsing, llm.py merges scored_clusters on cluster_id to produce the final OST:
{
  "goal": str,
  "opportunities": [
    {
      "jtbd": str,
      "job_type": str,                # passed through from LLM
      "jtbd_confidence": str,         # passed through from LLM, or "low" if overridden
      "jtbd_confidence_reason": str,  # passed through from LLM, or chunk-count sentence if overridden
      "cluster_id": int,
      # --- Injected from scored_clusters, never LLM-generated ---
      "importance": float | None,
      "satisfaction": float | None,
      "source_type_diversity": float | None,
      "odi_score": float | None,
      "evidence_robustness": float | None,
      "priority_score": float | None,
      "solutions": [...]
    }
  ]
}
```

Notes:
- Score fields are **never generated by the LLM** — injected post-parse by merging with `scored_clusters` on `cluster_id`.
- `job_type` **is** LLM-generated and passed through unchanged. Valid values: `functional` | `emotional` | `social`. Rubric in `prompts/system_prompt.txt` Rule 8. PM sign-off: Lucas (May 14 2026).
- If a `cluster_id` from the LLM has no match in `scored_clusters`, set all score fields to `null` — do not crash.
- Always instruct the model to return only valid JSON — no preamble, no markdown fences.
- On JSON parse or validation failure, retry once with `llama-3.1-8b-instant` (fallback).

---

## session_state keys

```python
st.session_state["goal"]            # str — product goal statement
st.session_state["product_name"]    # str — product name
st.session_state["chunks"]          # list[dict] — output of chunker.py
st.session_state["embeddings"]      # np.ndarray — output of embedder.py
st.session_state["clusters"]        # list[dict] — output of clusterer.py
st.session_state["scored_clusters"] # list[dict] — output of odi_scorer.py
st.session_state["ost"]             # dict — merged OST JSON (LLM output + injected scores)
st.session_state["source_map"]      # dict — chunk_id → {text, filename, source_type, cluster_id}
```

Notes: `scored_clusters` must be populated **before** `llm.py` is called so the merge step can do a simple dict lookup without a second pass through raw data.

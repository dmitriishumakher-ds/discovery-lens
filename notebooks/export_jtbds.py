"""
V-C JTBD Quality Check — export script
=======================================
Run from the discovery-lens project root:

    python notebooks/export_jtbds.py --product revolut --goal "..."
    python notebooks/export_jtbds.py --product asana   --goal "..."

Writes: notebooks/jtbds_{product}_{git_short_sha}.json

The JSON file has the format:
{
  "product": "revolut",
  "goal": "...",
  "git_sha": "abc1234",
  "jtbds": [
    {
      "rank": 1,
      "cluster_id": 2,
      "jtbd": "When I ...",
      "job_type": "functional",
      "jtbd_confidence": "high",
      "jtbd_confidence_reason": "...",
      "priority_score": 0.44,
      "odi_score": 0.31,
      "evidence_robustness": 0.64,
      "cluster_size": 42,
      "source_types": ["interview", "review", "ticket"]
    },
    ...
  ]
}

Run on main (post-T-10), then check out eebe554 and run again to get
the pre-T-10 baseline. Compare the two JSON files for blind rating.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks
from pipeline.clusterer import cluster
from pipeline.odi_scorer import score_clusters
from pipeline.llm import build_ost

# ── Source type inference (mirrors bertopic_hdbscan_prototype.ipynb) ──────────

def infer_source_type(filename: str) -> str:
    name = filename.lower()
    if "interview" in name:
        return "interview"
    if "review" in name:
        return "review"
    if "ticket" in name:
        return "ticket"
    if "usability" in name:
        return "usability"
    if "social" in name or "reddit" in name:
        return "social"
    if "internal" in name or "sales" in name or "cs_notes" in name:
        return "internal"
    raise ValueError(
        f"Cannot infer source_type from filename: {filename}\n"
        "Rename the file or pass --source-type explicitly."
    )

# ── Corpus loader ─────────────────────────────────────────────────────────────

def load_corpus(directory: Path) -> list[dict]:
    import pandas as pd
    chunks = []
    files = sorted(directory.glob("*.txt")) + sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found in {directory}")
    for filepath in files:
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
            text_cols = df.select_dtypes(include="str").columns.tolist()
            if not text_cols:
                print(f"  {filepath.name}: skipped (no text columns)")
                continue
            lines = []
            for _, row in df[text_cols].iterrows():
                cells = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            raw_text = "\n".join(lines)
        else:
            raw_text = filepath.read_text(encoding="utf-8")
        source_type = infer_source_type(filepath.name)
        file_chunks = chunk_text(
            raw_text=raw_text,
            filename=filepath.name,
            source_type=source_type,
        )
        chunks.extend(file_chunks)
        print(f"  {filepath.name} ({source_type}): {len(file_chunks)} chunks")
    return chunks

# ── Git SHA helper ────────────────────────────────────────────────────────────

def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export JTBDs for V-C quality check")
    parser.add_argument(
        "--product",
        required=True,
        help="Product name, must match a folder under data/synthetic/ (e.g. revolut, asana)",
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="Product goal string to pass to the pipeline",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top JTBDs to export (default: 10)",
    )
    args = parser.parse_args()

    product = args.product.lower()
    data_dir = PROJECT_ROOT / "data" / "synthetic" / product

    if not data_dir.exists():
        raise FileNotFoundError(
            f"No data directory found at {data_dir}\n"
            f"Expected: data/synthetic/{product}/*.txt"
        )

    sha = git_short_sha()
    output_path = PROJECT_ROOT / "notebooks" / f"jtbds_{product}_{sha}.json"

    print(f"\n=== V-C JTBD Export ===")
    print(f"Product:  {product}")
    print(f"Git SHA:  {sha}")
    print(f"Data dir: {data_dir}")
    print(f"Output:   {output_path}")
    print()

    # 1. Load + chunk
    print("Step 1/5 — Chunking...")
    chunks = load_corpus(data_dir)
    print(f"  Total chunks: {len(chunks)}")

    # 2. Embed
    print("Step 2/5 — Embedding...")
    embeddings = embed_chunks(chunks)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 3. Cluster
    print("Step 3/5 — Clustering...")
    clusters = cluster(chunks, embeddings)
    print(f"  Clusters found: {len(clusters)}")

    # 4. Score — pass goal embedding for goal_relevance
    print("Step 4/5 — Scoring...")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    goal_embedding = _model.encode([args.goal], convert_to_numpy=True)[0]
    import inspect
    _sc_params = inspect.signature(score_clusters).parameters
    if "goal_embedding" in _sc_params:
        scored = score_clusters(
            clusters=clusters,
            chunks=chunks,
            goal_embedding=goal_embedding,
            chunk_embeddings=embeddings,
        )
    else:
        scored = score_clusters(clusters=clusters, chunks=chunks)

    # 5. Build OST
    print("Step 5/5 — Calling LLM (build_ost)...")
    import inspect
    _bo_params = inspect.signature(build_ost).parameters
    if "context_block" in _bo_params:
        ost = build_ost(
            clusters=clusters,
            scored_clusters=scored,
            goal=args.goal,
            context_block="",
        )
    else:
        ost = build_ost(
            clusters=clusters,
            scored_clusters=scored,
            goal=args.goal,
        )

    # ── Extract JTBDs ─────────────────────────────────────────────────────────

    # Build a quick lookup: cluster_id → source_types present
    cluster_source_types: dict[int, list[str]] = {}
    for chunk in chunks:
        cid = next(
            (c["cluster_id"] for c in clusters if chunk["chunk_id"] in c["all_chunk_ids"]),
            None,
        )
        if cid is not None:
            cluster_source_types.setdefault(cid, [])
            if chunk["source_type"] not in cluster_source_types[cid]:
                cluster_source_types[cid].append(chunk["source_type"])

    # Sort opportunities by priority_score descending (should already be sorted,
    # but be explicit so the export is reproducible regardless of OST sort order)
    opps = sorted(
        ost.get("opportunities", []),
        key=lambda o: o.get("priority_score") or 0,
        reverse=True,
    )

    jtbds = []
    for rank, opp in enumerate(opps[: args.top], start=1):
        cid = opp.get("cluster_id")
        jtbds.append({
            "rank": rank,
            "cluster_id": cid,
            "jtbd": opp.get("jtbd", ""),
            "job_type": opp.get("job_type", ""),
            "jtbd_confidence": opp.get("jtbd_confidence", ""),
            "jtbd_confidence_reason": opp.get("jtbd_confidence_reason", ""),
            "priority_score": opp.get("priority_score"),
            "odi_score": opp.get("odi_score"),
            "evidence_robustness": opp.get("evidence_robustness"),
            "goal_relevance": opp.get("goal_relevance"),
            "cluster_size": next(
                (s["cluster_size"] for s in scored if s["cluster_id"] == cid), None
            ),
            "source_types": sorted(cluster_source_types.get(cid, [])),
        })

    result = {
        "product": product,
        "goal": args.goal,
        "git_sha": sha,
        "jtbds": jtbds,
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✓ Exported {len(jtbds)} JTBDs → {output_path}")
    print("\nJTBDs (preview):")
    for j in jtbds:
        print(f"  [{j['rank']}] cluster_{j['cluster_id']} | {j['jtbd'][:90]}...")


if __name__ == "__main__":
    main()

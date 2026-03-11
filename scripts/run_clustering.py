#!/usr/bin/env python3
"""
Run consensus clustering on aligned triples.
"""

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.clustering.consensus import (
    SemanticAwareConsensusClustering,
    compute_inter_triple_similarities,
)
from src.clustering.evaluation import evaluate_clustering
from src.clustering.utils import (
    extract_relation_labels_from_ontology,
    build_full_ground_truth,
    normalize_embeddings,
)


# ============================================================================
# Helpers
# ============================================================================

def _save_json(obj, path):
    """Serialize an object to JSON, converting numpy scalars automatically."""

    def _convert(o):
        if isinstance(o, dict):
            return {str(k): _convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_convert(i) for i in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, "w") as f:
        json.dump(_convert(obj), f, indent=2)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_clustering(
    run_dir: str,
    output_dir: str = None,
    similarity_threshold: float = 0.60,
    n_consensus_runs: int = 3,
    preserve_singletons: bool = True,
):
    """
    Run clustering using artifacts produced by run_alignment.

    Args:
        run_dir: Root run directory containing all prior stage artifacts
        output_dir: Where to write clustering results (default: run_dir/04_clustering)
        similarity_threshold: Similarity threshold for graph splitting
        n_consensus_runs: Number of consensus iterations
        preserve_singletons: Whether to keep singleton clusters as novel candidates
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "04_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ORAX-KG Consensus Clustering")
    print("=" * 70)
    print(f"\n  Loading from: {run_dir}")

    # ── Load Artifacts ────────────────────────────────────────────────────
    print(f"\n  Loading artifacts...")

    with open(run_dir / "03_alignment" / "alignment_results.json") as f:
        alignment_results = json.load(f)

    with open(run_dir / "02_extraction" / "extractions.jsonl") as f:
        llm_outputs = [json.loads(line) for line in f if line.strip()]

    extracted_embs = torch.load(
        run_dir / "03_alignment" / "extracted_embeddings.pt",
        map_location="cpu",
    )

    with open(run_dir / "02_extraction" / "test_samples.json") as f:
        test_samples = json.load(f)

    with open(run_dir / "01_ontology" / "known_ontology.json") as f:
        known_ontology = json.load(f)

    with open(run_dir / "01_ontology" / "hidden_ontology.json") as f:
        hidden_ontology = json.load(f)

    print(f"   ✓ Loaded {len(llm_outputs)} extractions, "
          f"{len(alignment_results)} alignment results, "
          f"{len(extracted_embs)} embeddings")

    # ── Prepare Labels & Ground Truth ─────────────────────────────────────
    hidden_relation_labels = extract_relation_labels_from_ontology(hidden_ontology)
    full_ground_truth = build_full_ground_truth(test_samples, llm_outputs, alignment_results)

    # ── Embeddings & Similarities ─────────────────────────────────────────
    normalize_embeddings(extracted_embs)

    print(f"\n  Computing inter-triple similarities...")
    inter_triple_sims = compute_inter_triple_similarities(extracted_embs)

    # ── Clustering ────────────────────────────────────────────────────────
    print(f"\n  Running consensus clustering...")
    clusterer = SemanticAwareConsensusClustering(
        n_consensus_runs=n_consensus_runs,
        similarity_threshold=similarity_threshold,
        preserve_singletons=preserve_singletons,
    )

    all_clusters, quality_scores = clusterer.fit_predict(
        alignment_results, extracted_embs, inter_triple_sims, llm_outputs
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    print(f"\n  Evaluating clustering...")
    stratified_metrics = evaluate_clustering(
        all_clusters,
        llm_outputs,
        alignment_results,
        full_ground_truth,
        hidden_relation_labels,
    )

    # Flatten top-level scalar metrics into quality_scores for convenience
    for subset_name, subset_metrics in stratified_metrics.items():
        if isinstance(subset_metrics, dict):
            for metric_name, value in subset_metrics.items():
                if not isinstance(value, dict):
                    quality_scores[f"{subset_name}_{metric_name}"] = value

    # ── Save Results ──────────────────────────────────────────────────────
    print(f"\n  Saving results...")

    # clusters.json — lightweight JSON with idx + triple metadata
    serializable_clusters = {
        mode: {
            str(cid): [
                {
                    "idx":         item["idx"],
                    "triple":      item["triple"],
                    "sentence_id": item.get("id"),
                }
                for item in items
            ]
            for cid, items in clusters.items()
        }
        for mode, clusters in all_clusters.items()
    }

    with open(output_dir / "clusters.json", "w") as f:
        json.dump(serializable_clusters, f, indent=2)

    # cluster_embeddings.pt — full item dicts including embedding tensors,
    # keyed by mode -> cid -> list of {idx, triple, embedding}
    # loaded by run_validation.py to reconstruct cluster items
    torch.save(all_clusters, output_dir / "cluster_embeddings.pt")
    print(f"   Saved cluster embeddings to {output_dir / 'cluster_embeddings.pt'}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    _save_json(stratified_metrics, output_dir / "stratified_metrics.json")

    print("\n" + "=" * 70)
    print(f"  Clustering complete!")
    print(f"   Artifacts saved to: {output_dir}")
    print("=" * 70)

    return output_dir


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ORAX-KG consensus clustering"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Root run directory containing prior stage artifacts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <run-dir>/04_clustering)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--consensus-runs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--no-preserve-singletons",
        action="store_true",
        help="Do not preserve singleton clusters"
    )

    args = parser.parse_args()

    run_clustering(
        run_dir=args.run_dir,
        output_dir=args.output,
        similarity_threshold=args.similarity_threshold,
        n_consensus_runs=args.consensus_runs,
        preserve_singletons=not args.no_preserve_singletons,
    )


if __name__ == "__main__":
    main()
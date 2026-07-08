#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.classification.llm_classifier import setup_client
from src.clustering.consensus import compute_inter_triple_similarities
from src.validation.llm_validator import (
    ClusterValidator,
    ConfidenceLevel,
    export_validated_relations,
    print_validation_summary,
)
from src.logger import setup_logger, close_logger

def load_clusters(clusters_path: str) -> dict:
    """Load clustering output from JSON."""
    with open(clusters_path) as f:
        raw = json.load(f)
    print(f"   Loaded clusters from {clusters_path}")
    return raw


def load_embeddings(embeddings_path: str) -> list:
    """Load pre-computed embeddings saved with torch.save()."""
    embs = torch.load(embeddings_path, map_location="cpu")
    print(f"   Loaded {len(embs)} embeddings from {embeddings_path}")
    return embs


def normalize_embeddings(classified_embs: list) -> list:
    """L2-normalize all embedding vectors in-place."""
    keys = ["relation_emb", "triple_emb", "subj_type_emb", "obj_type_emb"]
    for e in classified_embs:
        for key in keys:
            if key in e:
                e[key] = torch.nn.functional.normalize(e[key].float(), p=2, dim=-1)
    return classified_embs


def run_validation(
    classification_dir: str,
    clusters_path: str,
    embeddings_path: str,
    output_dir: str,
    base_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen3-32B",
    top_k_representatives: int = 5,
    min_cluster_size: int = 5,
    min_confidence: str = "Medium",
):
    """
    Run the full validation pipeline.

    Args:
        classification_dir: Directory produced by run_classification.py
                        (contains 01_ontology/ and 02_classification/)
        clusters_path: Path to clusters JSON file
        embeddings_path: Path to embeddings .pt file
        output_dir: Directory to save validation results
        base_url: Base URL of the vLLM server
        model_name: Model identifier served by vLLM
        top_k_representatives: Max triples per cluster in LLM prompt
        min_cluster_size: Skip clusters smaller than this
        min_confidence: Minimum confidence to accept ('Low'/'Medium'/'High')
    """
    classification_dir = Path(classification_dir)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("ORAX-KG Cluster Validation:")
    print(f"\n  classification dir: {classification_dir}")
    print(f"  Output dir:     {output_dir}")

    print(f"\n  Connecting to vLLM server at {base_url}...")
    client = setup_client(base_url=base_url)
    print(f"  Model: {model_name}")

    print(f"\n  Loading ontology...")
    known_path = classification_dir / "01_ontology" / "known_ontology.json"
    full_path  = classification_dir / "01_ontology" / "full_ontology.json"

    with open(known_path) as f:
        known_ontology = json.load(f)

    with open(full_path) as f:
        full_schema = json.load(f)

    ontology_classes: set = set()
    for domain, ranges in full_schema["relations"].items():
        ontology_classes.add(domain.upper())
        for range_t in ranges:
            ontology_classes.add(range_t.upper())

    print(f"   Known relations:  {len(known_ontology)}")
    print(f"   Ontology classes: {len(ontology_classes)}")

    print(f"\n  Loading clusters from {clusters_path}...")
    raw_clusters = load_clusters(clusters_path)
    n_relation_clusters = len(raw_clusters.get("relation", {}))
    print(f"   Relation-mode clusters: {n_relation_clusters}")

    # Load classified_embeddings.pt for inter-triple similarity computation
    print(f"\n  Loading embeddings from {embeddings_path}...")
    classified_embs = normalize_embeddings(load_embeddings(embeddings_path))

    print(f"\n  Computing inter-triple similarities...")
    inter_triple_sims = compute_inter_triple_similarities(classified_embs)

    # Load full cluster items (with embeddings)
    # cluster_embeddings.pt is saved by run_clustering.py alongside clusters.json
    cluster_embs_path = Path(clusters_path).parent / "cluster_embeddings.pt"
    print(f"\n  Loading cluster embeddings from {cluster_embs_path}...")
    all_clusters = torch.load(cluster_embs_path, map_location="cpu")
    total_items = sum(len(c) for c in all_clusters.get("relation", {}).values())
    print(f"   Loaded {total_items} items across {n_relation_clusters} relation clusters")

    # Validation
    validator = ClusterValidator(
        ontology_classes=ontology_classes,
        known_relations=known_ontology,
        llm_client=client,
        model_name=model_name,
        top_k_representatives=top_k_representatives,
        min_cluster_size=min_cluster_size,
        min_confidence=ConfidenceLevel(min_confidence),
    )

    novel_relations, rejected = validator.validate_all_clusters(
        all_clusters,
        inter_triple_sims,
    )

    print_validation_summary(novel_relations, rejected)

    # Save Results 
    export_validated_relations(novel_relations, str(output_dir / "novel_relations.json"))

    with open(output_dir / "rejected_candidates.json", "w") as f:
        json.dump(
            {
                "existing_match": [
                    {"name": c.name, "matched_to": c.existing_relation_name,
                     "cluster_id": c.cluster_id}
                    for c in rejected["existing_match"]
                ],
                "type_incompatible": [
                    {"name": c.name, "domain": c.domain, "range": c.range,
                     "cluster_id": c.cluster_id}
                    for c in rejected["type_incompatible"]
                ],
                "low_confidence": [
                    {"cluster_id": item[0] if isinstance(item, tuple) else item.cluster_id}
                    for item in rejected["low_confidence"]
                ],
            },
            f,
            indent=2,
        )
    print(f"   Rejected candidates saved to {output_dir / 'rejected_candidates.json'}")

    # Save Run Config
    config = {
        "validation": {
            "base_url":              base_url,
            "model":                 model_name,
            "top_k_representatives": top_k_representatives,
            "min_cluster_size":      min_cluster_size,
            "min_confidence":        min_confidence,
            "classification_dir":        str(classification_dir),
            "clusters_path":         str(clusters_path),
            "embeddings_path":       str(embeddings_path),
        },
        "timestamp":        datetime.now().isoformat(),
        "n_clusters_seen":  n_relation_clusters,
        "n_novel_accepted": len(novel_relations),
        "n_rejected":       sum(len(v) for v in rejected.values()),
    }
    with open(output_dir / "00_config.yaml", "w") as f:
        yaml.dump(config, f)

    print("\n")
    print(f"  Validation complete!")
    print(f"   Novel relations: {len(novel_relations)}")
    print(f"   Results saved to: {output_dir}")

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Run ORAX-KG cluster validation")
    parser.add_argument(
        "--classification-dir", type=str, required=True,
        help="Directory produced by run_classification.py (contains 01_ontology/)"
    )
    parser.add_argument(
        "--clusters", type=str, required=True,
        help="Path to clusters JSON file"
    )
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to embeddings .pt file (torch.save format)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-32B",
        help="Model identifier served by vLLM"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Max representative triples per cluster in LLM prompt"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=5,
        help="Skip clusters smaller than this"
    )
    parser.add_argument(
        "--min-confidence", type=str, default="Medium",
        choices=["Low", "Medium", "High"],
        help="Minimum confidence level to accept a novel relation"
    )

    args = parser.parse_args()
    setup_logger("validation")
    run_validation(
        classification_dir=args.classification_dir,
        clusters_path=args.clusters,
        embeddings_path=args.embeddings,
        output_dir=args.output,
        base_url=args.base_url,
        model_name=args.model,
        top_k_representatives=args.top_k,
        min_cluster_size=args.min_cluster_size,
        min_confidence=args.min_confidence,
    )
    close_logger()

if __name__ == "__main__":
    main()
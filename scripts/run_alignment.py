#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import sys

import torch
sys.path.append(str(Path(__file__).parent.parent))

from src.alignment.embeddings import TripleEmbedder
from src.alignment.similarity import compute_similarities
from src.alignment.aligner import align_triples, AlignmentConfig
from src.logger import setup_logger, close_logger
from src.alignment.metrics import evaluate_alignment, print_alignment_report

def run_alignment(
    classification_dir: str,
    output_dir: str,
    embedder_model: str = "Qwen/Qwen3-Embedding-8B",
    threshold: float = 0.90,
    device: str = None,
):
    """
    Run alignment pipeline using artifacts from classification.

    Args:
        classification_dir: Directory containing classification artifacts
        output_dir: Directory to save alignment artifacts
        embedder_model: HuggingFace embedding model name
        threshold: Cosine similarity threshold for alignment
        device: Device for computation (auto-detected if None)
    """
    classification_dir = Path(classification_dir)
    output_dir     = Path(output_dir)
    alignment_dir  = output_dir / "03_alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("ORAX-KG Ontology Alignment:")
    print(f"\n  Loading from:     {classification_dir}")
    print(f"  Output directory: {alignment_dir}")

    print(f"\n  Loading classification artifacts...")

    with open(classification_dir / "02_classification" / "classifications.jsonl") as f:
        llm_outputs = [json.loads(line) for line in f if line.strip()]
    print(f"   Loaded {len(llm_outputs)} classifications")

    with open(classification_dir / "01_ontology" / "known_ontology.json") as f:
        known_ontology = json.load(f)
    print(f"   Loaded known ontology ({len(known_ontology)} entries)")

    print(f"\n  Preparing classified triples...")
    classified_triples = [
        {
            "sentence_id": item["id"],
            "subject":     item["triple"]["subject"],
            "object":      item["triple"]["object"],
            "relation":    item["triple"]["relation"],
            "subj_type":   item["triple"].get("subj_type"),
            "obj_type":    item["triple"].get("obj_type"),
        }
        for item in llm_outputs
    ]

    ontology_classes = {
        t
        for ont in known_ontology
        for t in (ont.get("domain_type"), ont.get("range_type"))
        if t
    }

    print(f"\n  Initializing embedder: {embedder_model}")
    embedder = TripleEmbedder(embedder_model, device=device)

    print(f"\n  Embedding classified triples...")
    classified_embs = embedder.embed_triples(classified_triples)

    print(f"\n  Embedding ontology patterns...")
    ontology_embs = embedder.embed_ontology(known_ontology)

    print(f"\n  Saving embeddings...")
    torch.save(classified_embs, alignment_dir / "classified_embeddings.pt")
    torch.save(ontology_embs,  alignment_dir / "ontology_embeddings.pt")
    print(f"   Saved to {alignment_dir}")

    print(f"\n  Computing similarities...")
    similarities = compute_similarities(classified_embs, ontology_embs)
    torch.save(similarities, alignment_dir / "similarities.pt")

    print(f"\n  Aligning triples (threshold={threshold})...")
    config  = AlignmentConfig(threshold=threshold)
    results = align_triples(
        classified_triples,
        known_ontology,
        similarities,
        ontology_classes,
        config,
    )

    with open(alignment_dir / "alignment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    stats = {
        "total_triples":         len(results),
        "aligned":               sum(1 for r in results if r["match_type"] == "known_triple"),
        "to_cluster":            sum(1 for r in results if r["to_cluster"]),
        "constraint_violations": sum(1 for r in results if r.get("constraint_violation")),
        "threshold":             threshold,
        "embedder_model":        embedder_model,
    }
    with open(alignment_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Computing alignment metrics...")
    with open(classification_dir / "02_classification" / "test_samples.json") as f:
        import json as _json
        test_samples = _json.load(f)
 
    alignment_metrics = evaluate_alignment(results, test_samples, known_ontology)
    with open(alignment_dir / "evaluation_metrics.json", "w") as f:
        import json as _json
        _json.dump(alignment_metrics, f, indent=2)
 
    print_alignment_report(alignment_metrics)

    print("\n")
    print("  Alignment complete!")
    print(f"   Aligned:          {stats['aligned']}")
    print(f"   Novel candidates: {stats['to_cluster']}")
    print(f"   Artifacts saved:  {alignment_dir}")

    return alignment_dir


def main():
    parser = argparse.ArgumentParser(description="Run ORAX-KG ontology alignment")
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to classification output directory"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save alignment artifacts"
    )
    parser.add_argument(
        "--embedder", type=str, default="Qwen/Qwen3-Embedding-8B",
        help="HuggingFace embedding model name"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Alignment threshold (default: 0.90)"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Device for computation (default: auto-detect)"
    )

    args = parser.parse_args()
    setup_logger("alignment")
    run_alignment(
        classification_dir=args.run_dir,
        output_dir=args.output,
        embedder_model=args.embedder,
        threshold=args.threshold,
        device=args.device,
    )
    close_logger()

if __name__ == "__main__":
    main()
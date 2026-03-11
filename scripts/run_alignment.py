#!/usr/bin/env python3
"""
Run ontology alignment on extracted triples.
"""

import json
import argparse
from pathlib import Path
import sys

import torch
sys.path.append(str(Path(__file__).parent.parent))

from src.alignment.embeddings import TripleEmbedder
from src.alignment.similarity import compute_similarities
from src.alignment.aligner import align_triples, AlignmentConfig


def run_alignment(
    extraction_dir: str,
    output_dir: str,
    embedder_model: str = "Qwen/Qwen3-Embedding-8B",
    threshold: float = 0.90,
    device: str = None,
):
    """
    Run alignment pipeline using artifacts from extraction.

    Args:
        extraction_dir: Directory containing extraction artifacts
        output_dir: Directory to save alignment artifacts
        embedder_model: HuggingFace embedding model name
        threshold: Cosine similarity threshold for alignment
        device: Device for computation (auto-detected if None)
    """
    extraction_dir = Path(extraction_dir)
    output_dir     = Path(output_dir)
    alignment_dir  = output_dir / "03_alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ORAX-KG Ontology Alignment")
    print("=" * 70)
    print(f"\n  Loading from:     {extraction_dir}")
    print(f"  Output directory: {alignment_dir}")

    # ── Load Extraction Artifacts ─────────────────────────────────────────
    print(f"\n  Loading extraction artifacts...")

    with open(extraction_dir / "02_extraction" / "extractions.jsonl") as f:
        llm_outputs = [json.loads(line) for line in f if line.strip()]
    print(f"   Loaded {len(llm_outputs)} extractions")

    with open(extraction_dir / "01_ontology" / "known_ontology.json") as f:
        known_ontology = json.load(f)
    print(f"   Loaded known ontology ({len(known_ontology)} entries)")

    # ── Prepare Extracted Triples ─────────────────────────────────────────
    print(f"\n  Preparing extracted triples...")
    extracted_triples = [
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

    # ── Embed ─────────────────────────────────────────────────────────────
    print(f"\n  Initializing embedder: {embedder_model}")
    embedder = TripleEmbedder(embedder_model, device=device)

    print(f"\n  Embedding extracted triples...")
    extracted_embs = embedder.embed_triples(extracted_triples)

    print(f"\n  Embedding ontology patterns...")
    ontology_embs = embedder.embed_ontology(known_ontology)

    print(f"\n  Saving embeddings...")
    torch.save(extracted_embs, alignment_dir / "extracted_embeddings.pt")
    torch.save(ontology_embs,  alignment_dir / "ontology_embeddings.pt")
    print(f"   Saved to {alignment_dir}")

    # ── Similarities ──────────────────────────────────────────────────────
    print(f"\n  Computing similarities...")
    similarities = compute_similarities(extracted_embs, ontology_embs)
    torch.save(similarities, alignment_dir / "similarities.pt")

    # ── Align ─────────────────────────────────────────────────────────────
    print(f"\n  Aligning triples (threshold={threshold})...")
    config  = AlignmentConfig(threshold=threshold)
    results = align_triples(
        extracted_triples,
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

    print("\n" + "=" * 70)
    print("  Alignment complete!")
    print(f"   Aligned:          {stats['aligned']}")
    print(f"   Novel candidates: {stats['to_cluster']}")
    print(f"   Artifacts saved:  {alignment_dir}")
    print("=" * 70)

    return alignment_dir


def main():
    parser = argparse.ArgumentParser(description="Run ORAX-KG ontology alignment")
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to extraction output directory"
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

    run_alignment(
        extraction_dir=args.run_dir,
        output_dir=args.output,
        embedder_model=args.embedder,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
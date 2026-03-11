#!/usr/bin/env python3
"""
Run complete ORAX-KG pipeline end-to-end.
Stages: Extraction -> Alignment -> Clustering -> Validation
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from run_extraction import run_extraction
from run_alignment import run_alignment
from run_clustering import run_clustering
from run_validation import run_validation


def run_full_pipeline(config_path: str, resume: bool = True):
    """
    Run complete pipeline from a YAML config file.

    Args:
        config_path: Path to YAML config file
        resume: Whether to resume from existing output
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = config.get("run_name", "run")
    run_dir   = Path(config["output"]["base_dir"]) / f"{run_name}_{timestamp}"

    print("=" * 70)
    print("ORAX-KG FULL PIPELINE")
    print("=" * 70)
    print(f"\n  Run directory: {run_dir}")

    # Step 1: Extraction
    print("\n" + "=" * 70)
    print("STEP 1: EXTRACTION")
    print("=" * 70)

    run_extraction(
        data_path=config["data"]["test_path"],
        schema_path=config["data"]["schema_path"],
        output_dir=run_dir,
        base_url=config["extraction"]["base_url"],
        model_name=config["extraction"]["model"],
        resume=resume
    )

    # Step 2: Alignment
    print("\n" + "=" * 70)
    print("STEP 2: ALIGNMENT")
    print("=" * 70)

    run_alignment(
        extraction_dir=run_dir,
        output_dir=run_dir,
        embedder_model=config["alignment"]["embedder_model"],
        threshold=config["alignment"]["threshold"],
        device=config.get("device"),
    )

    # Step 3: Clustering
    print("\n" + "=" * 70)
    print("STEP 3: CLUSTERING")
    print("=" * 70)

    run_clustering(
        run_dir=run_dir,
        similarity_threshold=config["clustering"]["similarity_threshold"],
        n_consensus_runs=config["clustering"]["n_consensus_runs"],
        preserve_singletons=config["clustering"]["preserve_singletons"],
    )

    # Step 4: Validation
    print("\n" + "=" * 70)
    print("STEP 4: VALIDATION")
    print("=" * 70)

    run_validation(
        extraction_dir=run_dir,
        clusters_path=run_dir / "04_clustering" / "clusters.json",
        embeddings_path=run_dir / "03_alignment" / "extracted_embeddings.pt",
        output_dir=run_dir / "05_validation",
        base_url=config["validation"]["base_url"],
        model_name=config["validation"]["model"],
        top_k_representatives=config["validation"].get("top_k_representatives", 5),
        min_cluster_size=config["validation"].get("min_cluster_size", 5),
        min_confidence=config["validation"].get("min_confidence", "Medium"),
    )

    # Summary 
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  All results saved to: {run_dir}")
    print("\n  Output:")
    print(f"    01_ontology/   — known/hidden ontology splits")
    print(f"    02_extraction/ — LLM extractions + ground truth")
    print(f"    03_alignment/  — embeddings + alignment results")
    print(f"    04_clustering/ — consensus cluster assignments")
    print(f"    05_validation/ — novel relations + rejected candidates")


def main():
    parser = argparse.ArgumentParser(description="Run complete ORAX-KG pipeline")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh (ignore existing artifacts)"
    )

    args = parser.parse_args()

    run_full_pipeline(
        config_path=args.config,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
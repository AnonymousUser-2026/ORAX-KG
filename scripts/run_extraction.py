#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict
from datetime import datetime
import sys

import yaml
sys.path.append(str(Path(__file__).parent.parent))

from src.extraction.llm_extractor import (
    OntologyManager,
    extract_triple,
    load_triples,
    sample_dataset,
    setup_client,
)
from src.extraction.metrics import (
    evaluate_extraction,
    compute_extraction_metrics,
    print_extraction_report,
)

from src.logger import setup_logger, close_logger

def load_schema_from_file(schema_path: str) -> Dict:
    """
    Load pre-extracted schema from JSON file.

    Args:
        schema_path: Path to schema JSON file

    Returns:
        Schema dict with 'relations' key
    """
    with open(schema_path) as f:
        relations = json.load(f)

    schema = relations if "relations" in relations else {"relations": relations}

    print(f"   Loaded schema from {schema_path}")
    print(f"   Domain types: {len(schema['relations'])}")

    total_relations = sum(
        len(rels)
        for ranges in schema['relations'].values()
        for rels in ranges.values()
    )
    print(f"   Total relation types: {total_relations}")

    return schema


def run_extraction(
    data_path: str,
    schema_path: str,
    output_dir: str,
    base_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen3-14B",
    resume: bool = True,
):
    """
    Run extraction pipeline on dataset.

    Args:
        data_path: Path to test data JSON
        schema_path: Path to schema JSON file
        output_dir: Directory to save all extraction artifacts
        base_url: Base URL of the vLLM server
        model_name: Model identifier served by vLLM
        resume: Whether to resume from existing output file
    """
    output_dir    = Path(output_dir)
    ontology_dir  = output_dir / "01_ontology"
    extraction_dir = output_dir / "02_extraction"

    ontology_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("ORAX-KG Relation Extraction:")
    print(f"\n  Output directory: {output_dir}")

    print(f"\n  Connecting to vLLM server at {base_url}...")
    client = setup_client(base_url=base_url)
    print(f"  Model: {model_name}")

    print(f"\n  Loading data from {data_path}...")
    all_triples = load_triples(data_path)

    print(f"   Loaded {len(all_triples)} triples")

    print(f"\n  Loading schema from {schema_path}...")
    schema = load_schema_from_file(schema_path)

    with open(ontology_dir / "full_ontology.json", "w") as f:
        json.dump(schema, f, indent=2)

    ontology_manager = OntologyManager()
    ontology_manager.load_from_schema(schema)

    print(f"\n  Splitting ontology...")
    known_mgr, hidden_mgr, metadata = ontology_manager.split_ontology()

    print(f"\n  Saving ontology splits...")
    with open(ontology_dir / "known_ontology.json", "w") as f:
        json.dump(known_mgr.to_dict(), f, indent=2)

    with open(ontology_dir / "hidden_ontology.json", "w") as f:
        json.dump(hidden_mgr.to_dict(), f, indent=2)

    split_metadata = {
        "n_known_relations":  len(set(r["relation"] for r in known_mgr.to_dict())),
        "n_hidden_relations": len(set(r["relation"] for r in hidden_mgr.to_dict())),
        "n_known_entries":    len(known_mgr.to_dict()),
        "n_hidden_entries":   len(hidden_mgr.to_dict()),
        **metadata,
    }
    with open(ontology_dir / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    print(f"   Saved to {ontology_dir}")

    print(f"\n  Sampling test data...")
    sampled = sample_dataset(all_triples, known_mgr.to_dict(), hidden_mgr.to_dict())

    with open(extraction_dir / "test_samples.json", "w") as f:
        json.dump(sampled, f, indent=2)

    ground_truth = {
        sample.get("id", f"sample_{i}"): sample["relation"]
        for i, sample in enumerate(sampled)
    }
    with open(extraction_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    extraction_output = extraction_dir / "extractions.jsonl"
    llm_outputs = []
    start_idx = 0

    if resume and extraction_output.exists():
        print(f"\n  Resuming from existing output...")
        with open(extraction_output, "r", encoding="utf-8") as f:
            llm_outputs = [json.loads(line) for line in f if line.strip()]
        start_idx = len(llm_outputs)
        print(f"   Found {start_idx} existing extractions")
    else:
        print(f"\n  Starting fresh extraction")

    print(f"\n  Extracting from index {start_idx}...")
    print(f"   Total samples: {len(sampled)} | Remaining: {len(sampled) - start_idx}")

    for idx, sample in enumerate(sampled[start_idx:], start=start_idx):
        try:
            result = extract_triple(
                sample=sample,
                ontology_manager=known_mgr,
                client=client,
                model=model_name,
            )
            llm_outputs.append(result)

            with open(extraction_output, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                progress = (idx + 1) / len(sampled) * 100
                print(f"   Progress: {idx + 1}/{len(sampled)} ({progress:.1f}%)")

        except Exception as e:
            print(f"   Error at sample {idx}: {e}")

            fallback = {
                "id":       sample.get("id", f"sample_{idx}"),
                "sentence": sample["sentence"],
                "triple": {
                    "subject":   sample["subject"],
                    "relation":  "extraction_error",
                    "object":    sample["object"],
                    "subj_type": sample.get("subj_type"),
                    "obj_type":  sample.get("obj_type"),
                },
            }
            llm_outputs.append(fallback)
            with open(extraction_output, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")

    config = {
        "extraction": {
            "base_url":    base_url,
            "model":       model_name,
            "data_path":   str(data_path),
            "schema_path": str(schema_path),
        },
        "timestamp":   datetime.now().isoformat(),
        "n_samples":   len(sampled),
        "n_extracted": len(llm_outputs),
    }
    with open(output_dir / "00_config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"\n  Computing extraction metrics...")
 
    known_relation_set = {
        (o["relation"], o["domain_type"], o["range_type"])
        for o in known_mgr.to_dict()
    }
 
    raw_results     = evaluate_extraction(sampled, llm_outputs, known_relation_set)
    metrics         = compute_extraction_metrics(raw_results)
 
    with open(extraction_dir / "extraction_metrics.json", "w") as f:
        import json as _json
        _json.dump({k: v for k, v in metrics.items() if k != "detailed_results"}, f, indent=2)
 
    print_extraction_report(metrics)

    print("\n")
    print(f"  Extraction complete!")
    print(f"   Total extractions: {len(llm_outputs)}")
    print(f"   Artifacts saved to: {output_dir}")

    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Run ORAX-KG relation extraction"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--schema", type=str, required=True,
        help="Path to schema JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for all extraction artifacts"
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-14B",
        help="Model identifier served by vLLM"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh (ignore existing output)"
    )

    args = parser.parse_args()
    setup_logger(f"extraction")
    run_extraction(
        data_path=args.data,
        schema_path=args.schema,
        output_dir=args.output,
        base_url=args.base_url,
        model_name=args.model,
        resume=not args.no_resume,
    )
    close_logger()

if __name__ == "__main__":
    main()
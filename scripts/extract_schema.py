#!/usr/bin/env python3
"""
Extract relation schema from a TACRED-format training file.
Produces a JSON schema mapping domain types → range types → relation labels,
ready to be used as the ontology input for the ORAX-KG pipeline.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def extract_schema(train_path: str, output_path: str):
    """
    Build a relation schema from a TACRED-format JSON file.

    Args:
        train_path: Path to training data JSON (TACRED format)
        output_path: Path to write the schema JSON
    """
    print(f"  Loading data from {train_path}...")

    with open(train_path) as f:
        data = json.load(f)

    # domain → range → set of relation labels
    schema = defaultdict(lambda: defaultdict(set))
    skipped = 0

    for entry in data:
        relation = entry.get("relation", "")
        if relation == "no_relation":
            skipped += 1
            continue

        domain = entry.get("subj_type", "").strip().upper()
        range_t = entry.get("obj_type",  "").strip().upper()

        if not domain or not range_t or not relation:
            skipped += 1
            continue

        schema[domain][range_t].add(relation)

    # Convert sets to sorted lists for deterministic output
    serializable = {
        domain: {
            range_t: sorted(relations)
            for range_t, relations in ranges.items()
        }
        for domain, ranges in sorted(schema.items())
    }

    output = {"relations": serializable}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    n_domains    = len(serializable)
    n_relations  = sum(
        len(rels)
        for ranges in serializable.values()
        for rels in ranges.values()
    )
    n_type_pairs = sum(len(ranges) for ranges in serializable.values())

    print(f"\n  Schema extracted successfully")
    print(f"   Total entries processed: {len(data)}")
    print(f"   Skipped (no_relation / incomplete): {skipped}")
    print(f"   Domain types:  {n_domains}")
    print(f"   Type pairs:    {n_type_pairs}")
    print(f"   Relation types: {n_relations}")
    print(f"\n  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract relation schema from TACRED-format training data"
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training JSON file (TACRED format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the extracted schema JSON"
    )

    args = parser.parse_args()

    extract_schema(
        train_path=args.train,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
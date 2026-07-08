import re
import json
from typing import Dict, List, Optional


def entities_fuzzy_match(classified: str, ground_truth: str) -> bool:
    """Fuzzy entity matching — handles substrings and case differences."""
    if not classified or not ground_truth:
        return False
    ext_clean = classified.lower().strip()
    gt_clean  = ground_truth.lower().strip()
    return ext_clean == gt_clean or ext_clean in gt_clean or gt_clean in ext_clean


def is_meaningful_relation(relation: str) -> bool:
    """
    Check if an classified relation is semantically meaningful.
    Filters out placeholder strings, error markers, and overly generic labels.
    """
    if not relation or len(relation) <= 2:
        return False

    placeholders = [
        r"^new_relation",
        r"^unknown",
        r"^relation$",
        r"^error",
        r"^temp",
        r"^none",
        r"^null",
        r"^\w+_to_\w+$",
    ]
    relation_lower = relation.lower().strip()
    for pattern in placeholders:
        if re.match(pattern, relation_lower):
            return False

    return relation.count("_") <= 2 and len(relation) > 2


def evaluate_classification(
    sampled_data: List[Dict],
    llm_outputs: List[Dict],
    known_relation_set: set,
) -> Dict:
    """
    Evaluate LLM classification results against ground truth.

    Known relations  → entity accuracy + exact relation match
    Hidden relations → entity accuracy + relation_extractability

    Args:
        sampled_data: Original test samples with ground truth
        llm_outputs: LLM classification results (classifications.jsonl)
        known_relation_set: Set of (relation, subj_type, obj_type) tuples
                            for relations shown to the LLM

    Returns:
        Raw counts dict passed to compute_classification_metrics()
    """
    results = {
        "known_relations": {
            "total": 0, "entity_correct": 0,
            "exact_match": 0, "details": []
        },
        "hidden_relations": {
            "total": 0, "entity_correct": 0,
            "relation_predictable": 0, "details": []
        },
    }

    # Index by sentence ID for robust matching
    sample_by_id = {
        (s.get("id") or s.get("sentence_id") or s["sentence"]): s
        for s in sampled_data
    }
    output_by_id = {
        (o.get("id") or o.get("sentence_id") or o.get("sentence", "")): o
        for o in llm_outputs
    }

    matched = [(sample_by_id[sid], output_by_id[sid])
               for sid in sample_by_id if sid in output_by_id]

    print(f"   Matched {len(matched)}/{len(sampled_data)} sample-output pairs")

    for sample, classified in matched:
        gt_sub      = sample["subject"]
        gt_obj      = sample["object"]
        gt_rel      = sample["relation"]
        gt_sub_type = sample.get("subj_type", "UNKNOWN")
        gt_obj_type = sample.get("obj_type",  "UNKNOWN")

        triple  = classified.get("triple", {})
        ext_sub = triple.get("subject",  "")
        ext_obj = triple.get("object",   "")
        ext_rel = triple.get("relation", "")

        is_known  = (gt_rel, gt_sub_type, gt_obj_type) in known_relation_set
        ent_match = entities_fuzzy_match(ext_sub, gt_sub) and entities_fuzzy_match(ext_obj, gt_obj)

        detail = {
            "id":           sample.get("id") or sample.get("sentence_id"),
            "sentence":     sample["sentence"],
            "ground_truth": {"subject": gt_sub, "relation": gt_rel, "object": gt_obj,
                             "subj_type": gt_sub_type, "obj_type": gt_obj_type},
            "classified":    {"subject": ext_sub, "relation": ext_rel, "object": ext_obj},
            "entities_correct": ent_match,
        }

        if is_known:
            exact = (ext_rel == gt_rel)
            results["known_relations"]["total"]          += 1
            results["known_relations"]["entity_correct"] += int(ent_match)
            results["known_relations"]["exact_match"]    += int(exact)
            detail["relation_correct"] = exact
            results["known_relations"]["details"].append(detail)
        else:
            predictable = is_meaningful_relation(ext_rel)
            results["hidden_relations"]["total"]                += 1
            results["hidden_relations"]["entity_correct"]       += int(ent_match)
            results["hidden_relations"]["relation_predictable"] += int(predictable)
            detail["relation_predictable"] = predictable
            results["hidden_relations"]["details"].append(detail)

    return results


def compute_classification_metrics(results: Dict) -> Dict:
    """
    Compute final metrics from raw evaluation counts.

    Known relations  → entity accuracy, relation exact match, full triple accuracy
    Hidden relations → entity accuracy, relation classification rate
    Overall          → weighted average across both subsets
    """
    known       = results["known_relations"]
    hidden      = results["hidden_relations"]
    k_total     = known["total"]
    h_total     = hidden["total"]
    total       = k_total + h_total

    known_metrics = {
        "total_samples":        k_total,
        "entity_accuracy":      known["entity_correct"] / k_total if k_total > 0 else 0.0,
        "relation_exact_match": known["exact_match"]    / k_total if k_total > 0 else 0.0,
        "full_triple_accuracy": known["exact_match"]    / k_total if k_total > 0 else 0.0,
    }

    hidden_metrics = {
        "total_samples":            h_total,
        "entity_accuracy":          hidden["entity_correct"]       / h_total if h_total > 0 else 0.0,
        "relation_classification_rate": hidden["relation_predictable"] / h_total if h_total > 0 else 0.0,
    }

    overall_metrics = {
        "total_samples":         total,
        "entity_accuracy":       (known["entity_correct"] + hidden["entity_correct"]) / total if total > 0 else 0.0,
        "classification_success_rate": (known["exact_match"] + hidden["relation_predictable"]) / total if total > 0 else 0.0,
    }

    return {
        "known_relations":  known_metrics,
        "hidden_relations": hidden_metrics,
        "overall":          overall_metrics,
        "detailed_results": results,
    }


def print_classification_report(metrics: Dict):
    """Print formatted classification evaluation report."""
    k = metrics["known_relations"]
    h = metrics["hidden_relations"]
    o = metrics["overall"]

    print("\n")
    print("classification EVALUATION :")

    print(f"\n  Known Relations ({k['total_samples']} samples):")
    print(f"   Entity Accuracy:      {k['entity_accuracy']:.3f}")
    print(f"   Relation Exact Match: {k['relation_exact_match']:.3f}")
    print(f"   Full Triple Accuracy: {k['full_triple_accuracy']:.3f}")

    print(f"\n  Hidden Relations ({h['total_samples']} samples):")
    print(f"   Entity Accuracy:          {h['entity_accuracy']:.3f}")
    print(f"   Relation classification Rate: {h['relation_classification_rate']:.3f}")

    print(f"\n  Overall ({o['total_samples']} samples):")
    print(f"   Entity Accuracy:       {o['entity_accuracy']:.3f}")
    print(f"   classification Success:    {o['classification_success_rate']:.3f}")

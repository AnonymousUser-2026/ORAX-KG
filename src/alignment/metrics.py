from typing import List, Dict


def evaluate_alignment(
    alignment_results: List[Dict],
    test_samples: List[Dict],
    known_ontology: List[Dict],
) -> Dict:
    """
    Evaluate alignment performance against ground truth.

    Matches on (relation, subj_type, obj_type) tuples when type fields
    are present, otherwise falls back to relation name only.

    Args:
        alignment_results: Output from align_triples
        test_samples: Ground truth test data
        known_ontology: Known ontology patterns

    Returns:
        Dict of evaluation metrics
    """
    # Build known set — use typed tuples when available
    if known_ontology and "domain_type" in known_ontology[0]:
        known_set = {
            (ont["relation"], ont["domain_type"], ont["range_type"])
            for ont in known_ontology
        }
        def is_known(sample):
            return (sample["relation"], sample["subj_type"], sample["obj_type"]) in known_set
    else:
        known_set = {ont["relation"] for ont in known_ontology}
        def is_known(sample):
            return sample["relation"] in known_set

    tp = fp = fn = tn = 0

    for result, sample in zip(alignment_results, test_samples):
        ground_truth_known = is_known(sample)
        predicted_known = result["match_type"] == "known_triple"

        if ground_truth_known and predicted_known:
            tp += 1
        elif not ground_truth_known and predicted_known:
            fp += 1
        elif ground_truth_known and not predicted_known:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total": len(alignment_results),
    }


def print_alignment_report(metrics: Dict):
    """Print formatted alignment report."""
    print("\n")
    print("ALIGNMENT EVALUATION REPORT:")
    print(f"\nPrecision:    {metrics['precision']:.3f}")
    print(f"Recall:       {metrics['recall']:.3f}")
    print(f"F1-Score:     {metrics['f1']:.3f}")
    print(f"Specificity:  {metrics['specificity']:.3f}")
    print(f"\nTrue Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"Total:           {metrics['total']}")
import torch
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AlignmentConfig:
    """Configuration for alignment."""
    threshold: float = 0.90


def check_constraint_violation(
    extracted_triple: Dict,
    ontology_triple: Dict,
    ontology_classes: set,
) -> bool:
    """
    Check if an extracted triple violates ontology type constraints.

    Returns True only when both types are known but don't match the
    relation's domain/range — unknown types signal class expansion,
    not a violation.

    Args:
        extracted_triple: Triple dict with subj_type, obj_type
        ontology_triple: Ontology pattern dict with domain_type, range_type
        ontology_classes: Set of known entity type strings

    Returns:
        True if constraint is violated, False otherwise
    """
    subj_type = extracted_triple.get("subj_type")
    obj_type  = extracted_triple.get("obj_type")

    # Unknown types are not violations
    if subj_type not in ontology_classes or obj_type not in ontology_classes:
        return False

    ont_domain = ontology_triple["domain_type"]
    ont_range  = ontology_triple["range_type"]

    subj_mismatch = subj_type != ont_domain
    obj_mismatch  = (
        obj_type not in ont_range
        if isinstance(ont_range, list)
        else obj_type != ont_range
    )

    return subj_mismatch or obj_mismatch


def align_triples(
    extracted_triples: List[Dict],
    ontology_triples: List[Dict],
    similarities: Dict[str, torch.Tensor],
    ontology_classes: set,
    config: AlignmentConfig,
) -> List[Dict]:
    """
    Align extracted triples to ontology using threshold-based matching.

    Args:
        extracted_triples: List of extracted triple dicts
        ontology_triples: List of ontology pattern dicts
        similarities: Dict of similarity matrices keyed by view name
        ontology_classes: Set of known entity type strings
        config: AlignmentConfig instance

    Returns:
        List of alignment result dicts
    """
    sim_triple = similarities["triple"]
    sim_rel    = similarities["relation"]
    sim_sub    = similarities.get("subject")
    sim_obj    = similarities.get("object")

    # Aggregate score: mean over available views
    if sim_sub is not None and sim_obj is not None:
        score = (sim_triple + sim_rel + sim_sub + sim_obj) / 4
    else:
        score = 0.6 * sim_triple + 0.4 * sim_rel

    results = []

    for i, extracted in enumerate(extracted_triples):
        sent_id = extracted.get("id", extracted.get("sentence_id"))

        best_score, best_idx = torch.max(score[i], dim=0)
        best_idx   = int(best_idx.item())
        best_score = float(best_score.item())

        aligned = best_score >= config.threshold

        constraint_ok = True
        if aligned and sim_sub is not None and sim_obj is not None:
            constraint_ok = not check_constraint_violation(
                extracted,
                ontology_triples[best_idx],
                ontology_classes,
            )

        if aligned and constraint_ok:
            match_type   = "known_triple"
            cluster_mode = None
        elif aligned and not constraint_ok:
            match_type   = "constraint_violation"
            cluster_mode = "constraint"
        else:
            match_type   = "new_triple_candidate"
            cluster_mode = "relation"

        results.append({
            "extracted_idx":        i,
            "sentence_id":          sent_id,
            "matched_ontology_idx": best_idx if aligned else None,
            "match_type":           match_type,
            "constraint_violation": not constraint_ok,
            "to_cluster":           cluster_mode is not None,
            "cluster_mode":         cluster_mode,
            "best_score":           best_score,
            "triple_similarity":    float(sim_triple[i, best_idx].item()),
            "relation_similarity":  float(sim_rel[i, best_idx].item()),
            "subject_similarity":   float(sim_sub[i, best_idx].item()) if sim_sub is not None else None,
            "object_similarity":    float(sim_obj[i, best_idx].item()) if sim_obj is not None else None,
        })

    return results
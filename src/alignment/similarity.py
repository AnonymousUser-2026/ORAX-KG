import torch
from typing import Dict, List


def compute_similarities(
    classified_embs: List[Dict],
    ontology_embs: List[Dict],
) -> Dict[str, torch.Tensor]:
    """
    Compute pairwise cosine similarity matrices between classified triples
    and ontology patterns across four independent views.

    Args:
        classified_embs: Output of TripleEmbedder.embed_triples()
        ontology_embs: Output of TripleEmbedder.embed_ontology()

    Returns:
        Dict with keys 'triple', 'relation', 'subject', 'object',
        each a [N_classified, N_ontology] similarity tensor
    """
    device = classified_embs[0]["triple_emb"].device
    zero   = torch.zeros_like(classified_embs[0]["relation_emb"])

    def _stack(emb_list: List[Dict], key: str) -> torch.Tensor:
        """Stack embeddings, substituting zeros for any missing entry."""
        return torch.stack([
            e[key] if e.get(key) is not None else zero
            for e in emb_list
        ])

    ext_triple = _stack(classified_embs, "triple_emb")
    ext_rel    = _stack(classified_embs, "relation_emb")
    ext_sub    = _stack(classified_embs, "subj_type_emb")
    ext_obj    = _stack(classified_embs, "obj_type_emb")

    ont_triple = _stack(ontology_embs, "triple_emb").to(device)
    ont_rel    = _stack(ontology_embs, "relation_emb").to(device)
    ont_sub    = _stack(ontology_embs, "subj_type_emb").to(device)
    ont_obj    = _stack(ontology_embs, "obj_type_emb").to(device)

    return {
        "triple":   ext_triple @ ont_triple.T,
        "relation": ext_rel    @ ont_rel.T,
        "subject":  ext_sub    @ ont_sub.T,
        "object":   ext_obj    @ ont_obj.T,
    }
"""
Utility functions for clustering.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Set
from collections import Counter


def extract_relation_labels_from_ontology(ontology: List[Dict]) -> Set[str]:
    """
    Extract unique relation labels from an ontology list.

    Args:
        ontology: List of dicts with a 'relation' key

    Returns:
        Set of unique relation label strings
    """
    return {item['relation'] for item in ontology}


def build_full_ground_truth(
    sampled: List[Dict],
    llm_outputs: List[Dict],
    alignment_results: List[Dict],
) -> Dict[int, str]:
    """
    Build a complete ground truth mapping for all extracted triples.

    Args:
        sampled: Original test dataset samples (each with 'id'/'sentence_id' and 'relation')
        llm_outputs: Extracted triples from the LLM (each with 'id'/'sentence_id')
        alignment_results: Alignment results (each with 'extracted_idx')

    Returns:
        Dict mapping extraction_idx → ground_truth_relation
    """
    # sentence_id → ground-truth relation
    sentence_to_gt = {
        (item.get("sentence_id") or item.get("id")): item["relation"]
        for item in sampled
        if (item.get("sentence_id") or item.get("id")) is not None
    }

    full_ground_truth = {}
    for align_result in alignment_results:
        idx = align_result["extracted_idx"]
        if idx >= len(llm_outputs):
            continue
        sent_id = llm_outputs[idx].get("sentence_id") or llm_outputs[idx].get("id")
        if sent_id in sentence_to_gt:
            full_ground_truth[idx] = sentence_to_gt[sent_id]

    print(f"\n✓ Built ground truth for {len(full_ground_truth)} extracted triples")
    return full_ground_truth


def normalize_embeddings(extracted_embeddings: List[Dict]):
    """
    L2-normalize all embedding vectors in-place.

    Args:
        extracted_embeddings: List of embedding dicts
    """
    keys = ["relation_emb", "triple_emb", "subj_type_emb", "obj_type_emb"]
    for e in extracted_embeddings:
        for key in keys:
            if key in e and e[key] is not None:
                e[key] = F.normalize(e[key].float(), p=2, dim=-1)


def analyze_semantic_clusters(
    all_clusters: Dict[str, Dict[int, List[Dict]]],
    inter_triple_sims: Dict[str, torch.Tensor],
    similarity_threshold: float = 0.65,
):
    """
    Print detailed per-cluster statistics including similarity and connectivity.

    Args:
        all_clusters: Clustering results keyed by mode
        inter_triple_sims: Similarity matrices from compute_inter_triple_similarities()
        similarity_threshold: Edge threshold for connectivity analysis
    """
    import networkx as nx

    print("\n" + "=" * 70)
    print("SEMANTIC CLUSTER ANALYSIS")
    print("=" * 70)

    for mode, clusters in sorted(all_clusters.items()):
        print(f"\n{'='*70}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*70}")

        for cluster_id, items in sorted(clusters.items()):
            print(f"\n  CLUSTER {cluster_id} ({len(items)} items)")
            print("  " + "-" * 40)

            if len(items) >= 2:
                indices = [item["idx"] for item in items]
                sim_matrix = inter_triple_sims["relation"][indices][:, indices]

                diag = torch.diag(sim_matrix)
                norm = torch.sqrt(torch.outer(diag, diag)).clamp(min=1e-8)
                sim_np = ((sim_matrix / norm).clamp(-1, 1) + 1) / 2
                sim_np = sim_np.cpu().numpy()

                n = len(indices)
                pairwise = [sim_np[i, j] for i in range(n) for j in range(i + 1, n)]

                print(f"  Avg similarity: {np.mean(pairwise):.3f} | "
                      f"Range: [{np.min(pairwise):.3f}, {np.max(pairwise):.3f}]")

                G = nx.Graph()
                for i in range(n):
                    for j in range(i + 1, n):
                        if sim_np[i, j] >= similarity_threshold:
                            G.add_edge(i, j)
                print(f"  Connectivity: {nx.number_connected_components(G)} component(s)")

            relations = [it["triple"].get("relation", "Unknown") for it in items]
            counts = Counter(relations)
            print(f"\n  Relations ({len(counts)} unique):")
            for rel, count in counts.most_common(5):
                print(f"    • {rel}: {count}")
            if len(counts) > 5:
                print(f"    ... and {len(counts) - 5} more")


def print_clustering_summary(
    all_clusters: Dict[str, Dict[int, List[Dict]]],
    quality_scores: Dict[str, float],
):
    """
    Print a formatted clustering summary.

    Args:
        all_clusters: Clustering results keyed by mode
        quality_scores: Quality metrics dict
    """
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY")
    print("=" * 70)

    total_clusters = sum(len(c) for c in all_clusters.values())
    total_items    = sum(len(items) for c in all_clusters.values() for items in c.values())

    print(f"\n  Cluster Statistics:")
    print(f"   Total clusters: {total_clusters}")
    print(f"   Total items:    {total_items}")

    for mode, clusters in all_clusters.items():
        n_items = sum(len(items) for items in clusters.values())
        print(f"\n   Mode '{mode}':")
        print(f"      Clusters: {len(clusters)}")
        print(f"      Items:    {n_items}")

    print(f"\n  Quality Metrics:")
    for k, v in sorted(quality_scores.items()):
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        elif isinstance(v, int):
            print(f"   {k}: {v}")

    print("=" * 70)
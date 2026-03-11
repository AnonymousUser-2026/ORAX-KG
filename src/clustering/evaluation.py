"""
Evaluation metrics for clustering quality.
Includes both supervised and unsupervised metrics.
"""

import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
    adjusted_mutual_info_score,
)


# ============================================================================
# Unsupervised Metrics
# ============================================================================

def evaluate_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute unsupervised clustering quality metrics.

    Args:
        embeddings: Embedding matrix [N, D]
        labels: Cluster assignments [N]

    Returns:
        Dict of quality metrics
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1 or len(unique_labels) >= len(embeddings):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "density_sep_ratio": np.nan,
        }

    results = {}

    try:
        results["silhouette"] = silhouette_score(embeddings, labels)
    except Exception:
        results["silhouette"] = np.nan

    try:
        results["davies_bouldin"] = davies_bouldin_score(embeddings, labels)
    except Exception:
        results["davies_bouldin"] = np.nan

    try:
        results["calinski_harabasz"] = calinski_harabasz_score(embeddings, labels)
    except Exception:
        results["calinski_harabasz"] = np.nan

    try:
        centroids = np.array([
            embeddings[labels == c].mean(axis=0) for c in unique_labels
        ])
        intra = np.mean([
            np.linalg.norm(embeddings[labels == c] - centroids[i], axis=1).mean()
            for i, c in enumerate(unique_labels)
        ])
        inter = np.mean([
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(len(unique_labels))
            for j in range(i + 1, len(unique_labels))
        ])
        results["density_sep_ratio"] = intra / (inter + 1e-8)
    except Exception:
        results["density_sep_ratio"] = np.nan

    return results


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return 1 - np.dot(Xn, Xn.T)


def dunn_index(embs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Dunn index (higher is better).

    Args:
        embs: Embeddings
        labels: Cluster labels

    Returns:
        Dunn index value
    """
    clusters = [embs[labels == lbl] for lbl in np.unique(labels)]

    if len(clusters) < 2:
        return np.nan

    intra = [np.max(pairwise_distances(c)) for c in clusters if len(c) > 1]
    inter = [
        np.min(np.linalg.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0)))
        for c1, c2 in itertools.combinations(clusters, 2)
    ]

    if not intra or not inter:
        return np.nan

    return np.min(inter) / np.max(intra)


def evaluate_typepair_clusters(
    clusters: Dict[int, List[Dict]],
    embeddings: np.ndarray,
    type_signatures: Dict[int, str]
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """
    Evaluate clustering quality per (subject_type, object_type) pair.

    Args:
        clusters: Cluster assignments
        embeddings: Embedding matrix
        type_signatures: Mapping from idx to type signature string

    Returns:
        (per_typepair_results, summary_stats)
    """
    typepair_groups = defaultdict(list)
    for cid, triples in clusters.items():
        for t in triples:
            pair = type_signatures.get(t["idx"], "UNK:UNK")
            typepair_groups[pair].append((cid, t))

    results = {}

    for pair, data in typepair_groups.items():
        if len(data) < 2:
            continue

        idxs = [t["idx"] for _, t in data]
        pair_embs = embeddings[idxs]
        pair_labels = np.array([cid for cid, _ in data])

        # Keep only clusters with multiple members
        multi_mask = np.isin(
            pair_labels,
            [l for l in np.unique(pair_labels) if np.sum(pair_labels == l) > 1]
        )
        if np.sum(multi_mask) < 3:
            continue

        pair_embs = pair_embs[multi_mask]
        pair_labels = pair_labels[multi_mask]

        def _safe(fn):
            try:
                return fn()
            except Exception:
                return np.nan

        results[pair] = {
            "silhouette":        _safe(lambda: silhouette_score(pair_embs, pair_labels)),
            "calinski_harabasz": _safe(lambda: calinski_harabasz_score(pair_embs, pair_labels)),
            "davies_bouldin":    _safe(lambda: davies_bouldin_score(pair_embs, pair_labels)),
            "dunn_index":        _safe(lambda: dunn_index(pair_embs, pair_labels)),
            "n_samples":  len(pair_embs),
            "n_clusters": len(np.unique(pair_labels)),
        }

    if results:
        def _avg(key):
            vals = [v[key] for v in results.values() if not np.isnan(v[key])]
            return np.mean(vals) if vals else np.nan

        summary = {
            "avg_silhouette":        _avg("silhouette"),
            "avg_calinski_harabasz": _avg("calinski_harabasz"),
            "avg_davies_bouldin":    _avg("davies_bouldin"),
            "avg_dunn_index":        _avg("dunn_index"),
            "n_typepairs":           len(results),
        }
    else:
        summary = {}

    return results, summary


# ============================================================================
# Supervised Metrics (with Ground Truth)
# ============================================================================

def map_clusters_to_ground_truth(
    pred_labels: np.ndarray,
    true_labels: np.ndarray
) -> Tuple[np.ndarray, Dict[int, str], float]:
    """
    Use Hungarian algorithm to find the optimal mapping between predicted
    cluster IDs and ground-truth relation labels.

    Args:
        pred_labels: Predicted cluster IDs
        true_labels: Ground truth relation labels

    Returns:
        (mapped_pred_labels, cluster_to_gt_map, accuracy)
    """
    unique_pred = np.unique(pred_labels[pred_labels >= 0])
    unique_true = np.unique(true_labels[true_labels != ""])

    pred_to_idx = {label: i for i, label in enumerate(unique_pred)}
    true_to_idx = {label: i for i, label in enumerate(unique_true)}
    idx_to_true = {i: label for label, i in true_to_idx.items()}

    cost_matrix = np.zeros((len(unique_pred), len(unique_true)))
    for i, p in enumerate(unique_pred):
        for j, t in enumerate(unique_true):
            cost_matrix[i, j] = -np.sum((pred_labels == p) & (true_labels == t))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    cluster_to_gt_map = {
        unique_pred[pi]: idx_to_true[ti]
        for pi, ti in zip(row_ind, col_ind)
    }
    for cid in unique_pred:
        cluster_to_gt_map.setdefault(cid, "NOVEL_UNMAPPED")

    mapped_pred_labels = np.array([
        cluster_to_gt_map.get(p, "UNASSIGNED") if p >= 0 else "UNASSIGNED"
        for p in pred_labels
    ])

    valid = (pred_labels >= 0) & (true_labels != "")
    accuracy = float(np.mean(mapped_pred_labels[valid] == true_labels[valid])) if valid.any() else 0.0

    return mapped_pred_labels, cluster_to_gt_map, accuracy


def compute_bcubed(
    mapped_pred_labels: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute B³ precision, recall, and F1 after label mapping.

    Args:
        mapped_pred_labels: Predicted labels after Hungarian mapping
        true_labels: Ground truth labels

    Returns:
        Dict with bcubed_precision, bcubed_recall, bcubed_f1
    """
    valid = (mapped_pred_labels != "UNASSIGNED") & (true_labels != "")
    pred = mapped_pred_labels[valid]
    true = true_labels[valid]

    if len(pred) == 0:
        return {"bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0}

    n = len(pred)
    p_sum = r_sum = 0.0

    for i in range(n):
        same_pred = pred == pred[i]
        same_true = true == true[i]
        if same_pred.sum() > 0:
            p_sum += (same_pred & same_true).sum() / same_pred.sum()
        if same_true.sum() > 0:
            r_sum += (same_pred & same_true).sum() / same_true.sum()

    precision = p_sum / n
    recall = r_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"bcubed_precision": precision, "bcubed_recall": recall, "bcubed_f1": f1}


def _evaluate_subset(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    label: str
) -> Dict:
    """Run full suite of supervised metrics on a subset."""
    print(f"\n {label}")
    print(f"   Samples: {len(pred_labels)} | "
          f"GT classes: {len(np.unique(true_labels))} | "
          f"Pred clusters: {len(np.unique(pred_labels))}")

    mapped_pred, cluster_map, mapping_acc = map_clusters_to_ground_truth(
        pred_labels, true_labels
    )
    bcubed = compute_bcubed(mapped_pred, true_labels)

    true_int, _ = pd.factorize(true_labels)
    mapped_int, _ = pd.factorize(mapped_pred)

    metrics = {
        **bcubed,
        "v_measure":        v_measure_score(true_int, mapped_int),
        "homogeneity":      homogeneity_score(true_int, mapped_int),
        "completeness":     completeness_score(true_int, mapped_int),
        "ari":              adjusted_rand_score(true_int, mapped_int),
        "ami":              adjusted_mutual_info_score(true_int, mapped_int),
        "mapping_accuracy": mapping_acc,
        "n_samples":        len(pred_labels),
        "n_clusters":       len(np.unique(pred_labels)),
        "n_true_classes":   len(np.unique(true_labels)),
        "cluster_to_gt_map": cluster_map,
    }

    print(f"   B³ F1: {metrics['bcubed_f1']:.3f} | "
          f"V-measure: {metrics['v_measure']:.3f} | "
          f"ARI: {metrics['ari']:.3f}")

    return metrics


def evaluate_clustering(
    all_clusters: Dict[str, Dict[int, List[Dict]]],
    llm_outputs: List[Dict],
    alignment_results: List[Dict],
    full_ground_truth: Dict[int, str],
    hidden_relation_labels: Set[str],
) -> Dict[str, dict]:
    """
    Stratified evaluation following ACL 2022 best practices.

    Evaluates clustering separately on:
    1. Novel relations  — hidden ground truth only (primary discovery metric)
    2. Known relations  — known relations that failed alignment (robustness check)
    3. Overall pipeline — all clustered items combined

    Args:
        all_clusters: Clustering results keyed by mode
        llm_outputs: Original LLM outputs
        alignment_results: Alignment results
        full_ground_truth: Complete ground truth mapping {idx: relation}
        hidden_relation_labels: Set of withheld relation type strings

    Returns:
        Dict with keys 'novel_discovery', 'robustness_known', 'overall_pipeline'
    """
    n_items = len(llm_outputs)

    # Build flat prediction and ground-truth arrays
    pred_labels = np.full(n_items, -1, dtype=int)
    for cluster_id, items in all_clusters.get("relation", {}).items():
        for item in items:
            if item["idx"] < n_items:
                pred_labels[item["idx"]] = cluster_id

    true_labels = np.full(n_items, "", dtype=object)
    for idx, rel in full_ground_truth.items():
        if idx < n_items:
            true_labels[idx] = rel

    aligned_set = {
        r["extracted_idx"]
        for r in alignment_results
        if r.get("match_type") == "known_triple"
    }

    is_clustered   = pred_labels >= 0
    is_labeled     = true_labels != ""
    is_hidden      = np.array([true_labels[i] in hidden_relation_labels for i in range(n_items)])
    is_not_aligned = np.array([i not in aligned_set for i in range(n_items)])

    results = {}

    # 1. Novel relations
    mask = is_clustered & is_labeled & is_hidden & is_not_aligned
    if mask.sum() >= 2:
        results["novel_discovery"] = _evaluate_subset(
            pred_labels[mask], true_labels[mask], "NOVEL DISCOVERY EVALUATION"
        )
    else:
        print(f"\n  Only {mask.sum()} hidden relations — insufficient for evaluation")

    # 2. Known relations (robustness)
    mask = is_clustered & is_labeled & (~is_hidden) & is_not_aligned
    if mask.sum() >= 2:
        results["robustness_known"] = _evaluate_subset(
            pred_labels[mask], true_labels[mask], "ROBUSTNESS CHECK (Known Relations)"
        )
    else:
        print(f"\n  Only {mask.sum()} known relations — skipping robustness check")

    # 3. Overall pipeline
    mask = is_clustered & is_labeled & is_not_aligned
    if mask.sum() >= 2:
        results["overall_pipeline"] = _evaluate_subset(
            pred_labels[mask], true_labels[mask], "OVERALL PIPELINE PERFORMANCE"
        )

    return results


def get_flat_labels_from_clusters(
    all_clusters: Dict[str, Dict[int, List[Dict]]],
    n_items: int,
    mode: str = "relation"
) -> np.ndarray:
    """
    Convert cluster dict to a flat label array.

    Args:
        all_clusters: Clustering results
        n_items: Total number of items
        mode: Clustering mode to extract

    Returns:
        Flat array of cluster labels (-1 for unclustered)
    """
    labels = np.full(n_items, -1, dtype=int)
    for cluster_id, items in all_clusters.get(mode, {}).items():
        for it in items:
            labels[it["idx"]] = cluster_id
    return labels
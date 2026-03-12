import os
import random
import numpy as np
import torch
import networkx as nx
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

# Optional imports
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print(" HDBSCAN not available. Install: pip install hdbscan")

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print(" Leiden not available. Install: pip install leidenalg python-igraph")


SEED = 42


def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_all_seeds(SEED)


def compute_inter_triple_similarities(
    extracted_embeddings: List[Dict],
) -> Dict[str, torch.Tensor]:
    """
    Compute pairwise similarities between extracted triples.

    Args:
        extracted_embeddings: List of embedding dicts with keys:
            'triple_emb', 'relation_emb', 'subj_type_emb', 'obj_type_emb'

    Returns:
        Dict of similarity matrices keyed by view name
    """
    if not extracted_embeddings:
        raise ValueError("extracted_embeddings cannot be empty")

    zero = torch.zeros_like(extracted_embeddings[0]["relation_emb"])

    triple_embs   = torch.stack([e["triple_emb"]   for e in extracted_embeddings])
    relation_embs = torch.stack([e["relation_emb"] for e in extracted_embeddings])
    subj_embs     = torch.stack([
        e["subj_type_emb"] if e.get("subj_type_emb") is not None else zero
        for e in extracted_embeddings
    ])
    obj_embs      = torch.stack([
        e["obj_type_emb"] if e.get("obj_type_emb") is not None else zero
        for e in extracted_embeddings
    ])

    return {
        "triple":   triple_embs   @ triple_embs.T,
        "relation": relation_embs @ relation_embs.T,
        "subject":  subj_embs     @ subj_embs.T,
        "object":   obj_embs      @ obj_embs.T,
    }


class SemanticAwareConsensusClustering:
    """
    Multi-algorithm consensus clustering for novel relation discovery.
    Uses Spectral, HDBSCAN, and Leiden algorithms.
    """

    def __init__(
        self,
        n_consensus_runs: int = 3,
        algorithms: Optional[List[str]] = None,
        similarity_threshold: float = 0.60,
        graph_split_percentile: float = 25.0,
        spectral_max_k: int = 20,
        preserve_singletons: bool = True,
        singleton_merge_threshold: float = 0.80,
        early_stop_proposals: int = 40,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int = 3,
        leiden_resolution: float = 1.0,
    ):
        self.n_consensus_runs = n_consensus_runs

        if algorithms is None:
            algorithms = ["spectral"]
            if HDBSCAN_AVAILABLE:
                algorithms.append("hdbscan")
            if LEIDEN_AVAILABLE:
                algorithms.append("leiden")

        self.algorithms = algorithms
        self.similarity_threshold = similarity_threshold
        self.graph_split_percentile = graph_split_percentile
        self.spectral_max_k = spectral_max_k
        self.preserve_singletons = preserve_singletons
        self.singleton_merge_threshold = singleton_merge_threshold
        self.early_stop_proposals = early_stop_proposals
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.leiden_resolution = leiden_resolution
        self.cluster_quality_scores = {}

        print(f" Initialized with algorithms: {self.algorithms}")

    def fit_predict(
        self,
        alignment_results: List[Dict],
        extracted_embeddings: List[Dict],
        inter_triple_sims: Dict[str, torch.Tensor],
        llm_outputs: List[Dict],
    ) -> Tuple[Dict[str, Dict[int, List[Dict]]], Dict[str, float]]:
        """
        Main clustering pipeline.

        Args:
            alignment_results: Alignment results
            extracted_embeddings: Embedded triples
            inter_triple_sims: Similarity matrices
            llm_outputs: Original LLM outputs

        Returns:
            (all_clusters, quality_scores)
        """
        mode_groups = self._prepare_groups(
            alignment_results, extracted_embeddings, inter_triple_sims, llm_outputs
        )

        if not mode_groups:
            print(" No items to cluster.")
            return {}, {}

        all_clusters = {}
        for mode, items in mode_groups.items():
            if len(items) < 2:
                all_clusters[mode] = {0: items}
                continue

            clusters = self._cluster_mode(mode, items, inter_triple_sims)
            clusters = self._graph_split_clusters(clusters, inter_triple_sims)

            if not self.preserve_singletons:
                clusters = self._assign_noise_to_clusters(clusters, inter_triple_sims)
            else:
                n_singletons = sum(1 for c in clusters.values() if len(c) == 1)
                print(f"   Preserving {n_singletons} singletons as potential novel relations")

            all_clusters[mode] = clusters
            print(f"→ Mode '{mode}' produced {len(clusters)} clusters")

        self._calculate_quality_metrics(all_clusters, inter_triple_sims)
        return all_clusters, self.cluster_quality_scores

    def _prepare_groups(
        self,
        alignment_results: List[Dict],
        extracted_embeddings: List[Dict],
        inter_triple_sims: Dict[str, torch.Tensor],
        llm_outputs: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Group items by clustering mode."""
        cluster_groups = defaultdict(list)

        for result in alignment_results:
            if not result.get("to_cluster"):
                continue

            mode = result["cluster_mode"]
            idx  = result["extracted_idx"]

            if idx >= len(extracted_embeddings) or idx >= len(llm_outputs):
                continue

            cluster_groups[mode].append({
                "alignment_result": result,
                "embedding":        extracted_embeddings[idx],
                "triple":           llm_outputs[idx]["triple"],
                "idx":              idx,
                "relation_sims":    inter_triple_sims["relation"][idx],
                "triple_sims":      inter_triple_sims["triple"][idx],
                "subject_sims":     inter_triple_sims["subject"][idx],
                "object_sims":      inter_triple_sims["object"][idx],
            })

        return dict(cluster_groups)

    def _cluster_mode(
        self,
        mode: str,
        items: List[Dict],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> Dict[int, List[Dict]]:
        """Route to appropriate clustering method."""
        if mode in ("relation", "triple"):
            return self._cluster_by_relation(items, inter_triple_sims)
        return {0: items}

    def _cluster_by_relation(
        self,
        items: List[Dict],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> Dict[int, List[Dict]]:
        """Cluster items by relation with (subject_type, object_type) stratification."""
        if len(items) < 2:
            return {0: items}

        type_groups = self._stratify_by_types(items)
        print(f"   Type stratification: {len(type_groups)} type-pair groups")

        final_clusters = {}
        global_cid = 0

        for sig, group in type_groups.items():
            if len(group) < 2:
                final_clusters[global_cid] = group
                global_cid += 1
                continue

            print(f"   Clustering {sig} ({len(group)} items)")

            idxs     = [it["idx"] for it in group]
            rel_sim  = inter_triple_sims["relation"][idxs][:, idxs]
            tri_sim  = inter_triple_sims["triple"][idxs][:, idxs]
            subj_sim = inter_triple_sims["subject"][idxs][:, idxs]
            obj_sim  = inter_triple_sims["object"][idxs][:, idxs]

            combined = self._cosine_normalize(
                (rel_sim + tri_sim + subj_sim + obj_sim) / 4
            )
            sim_np  = combined.cpu().numpy()
            dist_np = 1.0 - sim_np

            if np.any(np.isnan(dist_np)) or np.any(np.isinf(dist_np)):
                print(f"  Invalid distance matrix for {sig}, using fallback")
                sim_np  = self._cosine_normalize(rel_sim).cpu().numpy()
                dist_np = 1.0 - sim_np

            labels = self._consensus_cluster(dist_np, sim_np)

            for local_lbl in set(labels):
                final_clusters[global_cid] = [
                    it for it, l in zip(group, labels) if l == local_lbl
                ]
                global_cid += 1

        return final_clusters

    def _stratify_by_types(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """Group items by (subject_type, object_type) signature."""
        type_groups = defaultdict(list)
        for item in items:
            triple = item["triple"]
            sig = f"{triple.get('subj_type', 'UNK')}:{triple.get('obj_type', 'UNK')}"
            type_groups[sig].append(item)
        return dict(type_groups)

    def _cosine_normalize(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize similarity matrix to [0, 1] range."""
        diag     = torch.diag(sim_matrix).clamp(min=1e-6)
        norm     = torch.sqrt(torch.outer(diag, diag))
        sim_norm = torch.clamp(
            torch.nan_to_num(sim_matrix / norm, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0, 1.0,
        )
        return (sim_norm + 1.0) / 2.0

    def _consensus_cluster(
        self,
        distance_matrix: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Consensus clustering across Spectral, HDBSCAN, and Leiden.
        Each algorithm contributes equally weighted votes.
        """
        rng = np.random.RandomState(SEED)
        n   = distance_matrix.shape[0]

        if n < 2 or np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            return np.zeros(n, dtype=int)

        consensus_votes = np.zeros((n, n), dtype=float)
        total_weight    = 0.0
        good_proposals  = 0
        algo_votes      = {algo: 0 for algo in self.algorithms}

        for run in range(self.n_consensus_runs):
            noise = rng.normal(0, 0.01, size=distance_matrix.shape)

            noisy_dist = np.clip(distance_matrix + noise, 0.0, None)
            noisy_dist = (noisy_dist + noisy_dist.T) / 2.0

            noisy_sim = np.clip(similarity_matrix + noise, 0.0, 1.0)
            noisy_sim = (noisy_sim + noisy_sim.T) / 2.0

            runners = [
                ("spectral", lambda: self._run_spectral(noisy_sim, n, consensus_votes)),
                ("hdbscan",  lambda: self._run_hdbscan(noisy_dist, n, consensus_votes) if HDBSCAN_AVAILABLE else 0.0),
                ("leiden",   lambda: self._run_leiden(noisy_sim,  n, consensus_votes) if LEIDEN_AVAILABLE else 0.0),
            ]

            for algo, fn in runners:
                if algo not in self.algorithms:
                    continue
                w = fn()
                if w > 0:
                    total_weight   += w
                    good_proposals += 1
                    algo_votes[algo] += 1

            if good_proposals >= self.early_stop_proposals:
                print(f"   Early stop: {good_proposals} proposals after {run + 1} runs")
                break

        if total_weight <= 0:
            return np.zeros(n, dtype=int)

        print(f"   Algorithm votes: {algo_votes}")

        consensus_affinity = consensus_votes / total_weight
        np.fill_diagonal(consensus_affinity, 1.0)
        return self._final_clustering(consensus_affinity, n)

    def _run_spectral(
        self,
        noisy_sim: np.ndarray,
        n: int,
        consensus_votes: np.ndarray,
    ) -> float:
        try:
            est_k = self._estimate_k_by_eigengap(noisy_sim, self.spectral_max_k)
            est_k = min(max(est_k or 2, 2), min(self.spectral_max_k, n - 1))
            if n < 3 or est_k < 2:
                return 0.0

            labels = SpectralClustering(
                n_clusters=est_k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=SEED,
                n_init=10,
            ).fit_predict(noisy_sim)

            return self._accumulate_votes(1.0 - noisy_sim, labels, consensus_votes)
        except Exception:
            return 0.0

    def _run_hdbscan(
        self,
        noisy_dist: np.ndarray,
        n: int,
        consensus_votes: np.ndarray,
    ) -> float:
        try:
            labels = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric="precomputed",
                cluster_selection_method="eom",
            ).fit_predict(noisy_dist)

            if len(np.unique(labels[labels >= 0])) == 0:
                return 0.0

            weight = self._proposal_weight(noisy_dist, labels)
            if weight > 0.1:
                for i in range(n):
                    for j in range(i + 1, n):
                        if labels[i] >= 0 and labels[j] >= 0 and labels[i] == labels[j]:
                            consensus_votes[i, j] += weight
                            consensus_votes[j, i] += weight
                return weight
        except Exception:
            pass
        return 0.0

    def _run_leiden(
        self,
        noisy_sim: np.ndarray,
        n: int,
        consensus_votes: np.ndarray,
    ) -> float:
        try:
            edges   = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if noisy_sim[i, j] > 0.1]
            weights = [float(noisy_sim[i, j]) for i, j in edges]

            if not edges:
                return 0.0

            g = ig.Graph(n=n, edges=edges)
            g.es["weight"] = weights

            labels = np.array(leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=self.leiden_resolution,
                seed=SEED,
            ).membership)

            return self._accumulate_votes(1.0 - noisy_sim, labels, consensus_votes)
        except Exception:
            return 0.0

    def _accumulate_votes(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
        consensus_votes: np.ndarray,
    ) -> float:
        """Compute proposal weight and accumulate co-cluster votes."""
        weight = self._proposal_weight(distance_matrix, labels)
        if weight > 0.1:
            n = len(labels)
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        consensus_votes[i, j] += weight
                        consensus_votes[j, i] += weight
        return weight

    def _proposal_weight(
        self,
        distance_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Weight a clustering proposal by its silhouette score."""
        try:
            from sklearn.metrics import silhouette_score
            valid = labels >= 0
            if np.sum(valid) < 2 or len(np.unique(labels[valid])) <= 1:
                return 0.0
            score = silhouette_score(
                distance_matrix[valid][:, valid], labels[valid], metric="precomputed"
            )
            return float(score) if score > 0 else 0.0
        except Exception:
            return 0.0

    def _estimate_k_by_eigengap(
        self,
        affinity: np.ndarray,
        max_k: int = 20,
    ) -> Optional[int]:
        """Estimate number of clusters via eigengap heuristic."""
        try:
            degrees    = np.sum(affinity, axis=1)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(degrees, 1e-8, None)))
            L          = np.eye(affinity.shape[0]) - d_inv_sqrt @ affinity @ d_inv_sqrt
            eigvals    = np.sort(np.linalg.eigvalsh(L))
            gaps       = np.diff(eigvals[:min(max_k + 1, len(eigvals))])
            return max(2, min(int(np.argmax(gaps) + 1), max_k)) if gaps.size else None
        except Exception:
            return None

    def _final_clustering(
        self,
        consensus_affinity: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Final clustering pass on the consensus affinity matrix."""
        try:
            vals = consensus_affinity[np.triu_indices(n, k=1)]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                thr = float(np.percentile(vals, 75))
                G   = nx.from_numpy_array((consensus_affinity >= thr).astype(int))
                labels = np.zeros(n, dtype=int)
                for cid, comp in enumerate(nx.connected_components(G)):
                    for i in comp:
                        labels[i] = cid
                return labels
        except Exception:
            pass

        try:
            dist = np.nan_to_num(1.0 - consensus_affinity, nan=1.0, posinf=1.0)
            return AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.4,
                metric="precomputed",
                linkage="average",
            ).fit_predict(dist)
        except Exception:
            return np.zeros(n, dtype=int)

    def _graph_split_clusters(
        self,
        clusters: Dict[int, List[Dict]],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> Dict[int, List[Dict]]:
        """Split internally disconnected clusters using graph connectivity."""
        refined = {}
        new_id  = 0

        for cid, items in clusters.items():
            if len(items) < 2:
                refined[new_id] = items
                new_id += 1
                continue

            idxs = [it["idx"] for it in items]
            sim  = inter_triple_sims["relation"][idxs][:, idxs].cpu().numpy()
            n    = len(idxs)

            tri_idx   = np.triu_indices(n, k=1)
            threshold = max(
                self.similarity_threshold,
                float(np.percentile(sim[tri_idx], self.graph_split_percentile)),
            ) if tri_idx[0].size > 0 else self.similarity_threshold

            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    if sim[i, j] >= threshold:
                        G.add_edge(i, j)

            for comp in nx.connected_components(G):
                refined[new_id] = [items[i] for i in comp]
                new_id += 1

        return refined

    def _assign_noise_to_clusters(
        self,
        clusters: Dict[int, List[Dict]],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> Dict[int, List[Dict]]:
        """Merge singletons into larger clusters or keep as novel."""
        small = [cid for cid, items in clusters.items() if len(items) < 2]
        if not small:
            return clusters

        large  = {cid: items for cid, items in clusters.items() if len(items) >= 2}
        novel  = {}
        next_id = max(large.keys()) + 1 if large else 0

        for cid in small:
            for it in clusters[cid]:
                best_score, best_cid = -1.0, None
                for lc_id, lc_items in large.items():
                    score = inter_triple_sims["relation"][it["idx"]][
                        [it2["idx"] for it2 in lc_items]
                    ].mean().item()
                    if score > best_score:
                        best_score, best_cid = score, lc_id

                if best_score > self.singleton_merge_threshold:
                    large[best_cid].append(it)
                else:
                    novel[next_id] = [it]
                    next_id += 1

        if novel:
            print(f"   Preserved {len(novel)} potential novel relations")

        return {**large, **novel}

    def _calculate_quality_metrics(
        self,
        clusters: Dict[str, Dict[int, List[Dict]]],
        sims: Dict[str, torch.Tensor],
    ):
        """Calculate and store clustering quality metrics."""
        total_items, total_clusters, cluster_sims = 0, 0, []

        for mode, cluster_map in clusters.items():
            for items in cluster_map.values():
                if len(items) < 2:
                    continue
                total_items    += len(items)
                total_clusters += 1
                idxs = [it["idx"] for it in items]
                M    = sims["relation"][idxs][:, idxs]
                vals = [
                    M[i, j].item()
                    for i in range(len(idxs))
                    for j in range(i + 1, len(idxs))
                ]
                if vals:
                    cluster_sims.append(np.mean(vals))

        self.cluster_quality_scores = {
            "total_items":                    total_items,
            "total_clusters":                 total_clusters,
            "avg_cluster_size":               total_items / total_clusters if total_clusters else 0.0,
            "avg_within_cluster_similarity":  float(np.mean(cluster_sims)) if cluster_sims else 0.0,
        }
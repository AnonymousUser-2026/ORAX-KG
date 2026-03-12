import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import torch
import pandas as pd

from ..clustering.consensus import compute_inter_triple_similarities
from ..extraction.llm_extractor import extract_json_from_text


class ConfidenceLevel(Enum):
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"


@dataclass
class RelationCandidate:
    """LLM validation output for a relation candidate."""
    name: Optional[str]
    description: str
    domain: str
    range: str
    existing_match: bool
    existing_relation_name: Optional[str]
    confidence: ConfidenceLevel
    reasoning: str
    cluster_id: int
    cluster_size: int


@dataclass
class ValidatedRelation:
    """Accepted novel relation with supporting evidence."""
    name: str
    description: str
    domain: str
    range: str
    evidence_triples: List[Dict]
    cluster_id: int
    confidence: ConfidenceLevel
    reasoning: str


class ClusterValidator:
    """
    Validates clusters using LLM reasoning to identify novel ontology relations.
    Ensures semantic coherence, novelty, and type compatibility.
    """

    SYSTEM_MESSAGE = (
        "You are an ontology engineer expert performing HIGH-PRECISION semantic validation."
    )

    def __init__(
        self,
        ontology_classes: Set[str],
        known_relations: List[Dict],
        llm_client,
        model_name: str = "Qwen/Qwen3-32B",
        top_k_representatives: int = 5,
        min_cluster_size: int = 5,
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ):
        """
        Args:
            ontology_classes: Valid entity type strings (uppercase)
            known_relations: List of known relation dicts from OntologyManager.to_dict()
            llm_client: OpenAI-compatible client
            model_name: Model identifier served by vLLM
            top_k_representatives: Max triples to include per cluster in prompt
            min_cluster_size: Clusters below this size are skipped
            min_confidence: Minimum confidence level to accept a candidate
        """
        self.ontology_classes = ontology_classes
        self.known_relations  = known_relations
        self.llm_client       = llm_client
        self.model_name       = model_name
        self.top_k            = top_k_representatives
        self.min_cluster_size = min_cluster_size
        self.min_confidence   = min_confidence

        self._build_known_relations_index()

    def _build_known_relations_index(self):
        """Index known relations by (domain, range) type pair for fast lookup."""
        self.known_by_types = defaultdict(list)
        for rel in self.known_relations:
            domain  = rel["domain_type"].upper()
            range_t = rel["range_type"].upper()
            self.known_by_types[f"{domain}:{range_t}"].append({
                "name":   rel["relation"],
                "domain": domain,
                "range":  range_t,
            })

    def select_representatives(
        self,
        cluster_items: List[Dict],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> List[Dict]:
        """
        Select the top-k triples closest to the cluster centroid.

        Implements: Representatives(C_k) = argmin_{t_i in C_k}^{k} ||z_i - z_k||
        """
        if len(cluster_items) <= self.top_k:
            return cluster_items

        embeddings = torch.stack([
            item["embedding"]["triple_emb"] for item in cluster_items
        ])
        centroid  = embeddings.mean(dim=0)
        distances = torch.norm(embeddings - centroid, dim=1)
        _, top_indices = torch.topk(distances, self.top_k, largest=False)

        return [cluster_items[i] for i in top_indices.tolist()]

    def _format_prompt(
        self,
        representatives: List[Dict],
        cluster_id: int,
        cluster_size: int,
    ) -> str:
        """Build the validation prompt for a single cluster."""
        first_triple = representatives[0]["triple"]
        subj_type = first_triple.get("subj_type", "UNKNOWN").upper()
        obj_type  = first_triple.get("obj_type",  "UNKNOWN").upper()
        type_pair = f"{subj_type}:{obj_type}"

        triples_text = "\n".join(
            f"{i}. {t['triple'].get('subject','?')} "
            f"--[{t['triple'].get('relation','?')}]--> "
            f"{t['triple'].get('object','?')}"
            for i, t in enumerate(representatives, 1)
        )

        same_type      = self.known_by_types.get(type_pair, [])
        same_type_text = (
            "\n".join(f"- {r['name']}" for r in same_type)
            if same_type else "None"
        )

        classes_text = ", ".join(sorted(self.ontology_classes))

        return f"""You are an ontology expert performing semantic validation of relation clusters.

### TASK
Your task is to decide whether a cluster of triples srepresents a GENUINELY NEW relation that should be added to the initial ontology,
or if it is REDUNDANT with existing ontology relations.

### INPUT DATA
Cluster ID: {cluster_id}
Cluster Size: {cluster_size}

Domain (subject type): {subj_type}
Range (object type): {obj_type}

Cluster Triples:
{triples_text}

Existing ontology relations with SAME domain-range ({subj_type} -> {obj_type}):
{same_type_text}

Valid ontology classes:
{classes_text}

### VALIDATION PROCEDURE (FOLLOW STRICTLY):
STEP 1 — Identify Core Semantics
• Ignore surface wording.
• Identify the SINGLE underlying semantic relationship.
• If multiple different meanings exist -> cluster is NOT coherent.

STEP 2 — Compare With Existing Relations
For EACH existing relation above:
  1. If I rename all cluster triples using that relation name, do they preserve meaning?
  2. Are these triples simply paraphrases or lexical variants?
  3. Is the semantic connection IDENTICAL (not just related)?

If cluster expresses THE SAME relationship as ANY existing ontology relation -> mark REDUNDANT.
If cluster expresses a DIFFERENT relationship from ALL existing relations -> mark NEW.

STEP 3 — Type Validation
• Domain MUST equal: {subj_type}
• Range  MUST equal: {obj_type}
• Both must exist in valid ontology classes.

STEP 4 — Coherence Check
All triples must express the SAME semantic relationship.


### DECISION CRITERIA
REDUNDANT if:
• Same semantic meaning as an existing relation
• Or incoherent cluster
• Or invalid types

NEW if:
• Distinct semantic meaning
• Valid types
• Strong internal consistency

If NEW, the relation name must be:
• A reusable, abstract verb phrase
• Free of surface entity words
• Free of temporal or contextual modifiers
• General rather than overly specific


### OUTPUT FORMAT (JSON ONLY):
Return ONLY valid JSON. No markdown, no code blocks, no extra text.

Schema:
{{
  "decision": "NEW" or "REDUNDANT",
  "name": "string or null",
  "description": "clear semantic explanation",
  "domain": "{subj_type}",
  "range": "{obj_type}",
  "existing_match": true or false,
  "existing_relation_name": "string or null",
  "confidence": "Low" or "Medium" or "High",
  "reasoning": "concise step-by-step justification"
}}

Hard Constraints:
- If decision = "REDUNDANT":
    - name MUST be null
    - existing_match MUST be true
    - existing_relation_name MUST be provided

- If decision = "NEW":
    - name MUST be provided
    - existing_match MUST be false
    - existing_relation_name MUST be null
- Be FAIR: Only mark REDUNDANT if truly THE SAME relationship, not just related
"""

    def _call_llm(self, prompt: str) -> Dict:
        """Call the OpenAI-compatible LLM and return parsed JSON response."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result  = extract_json_from_text(content)
            if result is None:
                raise ValueError("Failed to parse JSON from LLM response")
            return result

        except Exception as e:
            print(f"   LLM call failed: {e}")
            raise

    def validate_cluster(
        self,
        cluster_id: int,
        cluster_items: List[Dict],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> Optional[RelationCandidate]:
        """
        Validate a single cluster via LLM reasoning.

        Returns:
            RelationCandidate if the LLM responds successfully, else None.
        """
        if not cluster_items:
            return None

        representatives = self.select_representatives(cluster_items, inter_triple_sims)
        prompt = self._format_prompt(representatives, cluster_id, len(cluster_items))

        try:
            resp = self._call_llm(prompt)
            return RelationCandidate(
                name=resp.get("name"),
                description=resp["description"],
                domain=resp["domain"],
                range=resp["range"],
                existing_match=resp["existing_match"],
                existing_relation_name=resp.get("existing_relation_name"),
                confidence=ConfidenceLevel(resp["confidence"]),
                reasoning=resp["reasoning"],
                cluster_id=cluster_id,
                cluster_size=len(cluster_items),
            )
        except Exception as e:
            print(f"   Validation failed for cluster {cluster_id}: {e}")
            return None

    def _is_acceptable(self, candidate: RelationCandidate) -> Tuple[bool, str]:
        """
        Apply acceptance criteria:
          1. existing_match is False (genuinely novel)
          2. domain and range are valid ontology classes
          3. confidence >= min_confidence
        """
        if candidate.existing_match:
            return False, f"Matches existing: {candidate.existing_relation_name}"
        if candidate.domain not in self.ontology_classes:
            return False, f"Invalid domain: {candidate.domain}"
        if candidate.range not in self.ontology_classes:
            return False, f"Invalid range: {candidate.range}"

        level_order = [c.value for c in ConfidenceLevel]
        if level_order.index(candidate.confidence.value) < level_order.index(self.min_confidence.value):
            return False, "Confidence below threshold"

        return True, "Accepted"

    def validate_all_clusters(
        self,
        all_clusters: Dict[str, Dict[int, List[Dict]]],
        inter_triple_sims: Dict[str, torch.Tensor],
    ) -> Tuple[List[ValidatedRelation], Dict[str, list]]:
        """
        Validate all relation-mode clusters and categorize results.

        Args:
            all_clusters: Output of SemanticAwareConsensusClustering.fit_predict()
            inter_triple_sims: Output of compute_inter_triple_similarities()

        Returns:
            (novel_relations, rejected) where rejected is keyed by rejection reason
        """
        relation_clusters = all_clusters.get("relation", {})

        print(f"\n")
        print(f"VALIDATING {len(relation_clusters)} CLUSTERS")
        print(f"\n")

        novel_relations: List[ValidatedRelation] = []
        rejected: Dict[str, list] = {
            "existing_match":    [],
            "type_incompatible": [],
            "low_confidence":    [],
        }

        for cluster_id, cluster_items in sorted(relation_clusters.items()):
            if len(cluster_items) < self.min_cluster_size:
                print(f"Cluster {cluster_id}: Skipping "
                      f"(size {len(cluster_items)} < min {self.min_cluster_size})")
                continue

            print(f"\nCluster {cluster_id} ({len(cluster_items)} items):")
            candidate = self.validate_cluster(cluster_id, cluster_items, inter_triple_sims)

            if candidate is None:
                rejected["low_confidence"].append((cluster_id, cluster_items))
                print(f"   Validation failed (no response)")
                continue

            accepted, reason = self._is_acceptable(candidate)

            print(f"   Name:       {candidate.name}")
            print(f"   Types:      {candidate.domain} -> {candidate.range}")
            print(f"   Confidence: {candidate.confidence.value}")
            print(f"   Decision:   {reason}")

            if accepted:
                novel_relations.append(ValidatedRelation(
                    name=candidate.name,
                    description=candidate.description,
                    domain=candidate.domain,
                    range=candidate.range,
                    evidence_triples=cluster_items,
                    cluster_id=cluster_id,
                    confidence=candidate.confidence,
                    reasoning=candidate.reasoning,
                ))
            else:
                if candidate.existing_match:
                    rejected["existing_match"].append(candidate)
                elif (candidate.domain not in self.ontology_classes or
                      candidate.range  not in self.ontology_classes):
                    rejected["type_incompatible"].append(candidate)
                else:
                    rejected["low_confidence"].append(candidate)

        return novel_relations, rejected


def export_validated_relations(
    novel_relations: List[ValidatedRelation],
    output_path: str,
):
    """Serialize validated novel relations to JSON."""
    export_data = {
        "metadata": {
            "n_novel_relations": len(novel_relations),
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "novel_relations": [
            {
                "name":           rel.name,
                "description":    rel.description,
                "domain":         rel.domain,
                "range":          rel.range,
                "confidence":     rel.confidence.value,
                "cluster_id":     rel.cluster_id,
                "evidence_count": len(rel.evidence_triples),
                "reasoning":      rel.reasoning,
                "sample_triples": [
                    {
                        "subject":  t["triple"].get("subject",  "N/A"),
                        "relation": t["triple"].get("relation", "N/A"),
                        "object":   t["triple"].get("object",   "N/A"),
                    }
                    for t in rel.evidence_triples[:5]
                ],
            }
            for rel in novel_relations
        ],
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n  Exported {len(novel_relations)} novel relations to {output_path}")


def print_validation_summary(
    novel_relations: List[ValidatedRelation],
    rejected: Dict[str, list],
):
    print(f"\n")
    print("VALIDATION SUMMARY")
    print(f"\n")

    print(f"  Novel Relations Accepted:     {len(novel_relations)}")
    print(f"  Rejected (Existing Match):    {len(rejected['existing_match'])}")
    print(f"  Rejected (Type Incompatible): {len(rejected['type_incompatible'])}")
    print(f"  Rejected (Low Confidence):    {len(rejected['low_confidence'])}")

    if novel_relations:
        print(f"\n")
        print("ACCEPTED NOVEL RELATIONS:")
        for rel in novel_relations:
            print(f"\n  {rel.name}")
            print(f"   Description: {rel.description}")
            print(f"   Signature:   {rel.domain} -> {rel.range}")
            print(f"   Confidence:  {rel.confidence.value}")
            print(f"   Evidence:    {len(rel.evidence_triples)} triples")
            print(f"   Reasoning:   {rel.reasoning}")
            print(f"   Samples:")
            for t in rel.evidence_triples[:3]:
                triple = t["triple"]
                print(f"      - {triple.get('subject','N/A')} "
                      f"-> {triple.get('relation','N/A')} "
                      f"-> {triple.get('object','N/A')}")
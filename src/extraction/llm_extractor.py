import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random

from openai import OpenAI
from pydantic import BaseModel


@dataclass
class RelationSchema:
    """Unified relation schema."""
    relation: str
    domain_type: Optional[str] = None
    range_type: Optional[str] = None


class Triple(BaseModel):
    """Triple representation for Pydantic validation."""
    subject: str
    subj_type: Optional[str] = None
    relation: str
    object: str
    obj_type: Optional[str] = None


class ExtractionResult(BaseModel):
    """LLM extraction output format."""
    id: str
    sentence: str
    triple: Triple


class OntologyManager:
    """
    Manages ontology for relation extraction.
    Handles loading, splitting, and querying ontologies.
    """
 
    # Relations withheld from the LLM during evaluation
    HIDDEN_RELATIONS = {
        "per:employee_of",
        "org:top_members/employees",
        "per:age",
        "per:countries_of_residence",
        "per:origin",
        "per:charges",
        "org:founded_by",
        "per:spouse",
    }
 
    def __init__(self):
        self.ontology: List[RelationSchema] = []
        self.entity_types: set = set()
 
    def load_from_schema(self, schema: Dict) -> 'OntologyManager':
        """
        Load ontology from schema format.
        Creates one RelationSchema per (relation, domain, range) combination.
 
        Args:
            schema: Dict with 'relations' key containing domain -> range -> [relations] mappings
 
        Returns:
            self for chaining
        """
        for domain, ranges in schema["relations"].items():
            self.entity_types.add(domain)
            for range_type, relations in ranges.items():
                self.entity_types.add(range_type)
                for relation in relations:
                    if relation == "no_relation":
                        continue
                    self.ontology.append(RelationSchema(
                        relation=relation,
                        domain_type=domain.lower(),
                        range_type=range_type.lower(),
                    ))
 
        print(f"   Loaded {len(self.ontology)} ontology entries")
 
        relation_variants = defaultdict(list)
        for rel_schema in self.ontology:
            relation_variants[rel_schema.relation].append(
                (rel_schema.domain_type, rel_schema.range_type)
            )
 
        multi_variant = {
            rel: types for rel, types in relation_variants.items()
            if len(types) > 1
        }
        if multi_variant:
            print(f"   Found {len(multi_variant)} relations with multiple type signatures")
 
        return self
 
    def to_dict(self) -> List[Dict]:
        """Convert ontology to dictionary format for LLM prompts."""
        return [
            {
                "relation":    schema.relation,
                "domain_type": schema.domain_type,
                "range_type":  schema.range_type,
            }
            for schema in self.ontology
        ]
 
    def split_ontology(
        self,
        hidden_split: float = 0.2,
        seed: int = 42,
    ) -> Tuple['OntologyManager', 'OntologyManager', Dict]:
        """
        Split ontology into known and hidden portions.
        All domain/range variants of the same relation stay together.
 
        If HIDDEN_RELATIONS is non-empty and all its entries are present in
        the ontology, uses the predefined split. Otherwise falls back to a
        random split at the relation level using hidden_split as the fraction
        of relations to withhold.
 
        Args:
            hidden_split: Fraction of relations to hide when using random split (default 0.2)
            seed: Random seed for reproducibility of random split
 
        Returns:
            (known_manager, hidden_manager, metadata_dict)
        """
        # Group entries by relation name — variants stay together
        relation_to_entries = defaultdict(list)
        for rel_schema in self.ontology:
            relation_to_entries[rel_schema.relation].append(rel_schema)
 
        all_relations = list(relation_to_entries.keys())
 
        # Decide split strategy
        available_hidden = self.HIDDEN_RELATIONS & set(all_relations)
        if available_hidden:
            # Predefined split — use HIDDEN_RELATIONS
            split_mode = "predefined"
            hidden_relation_set = available_hidden
            missing = self.HIDDEN_RELATIONS - set(all_relations)
            if missing:
                print(f"   Note: {len(missing)} predefined hidden relations not in ontology: {missing}")
        else:
            # Random split — shuffle at relation level and take hidden_split fraction
            split_mode = "random"
            rng = random.Random(seed)
            shuffled = all_relations[:]
            rng.shuffle(shuffled)
            n_hidden = max(1, round(len(shuffled) * hidden_split))
            hidden_relation_set = set(shuffled[:n_hidden])
            print(f"   No predefined hidden relations found — using random {hidden_split:.0%} split")
 
        known_ontology, hidden_ontology = [], []
        for relation_label, entries in relation_to_entries.items():
            if relation_label in hidden_relation_set:
                hidden_ontology.extend(entries)
            else:
                known_ontology.extend(entries)
 
        assert len(known_ontology) + len(hidden_ontology) == len(self.ontology), \
            f"Split error: {len(known_ontology)} + {len(hidden_ontology)} != {len(self.ontology)}"
 
        known_rels  = len(set(e.relation for e in known_ontology))
        hidden_rels = len(set(e.relation for e in hidden_ontology))
 
        print(f"\n  Ontology Split ({split_mode}):")
        print(f"   Total ontology entries: {len(self.ontology)}")
        print(f"   Known:  {known_rels} relations -> {len(known_ontology)} entries")
        print(f"   Hidden: {hidden_rels} relations -> {len(hidden_ontology)} entries")
        if split_mode == "random":
            print(f"   Hidden relations: {sorted(hidden_relation_set)}")
 
        known_manager = OntologyManager()
        known_manager.set_ontology(known_ontology)
 
        hidden_manager = OntologyManager()
        hidden_manager.set_ontology(hidden_ontology)
 
        metadata = {
            "split_mode":          split_mode,
            "hidden_relations":    sorted(hidden_relation_set),
            "known_relation_count":  known_rels,
            "hidden_relation_count": hidden_rels,
            "hidden_split_ratio":  hidden_rels / (known_rels + hidden_rels) if (known_rels + hidden_rels) > 0 else 0,
        }
 
        return known_manager, hidden_manager, metadata
 
    def set_ontology(self, ontology_list: List[RelationSchema]) -> 'OntologyManager':
        """Set ontology from list of RelationSchema objects."""
        self.ontology = ontology_list
        return self

class PromptGenerator:
    """Generate LLM prompts for relation extraction."""

    @staticmethod
    def generate_system_message() -> str:
        return """You are an expert knowledge graph relation extraction system specialized in semantic relation identification and ontology alignment.

## Your Task
Given a sentence with marked subject and object entities, identify the semantic relation connecting them and align it to the provided ontology.

## CRITICAL RULES

### 1. Entity Extraction
- Extract EXACTLY the entities as marked (subject and object)
- DO NOT modify, paraphrase, or substitute entity mentions
- DO NOT swap subject and object positions

### 2. Relation Direction
- Relations are ALWAYS directional: subject -> relation -> object
- NEVER reverse the subject-object order

### 3. Ontology Alignment Strategy
Step 1: Check if an ontology relation semantically matches the sentence
Step 2: Verify TYPE COMPATIBILITY:
   - Subject type MUST match relation's domain_type
   - Object type MUST match relation's range_type
   - If types don't match, the relation CANNOT be used

Step 3: Decision:
   a) Exact match (relation + types) -> use ontology relation
   b) Semantic match but wrong types -> create NEW relation
   c) No semantic match -> create NEW relation

**New Relation Format:**
- Use descriptive, semantic labels
- Format: lowercase_with_underscores
- Be specific about the relationship

### 4. Forbidden Patterns
NEVER output:
- "new_relation", "unknown", "error", "relation"
- Generic patterns like "related_to"
- Empty strings or null values

### 5. Output Format
- Return ONLY valid JSON
- No markdown, no code blocks, no explanations
- Exactly one triple per response
"""

    @staticmethod
    def generate_prompt(sample: Dict, ontology: List[Dict]) -> str:
        """Generate extraction prompt with ontology and typed entities."""
        ontology_str = "\n".join(
            f"- {ont['relation']}"
            f"\n  Domain: {ont.get('domain_type', 'ANY')}"
            f"\n  Range: {ont.get('range_type', 'ANY')}"
            for ont in ontology
        )

        return f"""### EXTRACTION TASK

**Sentence:**
"{sample['sentence']}"

**Entities:**
- **Subject:** `{sample['subject']}`
  - Type: `{sample['subj_type']}`
- **Object:** `{sample['object']}`
  - Type: `{sample['obj_type']}`

### Available Ontology Relations

{ontology_str}

### Instructions

1. Interpret the semantic relationship between subject and object
2. Check ontology compatibility (meaning + types)
3. If no match, create a new descriptive relation

### Output Format

{{
  "id": "{sample['id']}",
  "sentence": "{sample['sentence']}",
  "triple": {{
    "subject": "{sample['subject']}",
    "subj_type": "{sample['subj_type']}",
    "relation": "<relation_name>",
    "object": "{sample['object']}",
    "obj_type": "{sample['obj_type']}"
  }}
}}
"""

def setup_client(base_url: str = "http://localhost:8000") -> OpenAI:
    """
    Set up an OpenAI-compatible client pointing at a local vLLM server.

    Args:
        base_url: Base URL of the vLLM server (default: http://localhost:8000)

    Returns:
        OpenAI client instance
    """
    return OpenAI(base_url=base_url, api_key="not-needed")

def extract_triple(
    sample: Dict,
    ontology_manager: OntologyManager,
    client: OpenAI,
    model: str = "Qwen/Qwen3-14B",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> Dict:
    """
    Extract a relation triple using the LLM with ontology guidance.

    Args:
        sample: Input sample with sentence, subject, object, and type fields
        ontology_manager: OntologyManager with the known ontology
        client: OpenAI-compatible client
        model: Model identifier served by vLLM
        temperature: Sampling temperature (0.0 for deterministic output)
        max_tokens: Maximum tokens in the LLM response

    Returns:
        Dict with 'id', 'sentence', and 'triple' keys
    """
    system_message = PromptGenerator.generate_system_message()
    prompt = PromptGenerator.generate_prompt(sample, ontology_manager.to_dict())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":   "relation_extraction",
                    "schema": ExtractionResult.model_json_schema(),
                },
            },
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = json.loads(response.choices[0].message.content)
        triple = result.get("triple", {})
        triple.setdefault("subject",  sample["subject"])
        triple.setdefault("object",   sample["object"])
        triple.setdefault("relation", "extraction_failed")

        result["triple"] = triple
        return result

    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return {
            "id":       sample.get("id", "unknown"),
            "sentence": sample["sentence"],
            "triple": {
                "subject":   sample["subject"],
                "relation":  "extraction_failed",
                "object":    sample["object"],
                "subj_type": sample.get("subj_type"),
                "obj_type":  sample.get("obj_type"),
            },
        }

def sample_dataset(
    test_samples: List[Dict],
    known_ontology: List[Dict],
    hidden_ontology: List[Dict],
    seed: int = 42,
) -> List[Dict]:
    """
    Sample test data ensuring coverage of both known and hidden relations.
    Matches on (relation, domain_type, range_type) tuples.
 
    Args:
        test_samples: Full test dataset
        known_ontology: Known ontology relations
        hidden_ontology: Hidden ontology relations
        seed: Random seed
 
    Returns:
        Shuffled list of samples covering known and hidden relations
    """
    random.seed(seed)
 
    known_set  = {(o['relation'], o['domain_type'].lower(), o['range_type'].lower()) for o in known_ontology}
    hidden_set = {(o['relation'], o['domain_type'].lower(), o['range_type'].lower()) for o in hidden_ontology}
 
    known_samples  = [s for s in test_samples
                      if s['relation'] != 'no_relation'
                      and (s['relation'], s['subj_type'].lower(), s['obj_type'].lower()) in known_set]
    hidden_samples = [s for s in test_samples
                      if s['relation'] != 'no_relation'
                      and (s['relation'], s['subj_type'].lower(), s['obj_type'].lower()) in hidden_set]
 
    final = known_samples + hidden_samples
    random.shuffle(final)
 
    print(f"Sampling Statistics:")
    print(f"   Known samples:  {len(known_samples)}")
    print(f"   Hidden samples: {len(hidden_samples)}")
    print(f"   Total sampled:  {len(final)}")
 
    return final


def validate_ontology_split(
    known_mgr: OntologyManager,
    hidden_mgr: OntologyManager,
    original_mgr: OntologyManager,
):
    """
    Validate ontology split for correctness.
    Checks for overlap, entry preservation, and label separation.
    """
    print("\n Validating Ontology Split...")

    known_dict  = known_mgr.to_dict()
    hidden_dict = hidden_mgr.to_dict()

    known_tuples  = {(o['relation'], o['domain_type'], o['range_type']) for o in known_dict}
    hidden_tuples = {(o['relation'], o['domain_type'], o['range_type']) for o in hidden_dict}

    overlap = known_tuples & hidden_tuples
    if overlap:
        print(f"   ERROR: {len(overlap)} overlapping entries!")
    else:
        print(f"   No overlap between known and hidden")

    total = len(known_tuples) + len(hidden_tuples)
    if total != len(original_mgr.ontology):
        print(f"   ERROR: Lost entries! {total} != {len(original_mgr.ontology)}")
    else:
        print(f"   All {len(original_mgr.ontology)} entries preserved")

    known_rels  = {o['relation'] for o in known_dict}
    hidden_rels = {o['relation'] for o in hidden_dict}
    rel_overlap = known_rels & hidden_rels

    if rel_overlap:
        print(f"   WARNING: {len(rel_overlap)} relations appear in both splits")
    else:
        print(f"   Clean split: {len(known_rels)} known, {len(hidden_rels)} hidden")


def load_triples(data_path: str) -> List[Dict]:
    """
    Convert a TACRED-format JSON file to a list of triple dicts.

    Args:
        data_path: Path to dataset JSON file

    Returns:
        List of triple dicts (no_relation entries excluded)
    """
    with open(data_path) as f:
        data = json.load(f)

    triples = []
    for entry in data:
        tokens = entry["token"]
        triple = {
            "id":        entry["id"],
            "sentence":  " ".join(tokens),
            "subject":   " ".join(tokens[entry["subj_start"]: entry["subj_end"] + 1]),
            "object":    " ".join(tokens[entry["obj_start"]:  entry["obj_end"]  + 1]),
            "relation":  entry["relation"],
            "subj_type": entry["subj_type"].strip().lower(),
            "obj_type":  entry["obj_type"].strip().lower(),
        }
        if triple["relation"] != "no_relation":
            triples.append(triple)

    return triples


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract the last valid JSON object from raw LLM output.
    Handles markdown code blocks and multiple JSON objects.

    Args:
        text: Raw LLM output text

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    if not text:
        return None

    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates, brace_count, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None
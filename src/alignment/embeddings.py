import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for base models without specialized embedding."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size())
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-token pooling for Qwen3-Embedding models."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


class TripleEmbedder:
    """Embeds relation triples and ontology patterns into per-view tensors."""

    def __init__(self, model_name: str, device: str = None):
        """
        Initialize embedder.

        Args:
            model_name: Embeddings model name
            device: Device for computation (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        if device == "cpu":
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float32
            ).to(device)
        else:
            self.model = AutoModel.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float16
            ).to(device)

        self.model_name = model_name
        self.pooling_fn = last_token_pool if "Embedding" in model_name else mean_pool

    def embed_texts(self, texts: List[str], batch_size: int = 4) -> torch.Tensor:
        """
        Batch-embed texts with pooling and L2 normalization.

        Args:
            texts: List of text strings
            batch_size: Batch size for inference

        Returns:
            Normalized embeddings [N, D]
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)
                embeddings = self.pooling_fn(
                    outputs.last_hidden_state,
                    batch["attention_mask"],
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def embed_triples(self, triples: List[Dict]) -> List[Dict]:
        """
        Embed triples into four independent views.

        Each result dict contains:
            triple_emb, relation_emb, subj_type_emb, obj_type_emb

        Args:
            triples: List of triple dicts with subject, object, relation,
                     subj_type, obj_type fields

        Returns:
            List of embedding dicts (one per triple)
        """
        texts, index_map = [], []

        for i, t in enumerate(triples):
            subj_type = t.get("subj_type", "entity")
            obj_type  = t.get("obj_type",  "entity")

            texts.append(
                f"A {subj_type} has the relation '{t['relation']}' with a {obj_type}."
            )
            index_map.append((i, "triple"))

            texts.append(f"Relation: {t['relation']}")
            index_map.append((i, "relation"))

            if subj_type:
                texts.append(f"Entity Type: {subj_type}")
                index_map.append((i, "subj_type"))
            if obj_type:
                texts.append(f"Entity Type: {obj_type}")
                index_map.append((i, "obj_type"))

        embs = self.embed_texts(texts)

        results = [
            {
                "id":           t.get("sentence_id", t.get("id")),
                "subject":      t["subject"],
                "object":       t["object"],
                "relation":     t["relation"],
                "subj_type":    t.get("subj_type"),
                "obj_type":     t.get("obj_type"),
                "triple_emb":   None,
                "relation_emb": None,
                "subj_type_emb": None,
                "obj_type_emb": None,
            }
            for t in triples
        ]

        for emb, (i, field) in zip(embs, index_map):
            if field == "triple":
                results[i]["triple_emb"] = emb
            elif field == "relation":
                results[i]["relation_emb"] = emb
            elif field == "subj_type":
                results[i]["subj_type_emb"] = emb
            elif field == "obj_type":
                results[i]["obj_type_emb"] = emb

        return results

    def embed_ontology(self, ontology_triples: List[Dict]) -> List[Dict]:
        """
        Embed ontology patterns into four independent views.

        Each result dict contains:
            triple_emb, relation_emb, subj_type_emb (domain), obj_type_emb (range)

        Args:
            ontology_triples: List of ontology relation schema dicts

        Returns:
            List of embedding dicts (one per ontology pattern)
        """
        texts, index_map = [], []

        for i, ont in enumerate(ontology_triples):
            relation    = ont.get("relation")
            domain_type = ont.get("domain_type", "entity")
            range_type  = ont.get("range_type",  "entity")

            texts.append(
                f"A {domain_type} has the relation '{relation}' with a {range_type}."
            )
            index_map.append((i, "triple"))

            texts.append(f"Relation: {relation}")
            index_map.append((i, "relation"))

            if domain_type:
                texts.append(f"Entity Type: {domain_type}")
                index_map.append((i, "domain_type"))
            if range_type:
                texts.append(f"Entity Type: {range_type}")
                index_map.append((i, "range_type"))

        embs = self.embed_texts(texts)

        results = [
            {
                "relation":     ont.get("relation"),
                "domain_type":  ont.get("domain_type"),
                "range_type":   ont.get("range_type"),
                "triple_emb":   None,
                "relation_emb": None,
                "subj_type_emb": None,
                "obj_type_emb": None,
            }
            for ont in ontology_triples
        ]

        for emb, (i, field) in zip(embs, index_map):
            if field == "triple":
                results[i]["triple_emb"] = emb
            elif field == "relation":
                results[i]["relation_emb"] = emb
            elif field == "domain_type":
                results[i]["subj_type_emb"] = emb
            elif field == "range_type":
                results[i]["obj_type_emb"] = emb

        return results
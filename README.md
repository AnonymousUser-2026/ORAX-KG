# ORAX-KG
Official implementation of ORAX-KG.
---

## Description

ORAX-KG is a unified framework that expands an initial ontology for ontology-grounded knowledge graph construction. Given a corpus and a partial ontology schema, it automatically extracts relation triples, aligns them against known relations, clusters the unaligned ones, and validates the resulting clusters with an LLM to propose genuinely new ontology relations.

### Key Components

**ORAX-KG combines:**

1. **LLM-based extraction** — An instruction-tuned model extracts (subject, relation, object) triples from sentences, guided by the known portion of the ontology.
2. **Semantic alignment** — Extracted triples are embedded and matched against known ontology patterns; unmatched triples are flagged for clustering.
3. **Consensus clustering** — A multi-algorithm ensemble groups unaligned triples by semantic similarity.
4. **LLM validation** — A second LLM pass inspects each cluster, decides whether it represents a genuinely new relation, and proposes a labeling.

---

## Dataset

The TACRED-family datasets can be obtained from the following sources:

- **TACRED**: [LDC2018T24](https://catalog.ldc.upenn.edu/LDC2018T24)
- **TACREV**: [GitHub](https://github.com/DFKI-NLP/tacrev)
- **Re-TACRED**: [GitHub](https://github.com/gstoica27/Re-TACRED)

**After downloading, organize the data as follows:**
```
data/
└── raw_data/
    ├── TACRED/
    │   ├── train.json
    │   ├── dev.json
    │   └── test.json
    ├── TACREV/
    │   ├── train.json
    │   ├── dev.json
    │   └── test.json
    └── ReTACRED/
        ├── train.json
        ├── dev.json
        └── test.json
```

**Note:** We provide a small synthetic sample dataset in `data/raw_data/DATASET_EXAMPLE.json` for quick testing without full dataset access.

---

## Installation

Before running the modules, ensure all dependencies are installed:
```bash
# Clone the repository
git clone https://github.com/AnonymousUser-2026/ORAX-KG.git
cd ORAX-KG

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Extract Ontology Schemas

Extract relation schema from training data (one-time setup):
```bash
python scripts/extract_schema.py \
    --train data/raw_data/ReTACRED/train.json \
    --output data/ontologies/ReTACRED_ontology.json
```

---

### Step 1: Ontology-Guided Triple Extraction

Extract relation triples guided by the initial ontology schema using an LLM:
```bash
python scripts/run_extraction.py \
    --data data/raw_data/ReTACRED/test.json \
    --schema data/ontologies/ReTACRED_ontology.json \
    --output results/test_run \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-14B
```

---

### Step 2: Triple Embeddings and Alignment

Embed the extracted triples and align them against the initial ontology:
```bash
python scripts/run_alignment.py \
    --run-dir results/test_run \
    --output results/test_run \
    --embedder Qwen/Qwen3-Embedding-8B \
    --threshold 0.90 \
    --device cuda
```

---

### Step 3: Clustering

Cluster unaligned triples into semantically coherent groups:
```bash
python scripts/run_clustering.py \
    --run-dir results/test_run \
    --similarity-threshold 0.60 \
    --consensus-runs 3
```

---

### Step 4: LLM-Based Validation and Ontology Expansion

Validate the clusters and label novel relations before integration into the ontology:
```bash
python scripts/run_validation.py \
    --extraction-dir results/test_run \
    --clusters results/test_run/04_clustering/clusters.json \
    --embeddings results/test_run/03_alignment/extracted_embeddings.pt \
    --output results/test_run/05_validation \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-32B \
    --min-cluster-size 5 \
    --min-confidence Medium
```

---

### Run Full Pipeline

Execute the complete pipeline end-to-end from a configuration file:
```bash
python scripts/run_full_pipeline.py --config configs/config.yaml
```
---

## Repository Structure
```
ORAX-KG/
│
├── configs/                     # Configuration files
│   ├── config.yaml
│   └── test_config.yaml         # For testing
│
├── README.md
├── requirements.txt
├── logger.py
├── run_predictions.ipynb
│
├── scripts/                     # Entry-point scripts
│   ├── run_extraction.py
│   ├── run_alignment.py
│   ├── run_clustering.py
│   ├── run_validation.py
│   ├── run_full_pipeline.py
│   └── extract_schema.py
│
├── src/                         # Core library
│   ├── extraction/              # LLM extraction + prompt generation
│   │   ├── llm_extractor.py
│   │   └── metrics.py 
│   ├── alignment/               # Embedding + similarity computation
│   │   ├── embeddings.py
│   │   ├── similarity.py
│   │   ├── aligner.py
│   │   └── metrics.py
│   ├── clustering/              # Consensus clustering + evaluation
│   │   ├── consensus.py
│   │   ├── evaluation.py
│   │   └── utils.py
│   └── validation/              # LLM cluster validation
│       └── llm_validator.py
│
├── data/
│   ├── raw_data/
│   │   └── DATASET_EXAMPLE/     # Small synthetic sample for testing
│   │       
│   └── ontologies/
│       ├── ReTACRED_ontology.json
│       └── TACRED_ontology.json
│
├── results/                     # Created at runtime; one sub-folder per run
└── logs/                        # Run logs
```
---

## Output Structure

Each run creates a timestamped directory under `results/` with the following layout:
```
results/retacred_qwen_20260310_142501/
│
├── 00_config.yaml               # Saved run configuration
│
├── 01_ontology/
│   ├── full_ontology.json       # Complete relation schema
│   ├── known_ontology.json      # Ontology triples shown to the LLM
│   ├── hidden_ontology.json     # Ontology triples withheld (evaluation targets)
│   └── split_metadata.json      # Split statistics
│
├── 02_extraction/
│   ├── extractions.jsonl        # LLM-extracted triples (one JSON per line)
│   ├── test_samples.json        # Sampled input sentences
│   └── ground_truth.json        # Ground truth relation labels
│
├── 03_alignment/
│   ├── alignment_results.json   # Per-triple alignment decisions
│   ├── extracted_embeddings.pt  # Saved embedding tensors
│   ├── ontology_embeddings.pt   # Ontology embeddings
│   └── statistics.json          # Alignment statistics
│
├── 04_clustering/
│   ├── cluster_embeddings.pt    # Full cluster items with embedding tensors
│   ├── clusters.json            # Cluster assignments with triple metadata
│   ├── metrics.json             # Flat quality metrics
│   └── stratified_metrics.json  # Per-subset evaluation (novel/known/overall)
│
└── 05_validation/
    ├── novel_relations.json     # Accepted novel relation candidates
    ├── rejected_candidates.json # Rejected clusters with reasons
    └── 00_config.yaml           # Validation-specific configuration
```
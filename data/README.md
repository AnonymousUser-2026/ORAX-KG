# Datasets

The full TACRED-family datasets are not included in this repository due to licensing restrictions. Download them from the official sources:

- **TACRED**: [LDC2018T24](https://catalog.ldc.upenn.edu/LDC2018T24) (requires LDC license)
- **TACREV**: [GitHub - DFKI-NLP/tacrev](https://github.com/DFKI-NLP/tacrev) (requires TACRED license)
- **Re-TACRED**: [GitHub - gstoica27/Re-TACRED](https://github.com/gstoica27/Re-TACRED) (CC BY-SA 4.0)

After downloading, organize the datasets as follows:
```
data/
├── raw_data/
│   ├── TACRED/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── TACREV/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── ReTACRED/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   └── DATASET_EXAMPLE.json 
└── ontologies/
    ├── ReTACRED_ontology.json
    └── TACRED_ontology.json
```

**Note:**DATASET_EXAMPLE.json is a small synthetic file in the TACRED JSON format. It is enough to run the full pipeline end-to-end and verify the setup. Point config.yaml at it:

```yaml
data:
  test_path: "data/raw_data/DATASET_EXAMPLE.json"
```

---

## Generating an Ontology Schema
If you are working with a new dataset or need to extract the ontology schema from training data:
```bash
python scripts/extract_schema.py \
    --train data/raw_data/ReTACRED/train.json \
    --output data/ontologies/ReTACRED_ontology.json
```


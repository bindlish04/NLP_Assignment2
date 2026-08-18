# Solution – PS4 Knowledge-Grounded RAG

All project code and the submission notebook live in this folder.

**Start here:** see the [main README](../README.md) for clone, setup, OSHA Virtual Lab steps, and screenshot instructions.

## Run locally

```bash
cd solution
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m jupyter notebook PS4_Knowledge_Grounded_RAG.ipynb
```

## Key files

| File | Purpose |
|------|---------|
| `PS4_Knowledge_Grounded_RAG.ipynb` | Submission notebook |
| `Group44_PS4_Report.md` | Report draft for PDF export |
| `src/` | Ontology, KG, retrieval, RAG modules |
| `data/documents/` | 31 Wikipedia articles (bundled) |

Figures are written to `outputs/` when the notebook runs (not committed to Git).

# Execution Instructions for Evaluators

This document explains how to run **PS4 � Knowledge-Grounded RAG** independently on a fresh machine. No prior knowledge of the student codebase is required.

---

## 1. What You Are Evaluating

| Item | Location |
|------|----------|
| Main submission notebook | `PS4_Knowledge_Grounded_RAG.ipynb` |
| Implementation modules | `src/` |
| Domain documents (31 Wikipedia articles, offline-ready) | `data/documents/` |
| Generated figures | `outputs/` (created when notebook runs) |

**Domain:** Climate Change and Environmental Science  
**Pipeline:** Ontology ? Knowledge Graph ? Semantic Retrieval ? RAG ? Grounding Analysis

---

## 2. System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.10 or newer (tested on 3.12) |
| RAM | 8 GB (16 GB recommended) |
| Disk | ~2 GB free (Python packages + Hugging Face models) |
| Internet | Required **only on first run** to download models; documents are bundled |

**Optional:** NVIDIA GPU with CUDA speeds up LLM inference but is **not required**.

---

## 3. Step-by-Step Setup

Open a terminal in the `solution/` folder (the directory containing this file and the notebook).

### Step 3.1 � Install Python dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

> **Note:** If `pip` is not found, always use `python3 -m pip` instead.

### Step 3.2 � Verify document cache (optional)

```bash
ls data/documents/*.txt | wc -l
```

Expected output: **31** (assignment requires at least 30).  
If fewer than 30 files exist and internet is available:

```bash
python3 -c "from pathlib import Path; from src.data_collection import collect_documents; collect_documents(Path('data/documents'))"
```

---

## 4. Running the Submission

### Option A � Jupyter Notebook (recommended for grading)

```bash
python3 -m jupyter notebook PS4_Knowledge_Grounded_RAG.ipynb
```

In Jupyter: **Kernel ? Restart & Run All**

Allow **10�20 minutes** on CPU for the first complete run (model download + RAG over 25 questions).

### Option B � Non-interactive execution (batch)

```bash
python3 -m jupyter nbconvert \
  --to notebook \
  --execute PS4_Knowledge_Grounded_RAG.ipynb \
  --output PS4_Knowledge_Grounded_RAG.ipynb \
  --ExecutePreprocessor.timeout=900
```

### Option C � One-command script

```bash
chmod +x run.sh
./run.sh
```

---

## 5. Expected Outputs (Checklist)

After a successful run you should see:

- [ ] **Section 3:** Document table with 31 articles and word counts
- [ ] **Task 1:** Ontology table (6 entity classes, 6 relationship types) + schema figure
- [ ] **Task 2:** KG stats with **? 75 entities** and **? 150 relationships**; 3 subgraph plots
- [ ] **Task 3:** Retrieval results for **25 questions**; Precision@3, Recall@3, MRR
- [ ] **Task 4:** RAG answers with evidence, sources, and grounding assessment
- [ ] **Task 5:** LLM-only vs RAG comparison (12 questions); ? 5 problematic-answer analyses

Figures are written to:

```
outputs/ontology_schema.png
outputs/subgraph_global_warming.png
outputs/subgraph_paris_agreement.png
outputs/subgraph_carbon_dioxide.png
```

---

## 6. Models and External Resources

| Component | Model / Tool | Purpose |
|-----------|--------------|---------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Semantic retrieval |
| LLM | `google/flan-t5-small` | Answer generation (pretrained, not fine-tuned) |
| NER | spaCy `en_core_web_sm` | Entity extraction for knowledge graph |
| Graph | NetworkX | In-memory knowledge graph storage |

Models download automatically from Hugging Face on first use (~500 MB total).

Configuration (model names, chunk size, thresholds) is centralized in `src/config.py`.

---

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found: pip` | Use `python3 -m pip install -r requirements.txt` |
| `command not found: jupyter` | Use `python3 -m jupyter notebook ...` |
| spaCy model missing | Run `python3 -m spacy download en_core_web_sm` |
| Hugging Face download slow/fails | Check internet; retry; or set `HF_TOKEN` env var |
| Out of memory | Close other apps; notebook already uses `flan-t5-small` (lightweight) |
| Notebook timeout | Increase `--ExecutePreprocessor.timeout=900` (15 min) |
| Empty `outputs/` folder | Re-run Task 1�2 cells or full notebook; figures save to `outputs/` |

---

## 8. Quick Sanity Test (2 minutes)

Runs core logic without full notebook execution:

```bash
python3 << 'EOF'
from pathlib import Path
from src.data_collection import load_local_documents
from src.ontology import build_climate_ontology
from src.kg_builder import build_knowledge_graph
from src.retrieval import chunk_documents, SemanticRetriever

docs = load_local_documents(Path("data/documents"))
ontology = build_climate_ontology()
kg = build_knowledge_graph(docs, ontology)
chunks = chunk_documents(docs)
retriever = SemanticRetriever()
retriever.fit(chunks)
hits = retriever.retrieve("What causes global warming?", top_k=1)

print(f"Documents: {len(docs)}")
print(f"KG: {kg.num_entities} entities, {kg.num_relationships} relationships")
print(f"Chunks: {len(chunks)}")
print(f"Top retrieval: {hits[0].chunk.doc_title} (score={hits[0].score:.3f})")
EOF
```

Expected: 31 documents, 75+ entities, 150+ relationships, non-empty retrieval hit.

---

## 9. Contact

Update the User-Agent contact email in `src/data_collection.py` if re-fetching Wikipedia data.

For assignment metadata (group number, BITS IDs), see **Section 1** of the notebook.

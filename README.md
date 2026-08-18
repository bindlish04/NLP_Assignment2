# NLP Assignment 2 – Group 44 (PS4)

**Knowledge-Grounded Question Answering using Knowledge Graphs and RAG**

| | |
|---|---|
| **Course** | Natural Language Processing (S2-25_DSECLZG530) |
| **Group** | 44 |
| **Domain** | Climate Change & Environmental Science |

### Team

| Name | BITS ID |
|------|---------|
| ABHAY BINDLISH | 2024DC04243 |
| DHARMAVARAPU MOUNIKA | 2024DC04245 |
| E VIVIN RAJ | 2024DC04240 |
| GOVIND GOPAL GOEL | 2024DC04244 |

---

## Quick start (local machine)

```bash
git clone https://github.com/bindlish04/NLP_Assignment2.git
cd NLP_Assignment2/solution

python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

python3 -m jupyter notebook PS4_Knowledge_Grounded_RAG.ipynb
```

In Jupyter: **Kernel ? Restart & Run All** (first run takes 10–20 min on CPU; downloads ~500 MB of models).

> Use `python3 -m pip` if `pip` is not found.

---

## Repository layout

```
solution/
??? PS4_Knowledge_Grounded_RAG.ipynb   # Main notebook (submission artifact)
??? Group44_PS4_Report.md              # Report draft (export to PDF for submission)
??? requirements.txt
??? run.sh                             # Optional one-command setup + execute
??? data/documents/                    # 31 cached Wikipedia articles (offline-ready)
??? src/                               # Implementation modules used by the notebook
```

---

## BITS OSHA Virtual Lab – step-by-step

Use these steps when running on the **OSHA Virtual Lab** for submission and screenshots.

### 1. Open the lab

1. Log in to the BITS OSHA Virtual Lab portal.
2. Launch a **Python 3** / **Jupyter** environment (Python 3.10+ recommended).
3. Upload this repo or clone inside the lab:

```bash
git clone https://github.com/bindlish04/NLP_Assignment2.git
cd NLP_Assignment2/solution
```

### 2. Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 3. Run the notebook

```bash
python3 -m jupyter notebook PS4_Knowledge_Grounded_RAG.ipynb
```

Or from the Jupyter file browser: open the notebook ? **Kernel ? Restart & Run All**.

Wait until **all 8 code cells** finish (KG build + RAG cells are the slowest).

### 4. Verify outputs before screenshots

Confirm the following are visible in the notebook:

| Task | What to check |
|------|----------------|
| Section 3 | `31` documents, word count table |
| Task 1 | Ontology table + schema figure |
| Task 2 | KG stats (>= 75 entities, >= 150 relationships) + 3 subgraph figures |
| Task 3 | Retrieval table for 25 questions; `mean_precision@3`, `mean_recall@3`, `mean_mrr` |
| Task 4 | RAG table with evidence, answers, sources |
| Task 5 | LLM-only vs RAG comparison (12 questions) + problematic-answer analysis |

### 5. Screenshots to capture (for report Appendix B)

Take **full-screen screenshots** showing:

1. **Notebook overview** – Section 1 with Group 44 member details visible.
2. **Task 2** – Knowledge graph statistics printed in the output.
3. **Task 3** – Retrieval metrics (`mean_precision@3`, `mean_recall@3`, `mean_mrr`).
4. **Task 4** – At least one RAG row showing question, evidence, answer, and sources.
5. **Task 5** – LLM-only vs RAG comparison table.
6. **Run completion** – Last cell executed successfully (no error traceback).

Save screenshots as `osha_screenshot_1.png`, `osha_screenshot_2.png`, etc.

### 6. Insert screenshots into the report

1. Open `Group44_PS4_Report.md` (or export to Word/PDF).
2. Paste screenshots under **Appendix B – BITS OSHA Virtual Lab Execution**.
3. Export final report as **`Group44_PS4_Report.pdf`**.

### 7. Download from lab (before session ends)

Download these from the virtual lab to your machine:

- `PS4_Knowledge_Grounded_RAG.ipynb` (with all outputs saved)
- Screenshot files
- Updated report PDF

---

## Optional: non-interactive run

```bash
cd solution
chmod +x run.sh
./run.sh
```

Or:

```bash
python3 -m jupyter nbconvert \
  --to notebook \
  --execute PS4_Knowledge_Grounded_RAG.ipynb \
  --output PS4_Knowledge_Grounded_RAG.ipynb \
  --ExecutePreprocessor.timeout=900
```

---

## Models used

| Component | Model |
|-----------|--------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | `google/flan-t5-small` |
| NER | spaCy `en_core_web_sm` |

Settings can be changed in `src/config.py`.

---

## Submission checklist (Group 44)

- [ ] Section 1 filled with group details (already done in notebook)
- [ ] All notebook cells executed; outputs visible
- [ ] Report PDF with figures and metrics matching notebook
- [ ] OSHA Virtual Lab screenshots in report Appendix B
- [ ] Zip / upload per course portal instructions

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pip` not found | Use `python3 -m pip install -r requirements.txt` |
| spaCy model error | `python3 -m spacy download en_core_web_sm` |
| Hugging Face download fails | Check internet; retry; set `HF_TOKEN` if rate-limited |
| Kernel dies (OOM) | Restart kernel; close other tabs; run cells one section at a time |
| Missing documents | `data/documents/` must contain 31 `.txt` files (included in repo) |

---

## References

- Notebook: `solution/PS4_Knowledge_Grounded_RAG.ipynb`
- Report draft: `solution/Group44_PS4_Report.md`
- Wikipedia data source: https://en.wikipedia.org/

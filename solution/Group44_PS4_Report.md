# NLP Assignment 2 - Problem Statement 4

## Knowledge-Grounded Question Answering using Knowledge Graphs and RAG

---

**Course:** Natural Language Processing (S2-25_DSECLZG530)  
**Problem Statement Number:** PS4  
**Group Number:** Group 44  
**Date of Submission:** August 2026

### Team Members

| Name | BITS ID | Contribution |
|------|---------|--------------|
| ABHAY BINDLISH | 2024DC04243 | 25% |
| DHARMAVARAPU MOUNIKA | 2024DC04245 | 25% |
| E VIVIN RAJ | 2024DC04240 | 25% |
| GOVIND GOPAL GOEL | 2024DC04244 | 25% |

**Companion notebook:** `PS4_Knowledge_Grounded_RAG.ipynb` (all metrics and tables below are traceable to executed notebook cells).

---

## 1. Introduction

### 1.1 Problem Background

Large language models (LLMs) can generate fluent answers but often produce **hallucinations**-statements not supported by factual evidence. **Knowledge graphs (KGs)** organize domain entities and relationships in a structured form, while **ontologies** define the vocabulary and semantics of that structure. **Semantic retrieval** finds relevant text using meaning rather than keyword overlap alone. **Retrieval-Augmented Generation (RAG)** combines retrieval with an LLM so answers are conditioned on external evidence, improving reliability for domain-specific question answering.

### 1.2 Objectives

1. Design a domain ontology and construct a knowledge graph from public climate-domain documents.
2. Implement semantic retrieval over document chunks and evaluate retrieval quality.
3. Build a RAG pipeline (Question ? Retrieval ? Evidence ? LLM ? Answer).
4. Compare LLM-only vs RAG answers, summarize evidence, and analyse hallucination patterns.

---

## 2. Domain and Document Description

### 2.1 Domain Selection

We selected **Climate Change and Environmental Science** because it has rich public documentation, clear entity types (phenomena, gases, policies, organizations, regions, energy sources), and well-defined causal and regulatory relationships suitable for ontology and KG construction.

### 2.2 Document Sources

All documents are **public English Wikipedia articles** fetched via the MediaWiki API:

- Base URL: https://en.wikipedia.org/
- Cached locally in `data/documents/` (31 `.txt` files with Title and Source metadata)

Representative articles include: *Climate change*, *Global warming*, *Paris Agreement*, *IPCC*, *Renewable energy*, *Sea level rise*, *Deforestation*, and others listed in notebook Section 3.

### 2.3 Dataset / Document Statistics

| Statistic | Value |
|-----------|-------|
| Number of documents | 31 |
| Total words | 208,459 |
| Retrieval chunks (180-word windows, 40-word overlap) | 1,500 |
| Source | Wikipedia (public) |

### 2.4 Sample Document Excerpt

From *Climate change* (https://en.wikipedia.org/wiki/Climate_change): the corpus contains full plain-text extracts covering causes, impacts, mitigation, and policy responses. Each file header records title and source URL for citation in RAG outputs.

---

## 3. Ontology and Knowledge Graph

### 3.1 Ontology Design

We defined **six entity classes** and **six relationship types** (assignment minimum: 5 each).

**Entity classes**

| Class | Description | Example attributes |
|-------|-------------|-------------------|
| ClimatePhenomenon | Observable climate processes/impacts | severity, time_scale, geographic_scope |
| GreenhouseGas | Heat-trapping atmospheric gases | chemical_formula, GWP, lifetime |
| PolicyInstrument | Laws/treaties/economic mechanisms | jurisdiction, start_year, target |
| Organization | Research/governance institutions | role, founding_year, headquarters |
| GeographicRegion | Places/ecosystems | latitude_band, biome |
| EnergySource | Fuels/technologies for energy | carbon_intensity, capacity_factor |

**Relationship types**

| Relation | Meaning | Example |
|----------|---------|---------|
| causes | Direct contribution to a phenomenon | (carbon dioxide, causes, global warming) |
| mitigated_by | Reduction via policy/technology | (deforestation, mitigated_by, reforestation policy) |
| regulated_by | Governance by institution/treaty | (methane emissions, regulated_by, Paris Agreement) |
| located_in | Geographic association | (Arctic sea ice decline, located_in, Arctic) |
| emits | GHG release from energy source | (coal, emits, carbon dioxide) |
| contributes_to | Institution/report links to phenomenon | (IPCC, contributes_to, attribution of climate change) |

### 3.2 Ontology Visualization

![Ontology Schema](outputs/ontology_schema.png)

*Figure 1: Ontology schema (see notebook Task 1).*

### 3.3 Knowledge Graph Construction

**Entity extraction**

1. **Lexicon matching** - domain phrases mapped to ontology classes (e.g., "Paris Agreement" ? PolicyInstrument).
2. **spaCy NER** - `en_core_web_sm` labels (ORG, GPE, LOC, etc.) mapped to ontology types.

**Relationship extraction**

1. **Regex patterns** over sentences (e.g., *X causes Y*, *X emits Y*).
2. **Typed co-occurrence** - when two entity types appear in the same sentence/document, infer relations from ontology rules (e.g., GreenhouseGas + ClimatePhenomenon ? causes).

**Graph technology:** `networkx.MultiDiGraph` (in-memory); visualized with Matplotlib.

### 3.4 Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| Entities | 2,149 |
| Relationships | 2,711 |
| Entity types | GeographicRegion (667), Organization (1,317), EnergySource (67), ClimatePhenomenon (67), GreenhouseGas (14), PolicyInstrument (17) |
| Relationship types | contributes_to (1,158), located_in (705), regulated_by (376), causes (249), mitigated_by (118), emits (105) |

### 3.5 Knowledge Graph Visualizations

![Subgraph: Global Warming](outputs/subgraph_global_warming.png)

*Figure 2: Ego subgraph centred on "global warming".*

![Subgraph: Paris Agreement](outputs/subgraph_paris_agreement.png)

*Figure 3: Ego subgraph centred on "Paris Agreement".*

![Subgraph: Carbon Dioxide](outputs/subgraph_carbon_dioxide.png)

*Figure 4: Ego subgraph centred on "carbon dioxide".*

### 3.6 Representative Triples

| Subject | Relation | Object | Semantics |
|---------|----------|--------|-----------|
| greenhouse effect | causes | global warming | Physical mechanism linking effect to warming |
| ipcc | contributes_to | climate change | Institution synthesizing climate evidence |
| coal | emits | carbon dioxide | Fossil fuel GHG emission |
| paris agreement | mitigated_by | climate change | Policy intended to limit warming |
| arctic | located_in | sea level rise | Regional climate impact context |

*(Full triple table in notebook Task 2.)*

---

## 4. Semantic Search and Retrieval

### 4.1 Document Preparation

- Documents cleaned (citation markers removed, whitespace normalized).
- Chunked into **180-word windows** with **40-word overlap** ? 1,500 chunks.
- Embedded with **sentence-transformers/all-MiniLM-L6-v2**; L2-normalized vectors for cosine similarity.

### 4.2 Query Set

**25 domain-specific test questions** covering causes, policies, gases, impacts, and mitigation (notebook Task 3).

### 4.3 Retrieval Results

For each question, top-3 chunks are returned with **similarity scores** (0-1). Example:

| Question | Rank-1 Document | Score |
|----------|-----------------|-------|
| What causes global warming? | Global warming | 0.674 |
| How does the greenhouse effect work? | Greenhouse effect | 0.651 |
| What is the Paris Agreement? | Paris Agreement | 0.712 |

*(Full retrieval table for all 25 questions in notebook Task 3.)*

### 4.4 Retrieval Evaluation

**Metrics:** Precision@3, Recall@3, Mean Reciprocal Rank (MRR).

**Justification:** Each question expects a small set of highly relevant chunks; Precision@3 measures retrieval accuracy in the top results; Recall@3 checks coverage; MRR rewards ranking the first relevant chunk early.

| Metric | Value |
|--------|-------|
| Mean Precision@3 | **0.6667** |
| Mean Recall@3 | **0.9800** |
| Mean MRR | **1.0000** |

Relevance labels combine top retrieved chunks with manual overrides for key questions (notebook Task 3, Cell 6).

---

## 5. RAG-Based Question Answering

### 5.1 RAG Architecture

```
User Question ? Semantic Retriever (top-3 chunks) ? Evidence summary
                ? FLAN-T5-small (pretrained) ? Grounded answer + sources
```

Retrieval output (chunks, scores, URLs) is displayed **separately** from LLM-generated text in the notebook.

### 5.2 Model and Configuration

| Component | Model / Setting |
|-----------|---------------|
| Embeddings | all-MiniLM-L6-v2 |
| LLM | google/flan-t5-small (no fine-tuning) |
| Top-K retrieval | 3 |
| Max new tokens | 100 |
| Grounding threshold | 0.18 lexical overlap |

### 5.3 Question Answering Results

**25 questions** answered with: question, evidence summary, generated answer, sources (document title + Wikipedia URL + score), and grounding assessment. See notebook Task 4 table.

**Sample (Question: What causes global warming?)**

- **Retrieved source:** Global warming (Wikipedia), score ? 0.67  
- **RAG answer:** *"climate change, and climate change has an increasingly large impact on the environment."*  
- **Grounded:** Yes (lexical overlap with evidence)

### 5.4 Grounding Assessment

Grounding is assessed by:

1. Detecting abstention phrases (*"I do not know"*, etc.).
2. Measuring **lexical overlap** between answer and retrieved evidence (threshold 0.18).

Most RAG answers show overlap with retrieved text; failures are flagged when overlap is low despite high retrieval scores.

---

## 6. Grounding, Summarization and Hallucination Analysis

### 6.1 LLM-only vs RAG Comparison (12 questions)

| Question | LLM-only (excerpt) | RAG (excerpt) | Grounded? | Category |
|----------|-------------------|---------------|-----------|----------|
| What causes global warming? | climate change | climate change impact on the environment | Yes | acceptable |
| How does the greenhouse effect work? | chemical change in the atmosphere | displaystyle G =  (formula fragment) | Yes | acceptable |
| What is the Paris Agreement? | Paris Agreement | ratify the agreement | Yes | incomplete synthesis |
| What are the main greenhouse gases? | greenhouse gases | carbon monoxide, and methane | Yes | incomplete synthesis |
| How does deforestation affect climate change? | deforestation is a **result** of climate change | net. | **No** | unsupported generation |
| What is the role of the IPCC? | IPCC is member of **UN Security Council** (false) | the Sixth Assessment Report cycle | Yes | incomplete synthesis |

*(Full 12-row comparison in notebook Task 5.)*

**Observation:** LLM-only answers often hallucinate (e.g., IPCC as UN Security Council member, reversed causality for deforestation). RAG answers stay closer to retrieved text but can be **fragmentary** due to the small FLAN-T5 model.

### 6.2 Evidence Summarization

Extractive summaries (longest 2 sentences from top chunks) are shown for representative questions in notebook Task 5-for example, summarizing IPCC assessment cycles or greenhouse-gas definitions from retrieved Wikipedia passages.

### 6.3 Hallucination Analysis (5+ problematic cases)

| # | Question | Category | Cause |
|---|----------|----------|-------|
| 1 | Deforestation vs climate change | **Unsupported generation** | LLM produced "net." despite evidence; low semantic synthesis |
| 2 | Paris Agreement | **Incomplete synthesis** | Answer too short; missed treaty scope |
| 3 | Main greenhouse gases | **Incomplete synthesis** | Listed partial gases; omitted N?O, fluorinated gases |
| 4 | IPCC role (LLM-only) | **Unsupported generation** | No retrieval; model invented UN Security Council role |
| 5 | Solar/wind emissions | **Incomplete synthesis** | Answer "energy costs" - off-focus fragment from evidence |
| 6 | Ocean acidification | **Incomplete synthesis** | Answer "Earth's ocean" - lacks definition of acidification mechanism |

### 6.4 Reliability Discussion

- **Factual correctness:** RAG reduces blatant LLM-only hallucinations when retrieval scores are high (>0.6).
- **Completeness:** Small seq2seq model often returns phrases rather than full explanations ? incomplete synthesis.
- **Grounding:** Lexical overlap heuristic marks most RAG outputs as grounded; human review still needed for semantic correctness (e.g., "carbon monoxide" vs "carbon dioxide").
- **Retrieval failures:** Low-score retrievals (<0.25) correlate with irrelevant context and weaker answers.

---

## 7. Results Summary

| Component | Outcome |
|-----------|---------|
| Documents | 31 Wikipedia articles, 208K words |
| Ontology | 6 classes, 6 relations |
| Knowledge graph | 2,149 entities, 2,711 relationships |
| Retrieval | P@3 = 0.67, R@3 = 0.98, MRR = 1.0 |
| RAG | 25 QA pairs with sources and grounding flags |
| Analysis | 12 LLM vs RAG comparisons; 6 failure cases categorised |

---

## 8. Conclusion

We implemented an end-to-end knowledge-grounded QA system for the climate domain: ontology design, KG construction with lexicon + NER + relation heuristics, Sentence-BERT retrieval, and FLAN-T5 RAG. Retrieval metrics indicate strong rank-1 performance (MRR = 1.0). RAG improves grounding over LLM-only baselines but **incomplete synthesis** remains the main weakness with a small pretrained model.

**Future work:** Human relevance judgments, cross-encoder reranking, graph-aware retrieval (KG ? entity-linked chunks), larger instruction-tuned LLMs, and Neo4j persistence for the knowledge graph.

---

## 9. References

1. Wikipedia articles - https://en.wikipedia.org/
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
3. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
4. Hugging Face Transformers - https://huggingface.co/docs/transformers
5. NetworkX - https://networkx.org/
6. spaCy - https://spacy.io/
7. Chollet, F. et al. FLAN-T5 - https://huggingface.co/google/flan-t5-small

---

## Appendix A - Additional Outputs

- Full retrieval, RAG, and comparison tables: `PS4_Knowledge_Grounded_RAG.ipynb`
- Source code: `src/` directory
- Cached documents: `data/documents/`

## Appendix B - BITS OSHA Virtual Lab Execution

**[INSERT SCREENSHOT HERE]**

*Add screenshot(s) showing successful notebook execution on the BITS OSHA Virtual Lab before final PDF submission.*

---

### Converting this report to PDF

From the `solution/` folder:

```bash
# Option 1: Pandoc (if installed)
pandoc Group44_PS4_Report.md -o Group44_PS4_Report.pdf --resource-path=.

# Option 2: Open in Word / Google Docs ? Export as PDF
# Option 3: Paste into Overleaf with graphicx for figures
```

Ensure `outputs/*.png` figures are included when exporting.

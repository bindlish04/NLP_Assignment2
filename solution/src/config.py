"""
Shared configuration for the PS4 RAG pipeline.

Centralizing constants here keeps the notebook and scripts consistent and makes
it easy for an evaluator to change models or thresholds in one place.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths (relative to the solution/ folder) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Document collection ---
MIN_DOCUMENTS = 30
MIN_WORDS_PER_DOCUMENT = 80
WIKIPEDIA_REQUEST_DELAY_SEC = 0.3

# --- Chunking for semantic retrieval ---
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 40

# --- Knowledge graph minimums (assignment requirements) ---
MIN_KG_ENTITIES = 75
MIN_KG_RELATIONSHIPS = 150

# --- Retrieval ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 3
EMBEDDING_BATCH_SIZE = 64

# --- RAG / LLM ---
LLM_MODEL = "google/flan-t5-small"  # smaller/faster; use flan-t5-base for higher quality
LLM_MAX_NEW_TOKENS = 100
LLM_MAX_INPUT_TOKENS = 512
GROUNDING_OVERLAP_THRESHOLD = 0.18

# --- Evaluation ---
RETRIEVAL_EVAL_K = 3
RELEVANCE_SEED_K = 5

"""
Task 3: Semantic search over document chunks.

Documents are split into overlapping word windows, embedded with Sentence-BERT,
and ranked by cosine similarity at query time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    MIN_CHUNK_WORDS,
    RETRIEVAL_TOP_K,
)


@dataclass
class DocumentChunk:
    """One retrievable text unit with traceable source metadata."""

    chunk_id: str
    doc_title: str
    doc_file: str
    source_url: str
    text: str


@dataclass
class RetrievalResult:
    """A ranked chunk returned for a user question."""

    query: str
    chunk: DocumentChunk
    score: float
    rank: int


def chunk_documents(
    documents: Iterable[dict],
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[DocumentChunk]:
    """
    Split each document into fixed-size overlapping word windows.

    Overlap helps avoid cutting important sentences at chunk boundaries.
    Very short tail chunks are dropped to keep embedding quality stable.
    """
    chunks: list[DocumentChunk] = []
    step = max(chunk_size - overlap, 1)

    for doc in documents:
        words = re.sub(r"\s+", " ", doc["text"]).split()
        for idx, start in enumerate(range(0, len(words), step)):
            piece = " ".join(words[start : start + chunk_size])
            if len(piece.split()) < MIN_CHUNK_WORDS:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc['file_name']}::chunk_{idx}",
                    doc_title=doc["title"],
                    doc_file=doc["file_name"],
                    source_url=doc.get("url", ""),
                    text=piece,
                )
            )
    return chunks


class SemanticRetriever:
    """
    Embedding-based retriever.

    After fit(), chunk embeddings are L2-normalized so cosine similarity
    reduces to a fast dot product at query time.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row so dot(a, b) equals cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit(self, chunks: list[DocumentChunk]) -> None:
        """Encode all chunks once and cache normalized embeddings."""
        self._load_model()
        self.chunks = chunks
        texts = [c.text for c in chunks]
        raw = self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )
        self.embeddings = self._normalize_rows(np.asarray(raw, dtype=np.float32))

    def _score_query(self, query: str) -> np.ndarray:
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Call fit() before retrieve().")
        query_vec = self._model.encode([query], show_progress_bar=False)
        query_vec = self._normalize_rows(np.asarray(query_vec, dtype=np.float32))
        return np.dot(self.embeddings, query_vec[0])

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievalResult]:
        """Return the top-k most similar chunks for a single question."""
        self._load_model()
        scores = self._score_query(query)
        # argpartition is O(n); full sort is O(n log n). We only need top-k.
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            RetrievalResult(
                query=query,
                chunk=self.chunks[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank,
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]

    def retrieve_batch(self, queries: list[str], top_k: int = RETRIEVAL_TOP_K) -> dict[str, list[RetrievalResult]]:
        """Retrieve for many questions in one embedding batch (faster in notebooks)."""
        self._load_model()
        if self.embeddings is None:
            raise RuntimeError("Call fit() before retrieve_batch().")

        query_vecs = self._model.encode(queries, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
        query_vecs = self._normalize_rows(np.asarray(query_vecs, dtype=np.float32))
        all_scores = np.dot(query_vecs, self.embeddings.T)

        results: dict[str, list[RetrievalResult]] = {}
        k = min(top_k, self.embeddings.shape[0])
        for query, scores in zip(queries, all_scores):
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            results[query] = [
                RetrievalResult(
                    query=query,
                    chunk=self.chunks[int(idx)],
                    score=float(scores[int(idx)]),
                    rank=rank,
                )
                for rank, idx in enumerate(top_indices, start=1)
            ]
        return results

"""
Task 3 (evaluation): Retrieval quality metrics.

Implements Precision@K, Recall@K, and Mean Reciprocal Rank (MRR) for the
semantic search component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import RELEVANCE_SEED_K, RETRIEVAL_EVAL_K
from .retrieval import SemanticRetriever


@dataclass
class QueryJudgment:
    """Ground-truth (or pseudo) relevant chunk IDs for one test question."""

    question: str
    relevant_chunk_ids: set[str]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for cid in top_k if cid in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant chunks found within top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for cid in top_k if cid in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant result; 0 if none appear."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retriever: SemanticRetriever,
    judgments: Iterable[QueryJudgment],
    k: int = RETRIEVAL_EVAL_K,
) -> dict:
    """Compute mean Precision@K, Recall@K, and MRR across all labeled questions."""
    judgment_list = list(judgments)
    questions = [j.question for j in judgment_list]

    # Batch retrieval avoids re-encoding the same query embedding repeatedly.
    batch_results = retriever.retrieve_batch(questions, top_k=k)

    p_scores: list[float] = []
    r_scores: list[float] = []
    mrr_scores: list[float] = []
    per_query: list[dict] = []

    for judgment in judgment_list:
        results = batch_results[judgment.question]
        retrieved_ids = [r.chunk.chunk_id for r in results]
        p = precision_at_k(retrieved_ids, judgment.relevant_chunk_ids, k)
        r = recall_at_k(retrieved_ids, judgment.relevant_chunk_ids, k)
        mrr = mean_reciprocal_rank(retrieved_ids, judgment.relevant_chunk_ids)
        p_scores.append(p)
        r_scores.append(r)
        mrr_scores.append(mrr)
        per_query.append(
            {
                "question": judgment.question,
                f"precision@{k}": round(p, 4),
                f"recall@{k}": round(r, 4),
                "mrr": round(mrr, 4),
                "retrieved_top1": results[0].chunk.chunk_id if results else "",
                "top1_score": round(results[0].score, 4) if results else 0.0,
            }
        )

    return {
        f"mean_precision@{k}": round(float(np.mean(p_scores)), 4),
        f"mean_recall@{k}": round(float(np.mean(r_scores)), 4),
        "mean_mrr": round(float(np.mean(mrr_scores)), 4),
        "per_query": per_query,
    }


def auto_label_relevance(
    retriever: SemanticRetriever,
    questions: list[str],
    manual_overrides: dict[str, set[str]] | None = None,
    seed_k: int = RELEVANCE_SEED_K,
) -> list[QueryJudgment]:
    """
    Build pseudo relevance labels for evaluation.

    Uses the top-2 retrieved chunks plus optional manual overrides. Manual labels
    on a few questions improve metric credibility when full human labeling is
    not available.
    """
    overrides = manual_overrides or {}
    batch = retriever.retrieve_batch(questions, top_k=seed_k)
    return [
        QueryJudgment(
            question=q,
            relevant_chunk_ids={r.chunk.chunk_id for r in batch[q][:2]} | overrides.get(q, set()),
        )
        for q in questions
    ]

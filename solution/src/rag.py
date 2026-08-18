"""
Task 4 & 5: RAG pipeline and grounding / hallucination analysis.

Pipeline: Question -> Semantic Retrieval -> Evidence -> Pretrained LLM -> Answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import (
    GROUNDING_OVERLAP_THRESHOLD,
    LLM_MAX_INPUT_TOKENS,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
)
from .retrieval import RetrievalResult, SemanticRetriever


@dataclass
class RAGAnswer:
    """Full trace for one RAG answer (retrieval + generation + grounding)."""

    question: str
    retrieved_evidence: list[RetrievalResult]
    evidence_summary: str
    answer: str
    sources: list[str]
    grounded: bool
    grounding_reason: str


class RAGPipeline:
    """
    Retrieval-Augmented Generation wrapper.

    The LLM and retriever are loaded lazily on first use so import/setup stays fast.
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        model_name: str = LLM_MODEL,
        top_k: int = RETRIEVAL_TOP_K,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self._generator = None
        self._tokenizer = None
        self._answer_cache: dict[tuple[str, bool], str] = {}

    def _load_llm(self):
        if self._generator is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._generator.eval()
            self._generator.to("cuda" if torch.cuda.is_available() else "cpu")

    def summarize_evidence(self, evidence_chunks: list[str], max_sentences: int = 2) -> str:
        """
        Extractive summary: pick the longest informative sentences.

        This is intentionally simple and transparent for the assignment write-up.
        """
        joined = " ".join(evidence_chunks)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
        top = sorted(sentences, key=lambda s: len(s.split()), reverse=True)[:max_sentences]
        return " ".join(top)[:500]

    def _generate(self, prompt: str, max_input: int = LLM_MAX_INPUT_TOKENS) -> str:
        """Run the seq2seq model once under inference_mode (no gradients)."""
        import torch

        self._load_llm()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input)
        inputs = {k: v.to(self._generator.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._generator.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate_answer(self, question: str, evidence: str) -> str:
        cache_key = (question, True)
        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        prompt = (
            "Answer the question using only the provided evidence. "
            "If the evidence is insufficient, say you do not know.\n"
            f"Evidence: {evidence}\nQuestion: {question}\nAnswer:"
        )
        answer = self._generate(prompt)
        self._answer_cache[cache_key] = answer
        return answer

    def generate_llm_only_answer(self, question: str) -> str:
        cache_key = (question, False)
        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        answer = self._generate(f"Question: {question}\nAnswer:", max_input=256)
        self._answer_cache[cache_key] = answer
        return answer

    def assess_grounding(self, answer: str, evidence: str) -> tuple[bool, str]:
        """
        Heuristic grounding check.

        - Abstention phrases count as grounded (model acknowledges missing evidence).
        - Otherwise require sufficient lexical overlap with retrieved text.
        """
        for pattern in (
            r"\bi do not know\b",
            r"\bnot enough information\b",
            r"\binsufficient evidence\b",
        ):
            if re.search(pattern, answer.lower()):
                return True, "Answer appropriately abstains or cites insufficient evidence."

        answer_tokens = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
        evidence_tokens = set(re.findall(r"[a-zA-Z]{4,}", evidence.lower()))
        if not answer_tokens:
            return False, "Empty or non-informative generation."

        ratio = len(answer_tokens & evidence_tokens) / len(answer_tokens)
        if ratio >= GROUNDING_OVERLAP_THRESHOLD:
            return True, f"Lexical overlap with evidence is {ratio:.2f} (>= {GROUNDING_OVERLAP_THRESHOLD})."
        return False, f"Low overlap with evidence ({ratio:.2f}); potential unsupported generation."

    def answer_question(self, question: str) -> RAGAnswer:
        """End-to-end RAG for one question."""
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        evidence_texts = [r.chunk.text for r in retrieved]
        evidence_joined = "\n\n".join(evidence_texts)
        summary = self.summarize_evidence(evidence_texts)
        answer = self.generate_answer(question, evidence_joined)
        grounded, reason = self.assess_grounding(answer, evidence_joined)

        sources = [
            f"{r.chunk.doc_title} ({r.chunk.source_url or r.chunk.doc_file}) [score={r.score:.3f}]"
            for r in retrieved
        ]
        return RAGAnswer(
            question=question,
            retrieved_evidence=retrieved,
            evidence_summary=summary,
            answer=answer,
            sources=sources,
            grounded=grounded,
            grounding_reason=reason,
        )

    def answer_questions_batch(self, questions: list[str]) -> list[RAGAnswer]:
        """Answer many questions while reusing cached LLM outputs when possible."""
        return [self.answer_question(q) for q in questions]


def categorize_failure(question: str, llm_only: str, rag: RAGAnswer) -> str:
    """
    Assign a failure category for Task 5 analysis.

    Categories match the assignment: retrieval failure, unsupported generation,
    incomplete synthesis, irrelevant context.
    """
    if not rag.retrieved_evidence:
        return "retrieval failure"
    if rag.retrieved_evidence[0].score < 0.25:
        return "irrelevant context"
    if rag.grounded:
        return "incomplete synthesis" if len(rag.answer.split()) < 8 else "acceptable"
    return "unsupported generation"

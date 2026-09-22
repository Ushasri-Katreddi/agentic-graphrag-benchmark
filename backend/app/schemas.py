from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CorpusDocument(BaseModel):
    """A single document from the hackathon corpus."""

    doc_id: str
    title: str
    url: str
    wikidata_qid: str
    wikipedia_pageid: int
    approx_tokens: int
    text: str


class PublicEvaluationQuestion(BaseModel):
    """A public benchmark question with ground-truth information."""

    qid: str
    question: str
    qtype: str
    answer_named_in_question: Optional[bool] = None
    guess_baseline: Optional[float] = None
    gold_doc_ids: List[str] = Field(default_factory=list)
    answer_verified: Optional[bool] = None
    answer: List[str] = Field(default_factory=list)


class HiddenEvaluationQuestion(BaseModel):
    """A hidden benchmark question.

    Hidden evaluation data must never be used to tune the system.
    """

    qid: str
    question: str
    qtype: str


class Evidence(BaseModel):
    """Evidence used to support an answer."""

    doc_id: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    source_url: Optional[str] = None
    relevance_score: Optional[float] = None


class BenchmarkTrace(BaseModel):
    """Execution trace shared by all benchmark pipelines."""

    steps: List[str] = Field(default_factory=list)
    retrieval_methods: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    strategy_changes: List[str] = Field(default_factory=list)
    stopping_reason: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Common output contract for RAG, GraphRAG and Agentic GraphRAG."""

    qid: str
    pipeline: str

    answer: str = ""

    retrieved_doc_ids: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    latency_ms: float = 0.0

    trace: Optional[BenchmarkTrace] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
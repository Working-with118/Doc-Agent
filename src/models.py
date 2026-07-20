"""
Shared data models for the Document Synthesis & Analysis Agent system.

These schemas are the contract between the Extractor Agent, Synthesis Agent,
and Verifier Agent. Keeping them explicit (rather than passing raw dicts)
is what lets us guarantee data integrity across the pipeline: every claim in
the final report can be traced back to a specific chunk of source text.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class Chunk(BaseModel):
    """A single extracted, addressable unit of source text."""
    chunk_id: str = Field(..., description="Stable unique id, e.g. 'doc1-p3-c2'")
    doc_id: str
    doc_name: str
    page: Optional[int] = None
    section: Optional[str] = None
    text: str
    char_count: int = 0

    def model_post_init(self, __context) -> None:
        self.char_count = len(self.text)


class ExtractionResult(BaseModel):
    """Output of the Extractor Agent for one document."""
    doc_id: str
    doc_name: str
    source_type: SourceType
    chunks: list[Chunk]
    warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)


class Claim(BaseModel):
    """A single synthesized statement, tied back to its supporting chunk(s)."""
    text: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0  # set by Synthesis Agent, adjusted by Verifier


class ReportSection(BaseModel):
    title: str
    claims: list[Claim]


class SynthesisReport(BaseModel):
    """Structured business report produced by the Synthesis Agent."""
    doc_ids: list[str]
    executive_summary: str
    sections: list[ReportSection]
    entities: list[str] = Field(default_factory=list)
    risks_or_action_items: list[str] = Field(default_factory=list)


class VerificationFlag(BaseModel):
    section_title: str
    claim_text: str
    issue: str  # e.g. "No supporting chunk found", "Chunk text does not support claim"
    severity: str  # "low" | "medium" | "high"


class VerificationResult(BaseModel):
    report_ok: bool
    flags: list[VerificationFlag]
    integrity_score: float  # 0-1, fraction of claims that passed verification

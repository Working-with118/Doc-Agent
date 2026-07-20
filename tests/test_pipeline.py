"""
Unit tests for the document synthesis pipeline.
Run with: pytest tests/ (from project root, with src/ on PYTHONPATH)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from extractor_agent import extract_document
from synthesis_agent import synthesize_mock
from verifier_agent import verify_report
from models import Claim, ReportSection, SynthesisReport

SAMPLE_TXT = Path(__file__).parent.parent / "sample_docs" / "sample_service_agreement.txt"


def test_extract_txt_produces_chunks():
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    assert result.chunks, "Extractor should produce at least one chunk"
    assert all(c.doc_id == "test_doc" for c in result.chunks)
    assert all(c.chunk_id for c in result.chunks)


def test_chunk_ids_are_unique():
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique for citation integrity"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_document("does_not_exist.txt")


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        extract_document(Path(__file__))  # a .py file


def test_mock_synthesis_grounds_every_claim_in_a_real_chunk():
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    report = synthesize_mock(result.chunks, doc_ids=["test_doc"])
    valid_ids = {c.chunk_id for c in result.chunks}
    for section in report.sections:
        for claim in section.claims:
            for cid in claim.supporting_chunk_ids:
                assert cid in valid_ids


def test_verifier_passes_grounded_claim():
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    real_chunk = result.chunks[0]
    report = SynthesisReport(
        doc_ids=["test_doc"],
        executive_summary="test",
        sections=[ReportSection(title="T", claims=[
            Claim(text=real_chunk.text[:100], supporting_chunk_ids=[real_chunk.chunk_id])
        ])],
    )
    verification = verify_report(report, result.chunks)
    assert verification.integrity_score == 1.0
    assert verification.report_ok is True
    assert not verification.flags


def test_verifier_flags_fabricated_citation():
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    report = SynthesisReport(
        doc_ids=["test_doc"],
        executive_summary="test",
        sections=[ReportSection(title="T", claims=[
            Claim(text="This document grants the reader a free spaceship.",
                  supporting_chunk_ids=["nonexistent-chunk-id"])
        ])],
    )
    verification = verify_report(report, result.chunks)
    assert verification.report_ok is False
    assert any(f.severity == "high" for f in verification.flags)


def test_verifier_flags_unsupported_claim_with_real_citation():
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    real_chunk = result.chunks[0]
    report = SynthesisReport(
        doc_ids=["test_doc"],
        executive_summary="test",
        sections=[ReportSection(title="T", claims=[
            Claim(text="The contract guarantees a free Ferrari for every renewal.",
                  supporting_chunk_ids=[real_chunk.chunk_id])
        ])],
    )
    verification = verify_report(report, result.chunks)
    assert verification.integrity_score < 1.0
    assert any(f.severity == "medium" for f in verification.flags)


def test_verifier_catches_swapped_figure_despite_topical_similarity():
    """
    Regression test for a real gap found during development: a fabricated claim
    that shares plenty of surrounding vocabulary with the source (so it would
    pass a pure similarity check) but states the WRONG number should still be
    caught, since numeric facts are usually the most consequential thing to
    get right in a business/legal document.
    """
    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    fee_chunk = next(c for c in result.chunks if "monthly fee" in c.text.lower())

    true_claim = Claim(
        text="The client must pay a monthly fee of $18,500, due within 15 days of invoice.",
        supporting_chunk_ids=[fee_chunk.chunk_id],
    )
    fabricated_claim = Claim(
        text="The client must pay $50,000 annually with no late fees ever applied.",
        supporting_chunk_ids=[fee_chunk.chunk_id],
    )

    report = SynthesisReport(
        doc_ids=["test_doc"], executive_summary="t",
        sections=[ReportSection(title="Fees", claims=[true_claim, fabricated_claim])],
    )
    verification = verify_report(report, result.chunks)

    high_severity_texts = [f.claim_text for f in verification.flags if f.severity == "high"]
    assert any("50,000" in t for t in high_severity_texts), \
        "Fabricated figure should be flagged high-severity even with topical overlap"
    assert not any("18,500" in t for t in high_severity_texts), \
        "True figure should not be flagged"


def test_feedback_loop_resolves_fabricated_citation():
    """
    End-to-end proof that the Synthesis Agent <-> Verifier Agent loop actually
    does something: a fabricated claim with a fake citation gets removed by the
    revision pass, and the report passes verification afterward.
    """
    from synthesis_agent import refine

    result = extract_document(SAMPLE_TXT, doc_id="test_doc")
    real_chunk = result.chunks[0]
    bad_report = SynthesisReport(
        doc_ids=["test_doc"],
        executive_summary="test",
        sections=[ReportSection(title="T", claims=[
            Claim(text=real_chunk.text[:80], supporting_chunk_ids=[real_chunk.chunk_id]),
            Claim(text="This document grants the reader a free spaceship.",
                  supporting_chunk_ids=["nonexistent-chunk-id"]),
        ])],
    )
    first_verification = verify_report(bad_report, result.chunks)
    assert first_verification.report_ok is False

    revised_report = refine(bad_report, first_verification.flags, result.chunks, mode="mock")
    second_verification = verify_report(revised_report, result.chunks)

    assert second_verification.report_ok is True
    assert second_verification.integrity_score == 1.0
    # the true claim should survive the revision; the fabricated one should not
    all_claim_texts = [c.text for s in revised_report.sections for c in s.claims]
    assert not any("spaceship" in t for t in all_claim_texts)
    assert any(real_chunk.text[:80] in t or t in real_chunk.text for t in all_claim_texts)

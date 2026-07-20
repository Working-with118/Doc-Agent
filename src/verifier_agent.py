"""
Verifier Agent
==============
Third agent in the pipeline. Its only job is to check that the Synthesis
Agent didn't hallucinate: every claim must cite chunk_ids that exist, and
the cited chunk text must actually support the claim.

Two checks are combined:
1. Lexical token overlap — instant, zero dependencies, catches obvious
   fabrications (a claim sharing almost no words with its cited source).
2. TF-IDF cosine similarity — a proper (if classical, not neural) semantic
   check: it weights rare/distinctive words more than common ones, so it
   catches cases where a claim uses different wording but the SAME rare
   entities/numbers as the source (a true paraphrase) while still flagging
   claims that introduce facts/entities absent from the source. This runs
   fully offline (scikit-learn's TfidfVectorizer, no model download) —
   unlike a neural embedding model, it needs no network access and no GPU,
   which matters for a verifier that has to run on every report with zero
   added latency or cost.

A claim passes if EITHER check clears its threshold — the two catch
different failure modes (lexical overlap is stricter on short claims,
TF-IDF is more robust to paraphrasing), so requiring only one to pass
avoids double-penalizing legitimate paraphrases while still catching
fabrications that fail both.
"""
from __future__ import annotations

import re
from models import Chunk, SynthesisReport, VerificationFlag, VerificationResult

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "this", "that", "with", "as", "by", "at", "it",
    "its", "from", "will", "shall", "which", "has", "have", "had", "not", "no",
}

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _overlap_ratio(claim_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


def _tfidf_similarity(claim_text: str, chunk_texts: list[str]) -> float:
    """Cosine similarity between the claim and its best-matching cited chunk,
    computed over a TF-IDF space fit on [claim] + all cited chunks. Returns 0.0
    if scikit-learn isn't installed (graceful degradation to lexical-only)."""
    if not _SKLEARN_AVAILABLE or not chunk_texts:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([claim_text] + chunk_texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
        return float(sims.max()) if len(sims) else 0.0
    except ValueError:
        # e.g. empty vocabulary after stopword removal on very short text
        return 0.0


def _extract_numbers(text: str) -> set[str]:
    """
    Extract normalized numeric tokens (dollar amounts, percentages, plain numbers)
    from text, e.g. "$18,500" -> "18500", "1.5%" -> "1.5%", "30 days" -> "30".
    This targets the specific failure mode where a fabricated claim shares plenty
    of topical vocabulary with the source (so lexical/TF-IDF checks pass) but
    swaps the actual figure — e.g. "$50,000 annually" instead of "$18,500 monthly".
    Numbers are the most concretely checkable facts in business/legal documents,
    so this check runs independently of the similarity-based checks above.
    """
    numbers = set()
    for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?%?", text):
        token = m.group().replace(",", "").replace("$", "")
        if token:
            numbers.add(token)
    return numbers


def _check_numeric_consistency(claim_text: str, chunk_texts: list[str]) -> list[str]:
    """Returns any numeric tokens in the claim that don't appear in any cited chunk."""
    claim_numbers = _extract_numbers(claim_text)
    if not claim_numbers:
        return []
    chunk_numbers: set[str] = set()
    for t in chunk_texts:
        chunk_numbers |= _extract_numbers(t)
    return sorted(claim_numbers - chunk_numbers)


def verify_report(report: SynthesisReport, chunks: list[Chunk],
                   overlap_threshold: float = 0.35,
                   tfidf_threshold: float = 0.25) -> VerificationResult:
    chunk_index = {c.chunk_id: c for c in chunks}
    flags: list[VerificationFlag] = []
    total_claims = 0
    passed_claims = 0

    for section in report.sections:
        for claim in section.claims:
            total_claims += 1

            if not claim.supporting_chunk_ids:
                flags.append(VerificationFlag(
                    section_title=section.title,
                    claim_text=claim.text,
                    issue="No supporting chunk_id cited.",
                    severity="high",
                ))
                continue

            missing = [cid for cid in claim.supporting_chunk_ids if cid not in chunk_index]
            if missing:
                flags.append(VerificationFlag(
                    section_title=section.title,
                    claim_text=claim.text,
                    issue=f"Cited chunk_id(s) not found in source: {missing}",
                    severity="high",
                ))
                continue

            cited_texts = [chunk_index[cid].text for cid in claim.supporting_chunk_ids]

            # Check 1 & 2: similarity-based (catches wholesale fabrication / off-topic claims)
            claim_tokens = _tokenize(claim.text)
            lexical_score = max(_overlap_ratio(claim_tokens, _tokenize(t)) for t in cited_texts)
            tfidf_score = _tfidf_similarity(claim.text, cited_texts)
            passes_similarity = lexical_score >= overlap_threshold or tfidf_score >= tfidf_threshold

            # Check 3: numeric consistency (catches swapped figures that similarity checks miss,
            # since a claim can share plenty of topical vocabulary while stating the wrong number)
            unsupported_numbers = _check_numeric_consistency(claim.text, cited_texts)

            if unsupported_numbers:
                flags.append(VerificationFlag(
                    section_title=section.title,
                    claim_text=claim.text,
                    issue=(f"Claim states figure(s) {unsupported_numbers} not found anywhere in the "
                           f"cited source chunk(s) — likely a fabricated or incorrect number, even "
                           f"though the surrounding text may be topically similar."),
                    severity="high",
                ))
            elif not passes_similarity:
                flags.append(VerificationFlag(
                    section_title=section.title,
                    claim_text=claim.text,
                    issue=(f"Low similarity to cited source on both checks "
                           f"(lexical overlap {lexical_score:.0%} < {overlap_threshold:.0%}, "
                           f"TF-IDF cosine {tfidf_score:.0%} < {tfidf_threshold:.0%}); "
                           f"claim may be fabricated or paraphrased beyond what the source supports."),
                    severity="medium",
                ))
            else:
                passed_claims += 1

    integrity_score = (passed_claims / total_claims) if total_claims else 1.0
    report_ok = integrity_score >= 0.8 and not any(f.severity == "high" for f in flags)

    return VerificationResult(report_ok=report_ok, flags=flags, integrity_score=integrity_score)

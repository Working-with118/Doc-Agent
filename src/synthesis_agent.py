"""
Synthesis Agent
===============
Takes the Chunk objects produced by the Extractor Agent and compiles them
into a structured business report (SynthesisReport): executive summary,
themed sections of claims, entities, and risks/action items.

Every claim the LLM produces must cite the chunk_id(s) it drew from. This
is enforced via the prompt + a strict JSON schema, and is what lets the
Verifier Agent later confirm the claim is actually supported by source text
rather than hallucinated.

Two modes:
- "live": calls the Anthropic API (requires ANTHROPIC_API_KEY env var).
- "mock": deterministic, offline synthesis for demos/tests without an API key.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Iterable

from models import Chunk, Claim, ReportSection, SynthesisReport

MAX_LLM_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def _parse_llm_json(raw_text: str) -> dict:
    """
    Best-effort JSON parsing of an LLM response. Real API calls occasionally
    wrap JSON in prose or code fences despite instructions; this makes a
    reasonable attempt to recover before giving up, rather than crashing the
    whole pipeline on a single malformed response.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first '{' and last '}' and try that slice —
    # handles cases where the model added a sentence of preamble/postamble.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse LLM response as JSON. Raw response (truncated): {raw_text[:300]}")


def _call_with_retries(fn, *args, **kwargs):
    """Retry transient API failures (rate limits, timeouts, connection errors)
    with exponential backoff. Does NOT retry on programmer errors (bad API key,
    missing package) — those raise immediately since retrying won't help."""
    import anthropic

    last_exc = None
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_exc = e
            if attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"[Synthesis Agent] Transient API error ({type(e).__name__}), "
                      f"retrying in {wait:.0f}s (attempt {attempt}/{MAX_LLM_RETRIES})...")
                time.sleep(wait)
    raise RuntimeError(f"Anthropic API call failed after {MAX_LLM_RETRIES} attempts: {last_exc}") from last_exc

SYSTEM_PROMPT = """You are the Synthesis Agent in a multi-agent document analysis system.
You receive numbered source chunks (each with a chunk_id) extracted from one or more
business/legal/technical documents. Your job is to synthesize them into a structured
business report.

CRITICAL RULE: Every claim you write must be traceable to specific chunk_ids that
actually support it. Never state something that isn't grounded in the provided chunks.
If the chunks don't contain enough information for a section, say so explicitly rather
than inventing content.

Respond ONLY with valid JSON matching this schema, and nothing else:
{
  "executive_summary": "string, 2-4 sentences",
  "sections": [
    {"title": "string", "claims": [{"text": "string", "supporting_chunk_ids": ["id1","id2"]}]}
  ],
  "entities": ["string", ...],
  "risks_or_action_items": ["string", ...]
}
"""


def _format_chunks_for_prompt(chunks: Iterable[Chunk]) -> str:
    lines = []
    for c in chunks:
        loc = f"page {c.page}" if c.page else (c.section or "unlabeled")
        lines.append(f"[chunk_id={c.chunk_id} | {c.doc_name} | {loc}]\n{c.text}\n")
    return "\n---\n".join(lines)


_SIGNAL_PATTERN = re.compile(
    r"\$[\d,]+|\d+%|\d{1,2}\s*(?:day|month|year)s?\b|\bshall\b|\bmust\b|\bwithin\b|\beffective\b",
    re.IGNORECASE,
)


def _score_sentence(sentence: str) -> float:
    """
    Lightweight extractive scoring: a sentence is more 'report-worthy' if it
    contains concrete obligations/figures (money, percentages, durations,
    modal-of-obligation words like 'shall'/'must') rather than boilerplate.
    This is a real (if simple, dependency-free) heuristic — not just picking
    the first N characters — so offline mock-mode output is genuinely more
    useful, not merely a placeholder for the live LLM path.
    """
    signals = len(_SIGNAL_PATTERN.findall(sentence))
    length_penalty = 1.0 if 40 <= len(sentence) <= 280 else 0.5
    return signals * length_penalty


def _best_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text.strip()[:220]
    best = max(sentences, key=_score_sentence)
    return best if len(best) <= 280 else best[:277] + "..."


def synthesize_mock(chunks: list[Chunk], doc_ids: list[str]) -> SynthesisReport:
    """
    Offline, deterministic synthesis for demos and CI — no API key required.
    Uses lightweight extractive summarization (scores sentences by presence of
    concrete obligations: money, percentages, durations, "shall"/"must") rather
    than naive truncation, so mock mode produces a genuinely useful report, not
    just a placeholder. Still clearly labeled as mock in the executive summary
    so it's never mistaken for the LLM-generated version.
    """
    if not chunks:
        return SynthesisReport(
            doc_ids=doc_ids,
            executive_summary="No extractable text was found in the provided document(s).",
            sections=[],
        )

    # Group chunks by section/page bucket for a lightweight structure
    buckets: dict[str, list[Chunk]] = {}
    for c in chunks:
        key = c.section or (f"Page {c.page}" if c.page else "General")
        buckets.setdefault(key, []).append(c)

    sections: list[ReportSection] = []
    for key, group in buckets.items():
        claims = []
        for c in group[:4]:
            best_sentence = _best_sentence(c.text)
            claims.append(Claim(text=best_sentence, supporting_chunk_ids=[c.chunk_id], confidence=0.6))
        sections.append(ReportSection(title=key, claims=claims))

    total_chars = sum(c.char_count for c in chunks)
    summary = (
        f"[MOCK MODE — extractive summarization, no LLM used] Synthesized {len(chunks)} chunks "
        f"(~{total_chars:,} characters) from {len(doc_ids)} document(s) into {len(sections)} section(s), "
        f"surfacing the most obligation-dense sentence per chunk. "
        "Run with --mode live and ANTHROPIC_API_KEY set for genuine LLM synthesis with true summarization."
    )
    return SynthesisReport(doc_ids=doc_ids, executive_summary=summary, sections=sections)


def synthesize_live(chunks: list[Chunk], doc_ids: list[str], model: str = "claude-sonnet-4-6") -> SynthesisReport:
    """Real synthesis via the Anthropic API. Requires ANTHROPIC_API_KEY.
    Retries transient failures (rate limits, timeouts) with backoff, and
    tolerates minor JSON formatting issues in the model's response."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install the anthropic package: pip install anthropic") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    prompt_body = _format_chunks_for_prompt(chunks)

    def _call():
        return client.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Source chunks:\n\n{prompt_body}"}],
        )

    response = _call_with_retries(_call)
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    data = _parse_llm_json(raw_text)

    sections = [
        ReportSection(
            title=s["title"],
            claims=[Claim(**c) for c in s["claims"]],
        )
        for s in data.get("sections", [])
    ]
    return SynthesisReport(
        doc_ids=doc_ids,
        executive_summary=data.get("executive_summary", ""),
        sections=sections,
        entities=data.get("entities", []),
        risks_or_action_items=data.get("risks_or_action_items", []),
    )


def synthesize(chunks: list[Chunk], doc_ids: list[str], mode: str = "mock",
               model: str = "claude-sonnet-4-6") -> SynthesisReport:
    if mode == "live":
        return synthesize_live(chunks, doc_ids, model=model)
    return synthesize_mock(chunks, doc_ids)


REFINE_SYSTEM_PROMPT = """You are the Synthesis Agent, now in a revision pass.
A separate Verifier Agent has independently fact-checked your previous report against
the source chunks and flagged specific claims as unsupported, over-extrapolated, or
citing a chunk that doesn't exist. Your job now is to produce a corrected report that
either (a) rewrites the flagged claim so it is strictly supported by its cited chunk,
(b) re-cites it against a chunk that actually supports it, or (c) removes it entirely
if no chunk supports it. Do not reintroduce the same issue. Keep all unflagged claims
unchanged. Respond with the same JSON schema as before, and nothing else.
"""


def refine_mock(report: SynthesisReport, flags: list, chunks: list[Chunk]) -> SynthesisReport:
    """
    Offline equivalent of the revision pass: deterministically drops any claim
    that the Verifier flagged, and records what was removed so the loop's effect
    is visible even without an LLM call.
    """
    flagged_texts = {f.claim_text for f in flags}
    new_sections: list[ReportSection] = []
    removed = 0
    for section in report.sections:
        kept_claims = [c for c in section.claims if c.text not in flagged_texts]
        removed += len(section.claims) - len(kept_claims)
        if kept_claims:
            new_sections.append(ReportSection(title=section.title, claims=kept_claims))

    summary = report.executive_summary
    if removed:
        summary += f" [Revision pass: removed {removed} claim(s) flagged by the Verifier Agent as unsupported.]"

    return SynthesisReport(
        doc_ids=report.doc_ids,
        executive_summary=summary,
        sections=new_sections,
        entities=report.entities,
        risks_or_action_items=report.risks_or_action_items,
    )


def refine_live(report: SynthesisReport, flags: list, chunks: list[Chunk],
                 model: str = "claude-sonnet-4-6") -> SynthesisReport:
    """Real revision pass: sends the flagged claims + full chunk set back to Claude.
    Same retry/JSON-repair robustness as synthesize_live."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    flags_text = "\n".join(f"- ({f.section_title}) \"{f.claim_text}\" — {f.issue}" for f in flags)
    prompt_body = _format_chunks_for_prompt(chunks)
    previous_json = json.dumps({
        "executive_summary": report.executive_summary,
        "sections": [s.model_dump() for s in report.sections],
        "entities": report.entities,
        "risks_or_action_items": report.risks_or_action_items,
    }, indent=2)

    user_msg = (
        f"Previous report:\n{previous_json}\n\n"
        f"Verifier Agent flags to address:\n{flags_text}\n\n"
        f"Source chunks (for re-checking citations):\n\n{prompt_body}"
    )

    def _call():
        return client.messages.create(
            model=model,
            max_tokens=4000,
            system=REFINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

    response = _call_with_retries(_call)
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    data = _parse_llm_json(raw_text)

    sections = [ReportSection(title=s["title"], claims=[Claim(**c) for c in s["claims"]])
                for s in data.get("sections", [])]
    return SynthesisReport(
        doc_ids=report.doc_ids,
        executive_summary=data.get("executive_summary", report.executive_summary),
        sections=sections,
        entities=data.get("entities", report.entities),
        risks_or_action_items=data.get("risks_or_action_items", report.risks_or_action_items),
    )


def refine(report: SynthesisReport, flags: list, chunks: list[Chunk], mode: str = "mock",
           model: str = "claude-sonnet-4-6") -> SynthesisReport:
    if mode == "live":
        return refine_live(report, flags, chunks, model=model)
    return refine_mock(report, flags, chunks)

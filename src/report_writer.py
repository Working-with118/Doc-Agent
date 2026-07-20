"""Renders a SynthesisReport + VerificationResult into a readable Markdown report."""
from __future__ import annotations

from models import SynthesisReport, VerificationResult


def render_markdown(report: SynthesisReport, verification: VerificationResult, timing: dict) -> str:
    lines = ["# Document Synthesis Report", ""]
    lines.append(f"**Documents analyzed:** {', '.join(report.doc_ids)}")
    lines.append(f"**Integrity score:** {verification.integrity_score:.0%} "
                 f"({'PASSED' if verification.report_ok else 'REVIEW NEEDED'})")
    if timing.get("revision_rounds"):
        lines.append(f"**Synthesis↔Verifier revision rounds:** {timing['revision_rounds']} "
                     f"(Synthesis Agent revised its report after Verifier Agent flags)")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(report.executive_summary)
    lines.append("")

    for section in report.sections:
        lines.append(f"## {section.title}")
        for claim in section.claims:
            cite = ", ".join(claim.supporting_chunk_ids) if claim.supporting_chunk_ids else "no citation"
            lines.append(f"- {claim.text}  \n  *[source: {cite}]*")
        lines.append("")

    if report.entities:
        lines.append("## Entities")
        lines.extend(f"- {e}" for e in report.entities)
        lines.append("")

    if report.risks_or_action_items:
        lines.append("## Risks / Action Items")
        lines.extend(f"- {r}" for r in report.risks_or_action_items)
        lines.append("")

    if verification.flags:
        lines.append("## ⚠ Verifier Agent Flags")
        lines.append("The following claims could not be fully confirmed against source text:")
        lines.append("")
        for f in verification.flags:
            lines.append(f"- **[{f.severity.upper()}]** ({f.section_title}) \"{f.claim_text[:100]}...\" — {f.issue}")
        lines.append("")

    lines.append("## Pipeline Performance")
    for k, v in timing.items():
        lines.append(f"- {k.replace('_', ' ')}: {v}")

    return "\n".join(lines)

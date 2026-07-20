"""
Visual demo UI for the Document Synthesis & Analysis Agent system.

Run with:
    streamlit run app.py

Lets you upload PDF/DOCX/TXT files, watch the three agents run in sequence
(Extractor -> Synthesis -> Verifier -> optional revision loop), and see the
final report with flagged claims highlighted. Built for a live hackathon
demo where "watch it work" matters more than a wall of text.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from extractor_agent import extract_document  # noqa: E402
from synthesis_agent import synthesize, refine  # noqa: E402
from verifier_agent import verify_report  # noqa: E402

st.set_page_config(page_title="Document Synthesis Agent", layout="wide")

st.image("logo.png", use_container_width=False, width=500)
# st.title("Intelligent Document Synthesis & Analysis Agent")
st.caption("Extractor Agent → Synthesis Agent → Verifier Agent, with a live revision loop when issues are found.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Synthesis mode", ["mock (offline, free)", "live (Claude API)"], index=0)
    mode_key = "mock" if mode.startswith("mock") else "live"
    if mode_key == "live":
        api_key = st.text_input("ANTHROPIC_API_KEY", type="password")
        if api_key:
            import os
            os.environ["ANTHROPIC_API_KEY"] = api_key
    max_rounds = st.slider("Max revision rounds", 0, 3, 2)
    st.divider()
    st.markdown(
        "**How the Verifier works:** every claim from the Synthesis Agent must cite a "
        "`chunk_id`. The Verifier independently checks that chunk exists and that its "
        "text actually supports the claim. Unsupported claims get sent back to the "
        "Synthesis Agent for revision."
    )

    st.markdown(
        "**Created by ~** Syed Abdur Rasheed & Md. Abdul Razzaq "
    )

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, or TXT) — try mixing formats",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

use_sample = st.checkbox("...or just use the bundled sample contract", value=not uploaded_files)

run = st.button("▶ Run pipeline", type="primary", disabled=not (uploaded_files or use_sample))

if run:
    tmpdir = Path(tempfile.mkdtemp())
    paths: list[Path] = []

    if uploaded_files:
        for f in uploaded_files:
            p = tmpdir / f.name
            p.write_bytes(f.read())
            paths.append(p)
    if use_sample:
        sample = Path(__file__).parent / "sample_docs" / "sample_service_agreement.txt"
        paths.append(sample)

    # --- Step 1: Extraction ---
    step1 = st.status("🔍 Extractor Agent: pulling text from documents...", expanded=True)
    t0 = time.perf_counter()
    extractions = []
    for p in paths:
        try:
            ext = extract_document(p)
            extractions.append(ext)
            step1.write(f"✅ **{p.name}** — {len(ext.chunks)} chunks extracted "
                        f"({ext.source_type.value.upper()})")
            for w in ext.warnings:
                step1.write(f"⚠️ {w}")
        except Exception as e:
            step1.write(f"❌ **{p.name}** failed: {e}")
    extraction_time = time.perf_counter() - t0
    all_chunks = [c for ext in extractions for c in ext.chunks]
    doc_ids = [ext.doc_id for ext in extractions]
    step1.update(label=f"🔍 Extractor Agent: done — {len(all_chunks)} chunks from "
                        f"{len(extractions)} document(s) in {extraction_time:.2f}s", state="complete")

    # --- Step 2: Synthesis ---
    step2 = st.status("🧠 Synthesis Agent: compiling structured report...", expanded=True)
    t1 = time.perf_counter()
    try:
        report = synthesize(all_chunks, doc_ids, mode=mode_key)
        synth_time = time.perf_counter() - t1
        step2.write(f"Produced {len(report.sections)} section(s), "
                    f"{sum(len(s.claims) for s in report.sections)} claim(s).")
        step2.update(label=f"🧠 Synthesis Agent: done in {synth_time:.2f}s", state="complete")
    except Exception as e:
        step2.update(label="🧠 Synthesis Agent: failed", state="error")
        st.error(str(e))
        st.stop()

    # --- Step 3: Verification (+ revision loop) ---
    step3 = st.status("🕵️ Verifier Agent: checking claims against sources...", expanded=True)
    verification = verify_report(report, all_chunks)
    round_num = 0
    while round_num < max_rounds and any(f.severity == "high" for f in verification.flags):
        round_num += 1
        step3.write(f"⚠️ Round {round_num}: {len(verification.flags)} flag(s) found — "
                    f"sending back to Synthesis Agent for revision...")
        report = refine(report, verification.flags, all_chunks, mode=mode_key)
        verification = verify_report(report, all_chunks)
        step3.write(f"↩️ Synthesis Agent revised the report. Re-checking...")

    status_state = "complete" if verification.report_ok else "error"
    step3.update(
        label=f"🕵️ Verifier Agent: integrity score {verification.integrity_score:.0%} "
              f"({'PASSED' if verification.report_ok else 'NEEDS REVIEW'}) "
              f"after {round_num} revision round(s)",
        state=status_state,
    )

    st.divider()

    # --- Final report ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents processed", len(extractions))
    col2.metric("Chunks extracted", len(all_chunks))
    col3.metric("Integrity score", f"{verification.integrity_score:.0%}")

    st.subheader("Executive Summary")
    st.write(report.executive_summary)

    for section in report.sections:
        with st.expander(f"📌 {section.title} ({len(section.claims)} claim(s))", expanded=True):
            for claim in section.claims:
                cite = ", ".join(claim.supporting_chunk_ids) if claim.supporting_chunk_ids else "none"
                st.markdown(f"- {claim.text}")
                st.caption(f"source: `{cite}`")

    if report.entities:
        st.subheader("Entities")
        st.write(", ".join(report.entities))

    if report.risks_or_action_items:
        st.subheader("Risks / Action Items")
        for r in report.risks_or_action_items:
            st.markdown(f"- {r}")

    if verification.flags:
        st.subheader("⚠️ Remaining Verifier Flags")
        for f in verification.flags:
            st.warning(f"**[{f.severity.upper()}]** ({f.section_title}) \"{f.claim_text[:120]}\" — {f.issue}")
    else:
        st.success("No unresolved integrity flags — every claim is grounded in a cited source chunk.")

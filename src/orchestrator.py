"""
Orchestrator
============
Coordinates the three agents (Extractor -> Synthesis -> Verifier) into a
single pipeline, and handles concurrent processing of multiple documents
to satisfy the "process large volumes of multi-format documents" scalability
requirement.

Concurrency model: extraction (CPU/IO-bound, per-file) runs in a thread pool
across all input documents simultaneously. Synthesis then runs once over the
combined chunk set (or per-document, depending on --per-doc flag) so the
report can reason across documents together when useful.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from extractor_agent import extract_document
from synthesis_agent import synthesize, refine
from verifier_agent import verify_report
from models import Chunk, ExtractionResult, SynthesisReport, VerificationResult


@dataclass
class PipelineResult:
    extractions: list[ExtractionResult]
    report: SynthesisReport
    verification: VerificationResult
    timing: dict[str, float]
    revision_rounds: int = 0


def _extract_all(paths: list[Path], max_workers: int = 8) -> tuple[list[ExtractionResult], list[str]]:
    results: list[ExtractionResult] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {pool.submit(extract_document, p): p for p in paths}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                results.append(future.result())
            except Exception as e:  # noqa: BLE001 - surface all extraction errors to the caller
                errors.append(f"{p.name}: {e}")
    return results, errors


def run_pipeline(input_paths: list[str], mode: str = "mock", model: str = "claude-sonnet-4-6",
                  max_workers: int = 8, max_revision_rounds: int = 2) -> PipelineResult:
    paths = [Path(p) for p in input_paths]
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    extractions, errors = _extract_all(paths, max_workers=max_workers)
    timing["extraction_seconds"] = round(time.perf_counter() - t0, 3)

    if errors:
        for e in errors:
            print(f"[Extractor Agent] WARNING: failed on {e}")

    all_chunks: list[Chunk] = []
    doc_ids: list[str] = []
    for ext in extractions:
        all_chunks.extend(ext.chunks)
        doc_ids.append(ext.doc_id)
        for w in ext.warnings:
            print(f"[Extractor Agent] {ext.doc_name}: {w}")

    t1 = time.perf_counter()
    report = synthesize(all_chunks, doc_ids, mode=mode, model=model)
    timing["synthesis_seconds"] = round(time.perf_counter() - t1, 3)

    t2 = time.perf_counter()
    verification = verify_report(report, all_chunks)
    timing["verification_seconds"] = round(time.perf_counter() - t2, 3)

    # Feedback loop: if the Verifier Agent found high-severity issues (fabricated or
    # unsupported claims), send them back to the Synthesis Agent for a revision pass,
    # then re-verify. Bounded so a stubborn disagreement can't loop forever.
    revision_rounds = 0
    refine_seconds = 0.0
    while (revision_rounds < max_revision_rounds
           and any(f.severity == "high" for f in verification.flags)):
        revision_rounds += 1
        print(f"[Verifier Agent] {len(verification.flags)} flag(s) found — "
              f"requesting revision pass {revision_rounds}/{max_revision_rounds} from Synthesis Agent.")
        t3 = time.perf_counter()
        report = refine(report, verification.flags, all_chunks, mode=mode, model=model)
        refine_seconds += time.perf_counter() - t3
        verification = verify_report(report, all_chunks)

    if refine_seconds:
        timing["refine_seconds"] = round(refine_seconds, 3)

    timing["total_seconds"] = round(sum(v for k, v in timing.items() if k != "documents_processed"
                                         and k != "chunks_extracted"), 3)
    timing["documents_processed"] = len(extractions)
    timing["chunks_extracted"] = len(all_chunks)

    return PipelineResult(extractions=extractions, report=report, verification=verification,
                           timing=timing, revision_rounds=revision_rounds)

#!/usr/bin/env python3
"""
Intelligent Document Synthesis & Analysis Agent — CLI

Usage:
    python cli.py --input sample_docs/*.txt --output outputs/report.md
    python cli.py --input contract.pdf --mode live --output outputs/report.md

Modes:
    mock  (default) — deterministic offline synthesis, no API key needed.
    live             — real synthesis via the Anthropic API (needs ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import run_pipeline  # noqa: E402
from report_writer import render_markdown  # noqa: E402


def expand_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    # de-dupe, preserve order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Synthesis & Analysis Agent")
    parser.add_argument("--input", nargs="+", required=True, help="File path(s) or glob pattern(s)")
    parser.add_argument("--output", default="outputs/report.md", help="Output markdown path")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Also write raw JSON report alongside markdown")
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    if not input_paths:
        print("No input files matched.", file=sys.stderr)
        sys.exit(1)

    print(f"[Orchestrator] Processing {len(input_paths)} document(s) in '{args.mode}' mode...")
    result = run_pipeline(input_paths, mode=args.mode, model=args.model, max_workers=args.max_workers)

    md = render_markdown(result.report, result.verification,
                         {**result.timing, "revision_rounds": result.revision_rounds})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[Orchestrator] Report written to {out_path}")

    if args.json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps({
            "report": result.report.model_dump(),
            "verification": result.verification.model_dump(),
            "timing": result.timing,
        }, indent=2), encoding="utf-8")
        print(f"[Orchestrator] JSON written to {json_path}")

    print(f"[Verifier Agent] Integrity score: {result.verification.integrity_score:.0%} "
          f"({'PASSED' if result.verification.report_ok else 'NEEDS REVIEW'})")
    if result.revision_rounds:
        print(f"[Orchestrator] Synthesis Agent revised the report {result.revision_rounds} time(s) "
              f"in response to Verifier Agent flags.")
    if result.verification.flags:
        print(f"[Verifier Agent] {len(result.verification.flags)} claim(s) still flagged after revision — see report.")


if __name__ == "__main__":
    main()

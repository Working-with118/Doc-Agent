"""
REST API for the Document Synthesis & Analysis Agent system.

Run with:
    uvicorn api:app --reload --port 8000

Then either open http://localhost:8000/docs for interactive Swagger UI,
or:
    curl -X POST http://localhost:8000/analyze \
        -F "files=@sample_docs/sample_service_agreement.txt" \
        -F "mode=mock"

This is the third interface alongside the CLI (cli.py) and the visual demo
(app.py) — included because a real deployment of this system would need to
be callable by other services, not just a human at a terminal or browser.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from orchestrator import run_pipeline  # noqa: E402

app = FastAPI(
    title="Intelligent Document Synthesis & Analysis Agent",
    description="Multi-agent pipeline: Extractor -> Synthesis -> Verifier, with a revision loop.",
    version="1.0.0",
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_MB = 25


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    files: list[UploadFile] = File(..., description="One or more PDF/DOCX/TXT files"),
    mode: Literal["mock", "live"] = Form("mock"),
    max_revision_rounds: int = Form(2),
):
    """
    Run the full Extractor -> Synthesis -> Verifier (-> revision loop) pipeline
    on the uploaded documents and return the structured report + verification
    result as JSON.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    tmpdir = Path(tempfile.mkdtemp())
    saved_paths: list[str] = []

    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for {f.filename}. "
                       f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )
        content = await f.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename} exceeds the {MAX_FILE_SIZE_MB}MB limit.",
            )
        dest = tmpdir / f.filename
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    try:
        result = run_pipeline(saved_paths, mode=mode, max_revision_rounds=max_revision_rounds)
    except Exception as e:  # noqa: BLE001 - surface pipeline errors as a clean 500, not a stack trace to the client
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    return JSONResponse({
        "report": result.report.model_dump(),
        "verification": result.verification.model_dump(),
        "timing": result.timing,
        "revision_rounds": result.revision_rounds,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import uuid
import shutil
import asyncio
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    CleanupResponse
)
from indexer.indexer import (
    index_pdf,
    cleanup_document
)

load_dotenv() 

UPLOADS_DIR = Path(__file__).parent / "uploads"
WORKSPACE_DIR = Path(__file__).parent / "workspace"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting — cleaning stale uploads...")
    shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "documents").mkdir(parents=True, exist_ok=True)
    print("Ready.")
    yield



app = FastAPI(
    title="Vectorless Agentic RAG",
    description="Reasoning-based RAG using PageIndex + LangGraph + Mistral",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        os.getenv("FRONTEND_URL"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Basic health check — used by Render to verify server is up."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, run Mistral OCR, build the section tree.
    Returns doc_id which the frontend uses for all subsequent requests.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    doc_id = str(uuid.uuid4())
    save_path = UPLOADS_DIR / f"{doc_id}.pdf"

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    try:
        doc_id = await asyncio.to_thread(index_pdf, str(save_path))
    except Exception as e:
        traceback.print_exc()
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index PDF: {str(e)}"
        )

    return UploadResponse(doc_id=doc_id)


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Run the LangGraph agentic RAG pipeline on a previously uploaded document.
    Returns the LLM answer and optional verification feedback.
    """
    from graph.builder import build_graph

    doc_workspace = WORKSPACE_DIR / "documents" / request.doc_id
    if not doc_workspace.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found. Please upload again."
        )

    try:
        graph = build_graph()

        initial_state = {
            "user_query": request.user_query,
            "doc_id": request.doc_id,
            "verification": request.verification,
            "query_intent": None,
            "document_tree": None,
            "page_range": None,
            "content": None,
            "pages_already_tried": [],
            "answer": None,
            "answer_valid": None,
            "verification_feedback": None
        }

        final_state = await graph.ainvoke(initial_state)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    return QueryResponse(
        llm_answer=final_state["answer"],
        llm_feedback=final_state.get("verification_feedback")
    )


@app.delete("/cleanup/{doc_id}", response_model=CleanupResponse)
async def cleanup(doc_id: str):
    """
    Delete a document's PDF and workspace tree from disk.
    Called when user uploads a new PDF or explicitly clears session.
    """
    success = cleanup_document(doc_id)

    if success:
        return CleanupResponse(
            success=True,
            message=f"Document {doc_id} cleaned successfully."
        )
    else:
        return CleanupResponse(
            success=False,
            message=f"Failed to clean document {doc_id}. It may not exist."
        )
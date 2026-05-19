# PageMind — Vectorless Agentic RAG

> **Ask your documents anything.** Precise, grounded answers from PDFs — powered by Mistral OCR, a custom tree index, and a LangGraph agentic pipeline. No vector database required.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [The Agentic Pipeline (LangGraph)](#the-agentic-pipeline-langgraph)
- [The PageIndex — Tree-Based Retrieval](#the-pageindex--tree-based-retrieval)
- [Optional Answer Verification](#optional-answer-verification)
- [Frontend UI](#frontend-ui)
- [Design Decisions](#design-decisions)

---

## Overview

PageMind is a **vectorless, agentic Retrieval-Augmented Generation (RAG)** system. Unlike traditional RAG, it doesn't encode your document into embeddings or query a vector store. Instead, it:

1. Runs **Mistral OCR** on the uploaded PDF to extract per-page markdown.
2. Builds a **hierarchical section tree** (the *PageIndex*) from document headings — title, page range, and a short summary for every section.
3. Uses a **LangGraph multi-node pipeline** where an LLM reads the tree, decides which pages are relevant, fetches only those pages, and generates a grounded answer.
4. Optionally runs a **second LLM verification pass** to check the answer for accuracy and hallucinations.

The result is fast, token-efficient retrieval that scales naturally with document size, with no embeddings, no cosine search, and no vector infrastructure.

---

## How It Works

```
PDF Upload
    │
    ▼
Mistral OCR  ──────────────────────────────────────────────────────────────────┐
    │  per-page markdown                                                        │
    ▼                                                                           │
PageIndex Builder                                                               │
    │  • scan headings (H1–H3) across all pages                                │
    │  • compute page_start / page_end for each section                        │
    │  • build nested JSON tree + 160-char summaries                           │
    ▼                                                                           │
workspace/documents/{doc_id}/                                                  │
    ├── tree.json          ← section tree (used by router node)                │
    └── pages/            ← per-page markdown files                ◄───────────┘
          ├── 1.md
          ├── 2.md
          └── …

User asks a question
    │
    ▼
LangGraph Pipeline
    │
    ├── [Node 0a] classify_intent
    │       ↓ "general"                    ↓ "document"
    │   handle_general_query          retrieve_structure
    │       ↓                               ↓
    │      END                         route_query  ◄────────────────────────┐
    │                                      ↓                                 │
    │                                  fetch_pages                           │
    │                                      ↓                                 │
    │                                 generate_answer                        │
    │                                      ↓                                 │
    │                          [verification=on?]                            │
    │                          ↓              ↓                              │
    │                        END         verify_answer                       │
    │                                      ↓            ↓                   │
    │                                  [valid]     [invalid + retries left]──┘
    │                                    ↓
    │                                   END
    ▼
Answer  (+  optional verification feedback)
```

---

## Architecture

```
vectorless-agentic-rag/
├── backend/                    FastAPI application
│   ├── main.py                 REST API + lifespan startup
│   ├── graph/                  LangGraph pipeline
│   │   ├── state.py            RAGState TypedDict
│   │   ├── nodes.py            All 7 async node functions
│   │   ├── edges.py            Conditional routing logic
│   │   └── builder.py          Graph assembly & compilation
│   ├── indexer/
│   │   └── indexer.py          Mistral OCR + PageIndex builder
│   ├── models/
│   │   └── models.py           Pydantic request / response models
│   ├── uploads/                Temporary PDF storage (cleared on startup)
│   ├── workspace/documents/    Persistent per-document index
│   └── requirements.txt
└── frontend/                   React + Vite SPA
    └── src/
        ├── App.jsx             Full application UI
        └── index.css           Design system & component styles
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Agentic Pipeline** | LangGraph |
| **LLM** | Mistral AI (`ministral-8b-latest`, `ministral-3b-latest`) |
| **OCR** | Mistral OCR (`mistral-ocr-latest`) |
| **Data Validation** | Pydantic v2 |
| **Frontend** | React 19 + Vite |
| **Styling** | Vanilla CSS (custom design system) |

---

## Project Structure

### Backend — `backend/`

| File / Dir | Purpose |
|---|---|
| `main.py` | FastAPI app with `/upload`, `/query`, `/cleanup/{doc_id}`, and `/health` endpoints |
| `graph/state.py` | `RAGState` TypedDict — the shared state object passed between all LangGraph nodes |
| `graph/nodes.py` | Seven async node functions (intent classifier, conversational handler, structure retriever, page router, page fetcher, answer generator, answer verifier) |
| `graph/edges.py` | Three conditional routing functions (`route_after_classification`, `route_after_generation`, `route_after_verification`) |
| `graph/builder.py` | `build_graph()` — assembles and compiles the LangGraph `StateGraph` |
| `indexer/indexer.py` | `index_pdf()`, `get_document_structure()`, `get_page_content()`, `cleanup_document()` — the full indexing and retrieval layer |
| `models/models.py` | `UploadResponse`, `QueryRequest`, `QueryResponse`, `CleanupResponse` Pydantic models |
| `uploads/` | Temporary storage for uploaded PDFs (wiped on every server startup) |
| `workspace/documents/` | Long-lived document workspace: one directory per `doc_id`, containing `tree.json` and `pages/*.md` |

### Frontend — `frontend/`

| File | Purpose |
|---|---|
| `src/App.jsx` | Single-page application: upload flow, chat history, message rendering, copy-to-clipboard, verification toggle, drag-and-drop, "How it works" modal |
| `src/index.css` | Full design system — CSS custom properties, sidebar, chat area, drop zone, message bubbles, spinner, modal, tooltips |

---

## API Reference

### `GET /health`
Basic liveness probe. Returns `{ "status": "ok" }`.

---

### `POST /upload`
Upload and index a PDF document.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | `File` | PDF document (`.pdf` extension required) |

**Response:** `200 OK`
```json
{ "doc_id": "550e8400-e29b-41d4-a716-446655440000" }
```

The `doc_id` is a UUID that identifies the indexed document for all subsequent requests.

**Errors:**
- `400` — non-PDF file uploaded
- `500` — OCR or indexing failure

---

### `POST /query`
Run the agentic RAG pipeline against a previously uploaded document.

**Request:** `application/json`
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_query": "What are the main conclusions of Chapter 3?",
  "verification": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `doc_id` | `string` | — | UUID returned by `/upload` |
| `user_query` | `string` | — | Natural-language question |
| `verification` | `boolean` | `false` | Enable second-pass answer verification |

**Response:** `200 OK`
```json
{
  "llm_answer": "Chapter 3 concludes that ...",
  "llm_feedback": "VALID: true — The answer is fully supported by pages 42-47."
}
```

`llm_feedback` is `null` when `verification` is `false`.

**Errors:**
- `404` — `doc_id` not found (document may need re-uploading)
- `500` — pipeline execution error

---

### `DELETE /cleanup/{doc_id}`
Delete a document's workspace and any stored PDF from disk.

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Document 550e8400... cleaned successfully."
}
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+ and npm
- A [Mistral AI](https://console.mistral.ai/) account with API access (both a chat key and an OCR key)

---

### Environment Variables

Create a `.env` file in the project root:

```env
# Mistral chat completions key (used by LangGraph nodes)
MISTRAL_API_KEY=your_mistral_api_key_here

# Mistral OCR key (used by the indexer)
MISTRAL_OCR_API_KEY=your_mistral_ocr_key_here

# Frontend URL for CORS (optional, defaults covered by localhost origins)
FRONTEND_URL=http://localhost:5173
```

> **Note:** Both keys can be the same if your Mistral account grants access to both the chat and OCR APIs under a single key.

The frontend reads one additional variable from `.env` (auto-loaded by Vite):

```env
# Backend base URL consumed by the React app
VITE_BACKEND_URL=http://localhost:8000
```

---

### Backend Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start the development server
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## The Agentic Pipeline (LangGraph)

The pipeline is defined in `backend/graph/` and runs as an async LangGraph `StateGraph`. All state is passed through a single `RAGState` TypedDict.

### Nodes

| Node | Function | Description |
|---|---|---|
| `classify_intent` | `classify_intent()` | Classifies the user message as `"general"` (greetings, chit-chat, off-topic) or `"document"` (requires PDF retrieval). Uses `ministral-8b-latest`. |
| `handle_general_query` | `handle_general_query()` | Responds conversationally to general messages using `ministral-3b-latest` (warmer temperature). Skips the entire RAG path. |
| `retrieve_structure` | `retrieve_structure()` | Loads `tree.json` from disk and formats it as a human-readable outline for the LLM. |
| `route_query` | `route_query()` | Reads the document outline + user question and outputs a page range (`"5"`, `"3-8"`, `"3,8"`). On retries, it receives feedback about why the previous range was insufficient and picks a different one. |
| `fetch_pages` | `fetch_pages()` | Reads the selected pages' markdown files from disk. Tracks all attempted page ranges to avoid infinite loops. |
| `generate_answer` | `generate_answer()` | Generates a strictly-grounded answer using only the fetched page content. Will explicitly say "not found" if the answer is absent. |
| `verify_answer` | `verify_answer()` | (Optional) Second LLM pass that checks: (1) is the answer supported by the source?, (2) does it address the question?, (3) does it contain hallucinations? Returns structured `VALID: true/false` + `FEEDBACK:` output. |

### Edges & Routing

| Routing function | Trigger | Branches |
|---|---|---|
| `route_after_classification` | After `classify_intent` | `"general"` → `handle_general_query` \| `"document"` → `retrieve_structure` |
| `route_after_generation` | After `generate_answer` | `verification=True` → `verify_answer` \| `verification=False` → `END` |
| `route_after_verification` | After `verify_answer` | `answer_valid=True` → `END` \| retries exhausted → `END` \| otherwise → `route_query` (retry loop) |

**Retry limit:** `MAX_RETRIES = 3` (defined in `edges.py`). After three failed page ranges, the pipeline returns the best-effort answer.

---

## The PageIndex — Tree-Based Retrieval

The `indexer/indexer.py` module implements the "PageIndex" — the core innovation that replaces vector embeddings.

### Indexing Steps

1. **OCR** — The PDF is base64-encoded and sent to `mistral-ocr-latest`. The response contains per-page markdown with preserved structure (headings, tables, lists).

2. **Heading extraction** — A regex (`^(#{1,3})\s+(.+)`) scans every page for H1–H3 headings, recording `(page_number, level, title)` tuples.

3. **Page range computation** — Each section's `page_end` is set to one before the next heading at the same or higher level (or the last page of the document).

4. **Summary generation** — The first 160 characters of each section's body text (headings stripped) are stored as a `summary` field to give the LLM quick context.

5. **Tree nesting** — A stack-based algorithm converts the flat list into a nested hierarchy (`H2` sections become children of their parent `H1`, etc.).

6. **Persistence** — `tree.json` and `pages/{n}.md` files are written to `workspace/documents/{doc_id}/`.

### Fallback (no headings)

If a PDF has no detectable headings (e.g., a scanned image-only document), the indexer creates one section per page titled `"Page N"` so the router can still operate.

### Tree-to-Readable Format

When passed to the LLM router, the tree is serialized as a compact outline:

```
Total pages: 48
[Pages 1-2] Introduction — overview of the product roadmap
[Pages 3-18] Part One: Foundations
  [Pages 3-8] Chapter 1 — History of the project and initial motivation
  [Pages 9-18] Chapter 2 — Technical architecture and key design decisions
[Pages 19-48] Part Two: Implementation
  ...
```

This lets the router find the right page range with a single LLM call at very low token cost.

---

## Optional Answer Verification

When `verification: true` is passed with a query, the pipeline runs a dedicated `verify_answer` node after answer generation.

The verifier checks three criteria:
1. Is the answer **supported** by the retrieved document content?
2. Does it **directly address** the user's question?
3. Does it contain **hallucinated** information not present in the source?

It responds in a structured format:
```
VALID: true
FEEDBACK: The answer is fully grounded in pages 12-15 and directly addresses the user's question about the company's Q2 revenue figures.
```

If `VALID: false` and retries remain, the pipeline loops back to `route_query` to try different pages, passing the feedback as context so the router avoids repeating the same mistake.

---

## Frontend UI

The frontend is a React 19 SPA built with Vite and pure CSS.

### Key Features

- **Sidebar + Chat layout** — Persistent navigation sidebar with document status indicator; scrollable central chat area.
- **Drag-and-drop upload zone** — Visual feedback for dragging, file selection, and upload states. Accepts PDF files only.
- **Persistent chat history** — Full conversation retained in component state; auto-scrolls to latest message.
- **Markdown rendering** — Custom `parseMarkdown()` function handles bold (`**text**`), unordered/ordered lists, and line spacing without any external parsing library.
- **Answer verification toggle** — Checkbox with a tooltip that explains what verification does. Wired directly to the `verification` field in the query payload.
- **Copy-to-clipboard** — Per-message copy button with a check-mark confirmation animation.
- **"How it Works" modal** — Dismissible step-by-step modal explaining the full pipeline.
- **Thinking indicator** — Animated spinner shown while the LangGraph pipeline is running.

### UI States

| State | Displayed |
|---|---|
| `idle` | Upload drop zone + hero text |
| `indexing` | Spinner button, "Indexing…" label in sidebar |
| `ready` | Chat input enabled, status dot turns green |
| `error` | Error toast with message |

---

## Design Decisions

**Why no vector database?**
Vector databases require embedding models, storage infrastructure, and chunking heuristics that introduce complexity and can split context in unhelpful ways. The PageIndex uses the document's own structure (its headings) as a navigational map, which is both more token-efficient and more interpretable — the LLM can read "Chapter 3: Results, pages 22–31" and decide directly.

**Why LangGraph?**
LangGraph provides clean, inspectable conditional branching with typed state. The retry loop (route on verification failure → re-route → re-fetch → re-generate) is trivial to express as a graph with conditional edges, and impossible to express cleanly as a simple chain.

**Why Mistral?**
Mistral provides both a high-quality OCR API (`mistral-ocr-latest`) and capable chat models at low cost. Using a single provider keeps configuration simple. `ministral-8b-latest` handles structured tasks (routing, generation, verification); `ministral-3b-latest` handles conversational replies at lower latency.

**Why two separate Mistral API keys?**
Mistral's OCR API (`/ocr`) uses a different billing quota from the chat completion API. The codebase separates them via `MISTRAL_API_KEY` and `MISTRAL_OCR_API_KEY` so you can use different keys or rate-limit them independently.

**Stale upload cleanup on startup**
The server clears the `uploads/` directory on every startup. Uploaded PDFs are only needed during the OCR step; once pages are persisted as markdown, the original PDF is no longer needed. This prevents disk accumulation across server restarts.

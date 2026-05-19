import os
import re
import json
import uuid
import base64
import shutil
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────
WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"
UPLOADS_DIR   = Path(__file__).parent.parent / "uploads"

# ── Mistral client ────────────────────────────────────────────────────
_mistral = Mistral(api_key=os.getenv("MISTRAL_OCR_API_KEY"))



def index_pdf(file_path: str) -> str:
    """
    Full indexing pipeline:
      1. Mistral OCR  -> per-page markdown
      2. Tree builder -> hierarchical section index
      3. Persist      -> workspace/documents/{doc_id}/

    Returns the doc_id used for all subsequent lookups.
    """
    doc_id = str(uuid.uuid4())
    doc_dir = _doc_dir(doc_id)
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: OCR
    pages_markdown = _run_ocr(file_path)

    # Step 2: persist per-page markdown
    for i, md in enumerate(pages_markdown, start=1):
        (pages_dir / f"{i}.md").write_text(md, encoding="utf-8")

    # Step 3: build & persist tree
    tree = _build_tree(pages_markdown)
    (doc_dir / "tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")

    return doc_id


def get_document_structure(doc_id: str) -> str:
    """
    Return the tree JSON as a compact, LLM-readable string.
    The LangGraph route_query node uses this to pick page ranges.
    """
    tree_path = _doc_dir(doc_id) / "tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(f"No tree found for doc_id={doc_id}")

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    return _tree_to_readable(tree)


def get_page_content(doc_id: str, pages: str) -> str:
    """
    Retrieve OCR markdown for the requested pages.
    pages format: "5-7" | "3,8" | "12"
    """
    pages_dir = _doc_dir(doc_id) / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"No pages found for doc_id={doc_id}")

    page_nums = _parse_page_range(pages)
    chunks = []
    for n in sorted(page_nums):
        p = pages_dir / f"{n}.md"
        if p.exists():
            chunks.append(f"[Page {n}]\n{p.read_text(encoding='utf-8')}")
        else:
            chunks.append(f"[Page {n}]\n(Page not available)")

    return "\n\n---\n\n".join(chunks) if chunks else "No content found for requested pages."


def cleanup_document(doc_id: str) -> bool:
    """
    Delete the document's workspace directory and any uploaded file.
    """
    try:
        doc_dir = _doc_dir(doc_id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

        # remove uploaded PDF(s)
        for f in UPLOADS_DIR.iterdir():
            if f.stem == doc_id:
                f.unlink(missing_ok=True)

        return True
    except Exception as e:
        print(f"[cleanup_document] Error for {doc_id}: {e}")
        return False



def _doc_dir(doc_id: str) -> Path:
    return WORKSPACE_DIR / "documents" / doc_id


def _run_ocr(file_path: str) -> list[str]:
    """
    Call Mistral OCR on the PDF.
    Returns a list of markdown strings, one per page (1-indexed → index 0 = page 1).
    """
    with open(file_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = _mistral.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        include_image_base64=False,
    )

    # response.pages is a list ordered by page index
    return [page.markdown or "" for page in response.pages]


def _build_tree(pages: list[str]) -> dict:
    """
    Build a hierarchical section tree from per-page markdown.

    Tree schema:
    {
      "total_pages": N,
      "sections": [
        {
          "title":      "Chapter 1 — The Beginning",
          "level":      1,          # heading depth: 1=H1, 2=H2, 3=H3
          "page_start": 3,
          "page_end":   8,
          "summary":    "First 120 chars of section text...",
          "children": [ ... ]       # nested sub-sections
        },
        ...
      ]
    }
    """
    flat = _extract_flat_sections(pages)
    nested = _nest_sections(flat)

    return {
        "total_pages": len(pages),
        "sections": nested,
    }


def _extract_flat_sections(pages: list[str]) -> list[dict]:
    """
    Walk each page's markdown looking for headings (#, ##, ###).
    Each heading starts a new section; the section's page range runs
    from its heading page until the next same-or-higher-level heading.
    """
    heading_re = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

    # collect all (page_num, level, title) events
    events = []
    for page_num, md in enumerate(pages, start=1):
        for match in heading_re.finditer(md):
            level = len(match.group(1))
            title = match.group(2).strip()
            events.append({"level": level, "title": title, "page_start": page_num})

    if not events:
        # Fallback: no headings found — create one section per page
        return [
            {
                "level": 1,
                "title": f"Page {i}",
                "page_start": i,
                "page_end": i,
                "summary": pages[i - 1][:120].replace("\n", " "),
                "children": [],
            }
            for i in range(1, len(pages) + 1)
        ]

    # compute page_end for each event
    total = len(pages)
    flat = []
    for i, ev in enumerate(events):
        page_end = events[i + 1]["page_start"] - 1 if i + 1 < len(events) else total
        page_end = max(ev["page_start"], page_end)  # guarantee ≥ start

        # collect page text for summary
        section_text = "\n".join(pages[ev["page_start"] - 1 : page_end])
        summary = re.sub(r"#+ .+\n?", "", section_text)  # strip heading lines
        summary = " ".join(summary.split())[:160]         # first 160 chars

        flat.append({
            "level":      ev["level"],
            "title":      ev["title"],
            "page_start": ev["page_start"],
            "page_end":   page_end,
            "summary":    summary,
            "children":   [],
        })

    return flat


def _nest_sections(flat: list[dict]) -> list[dict]:
    """
    Convert the flat list into a nested hierarchy based on heading level.
    H1 → top-level, H2 → children of preceding H1, H3 → children of preceding H2.
    """
    root: list[dict] = []
    stack: list[dict] = []   # (level, node) stack

    for node in flat:
        # pop stack until parent level < current level
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def _tree_to_readable(tree: dict) -> str:
    """
    Render the tree as a compact, LLM-readable outline string.

    Example output:
      Total pages: 312
      [Pages 1-2] Introduction — overview of the memoir
        [Pages 3-18] Part One: The Desert — family life in the Southwest
          [Pages 3-8]  # The Walls Family — Rex and Rose Mary...
    """
    lines = [f"Total pages: {tree.get('total_pages', '?')}"]

    def _render(nodes: list[dict], depth: int = 0):
        indent = "  " * depth
        for n in nodes:
            page_info = f"[Pages {n['page_start']}-{n['page_end']}]" \
                        if n["page_start"] != n["page_end"] \
                        else f"[Page {n['page_start']}]"
            summary = f" — {n['summary']}" if n.get("summary") else ""
            lines.append(f"{indent}{page_info} {n['title']}{summary}")
            _render(n.get("children", []), depth + 1)

    _render(tree.get("sections", []))
    return "\n".join(lines)


def _parse_page_range(pages: str) -> set:
    """
    Parse page range string into a set of page numbers.
    "5-7" → {5, 6, 7}
    "3,8" → {3, 8}
    "12"  → {12}
    """
    nums = set()
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            nums.update(range(int(lo), int(hi) + 1))
        else:
            if part.isdigit():
                nums.add(int(part))
    return nums
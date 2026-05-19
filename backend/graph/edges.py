from graph.state import RAGState

MAX_RETRIES = 3


def route_after_classification(state: RAGState) -> str:
    """
    After the intent classifier runs:
    - "general"  → handle conversationally, skip the RAG pipeline
    - "document" → run the full RAG pipeline
    """
    if state.get("query_intent") == "general":
        return "handle_general_query"
    return "retrieve_structure"


def route_after_generation(state: RAGState) -> str:
    """
    After Node 4 generates an answer:
    - if verification is enabled -> go to Node 5
    - if verification is disabled -> end immediately
    """
    if state.get("verification"):
        return "verify_answer"
    else:
        return "end"


def route_after_verification(state: RAGState) -> str:
    """
    After Node 5 verifies the answer:
    - if answer is valid -> end
    - if retries exhausted -> end anyway (return best effort answer)
    - if answer is invalid and retries remain -> go back to Node 3
    """
    if state.get("answer_valid"):
        return "end"

    attempts = len(state.get("pages_already_tried", []))
    if attempts >= MAX_RETRIES:
        return "end"

    return "route_query"
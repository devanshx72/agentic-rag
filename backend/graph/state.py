from typing import TypedDict, Optional, List


class RAGState(TypedDict):
    # inputs
    user_query: str
    doc_id: str
    verification: bool

    # intent classification
    query_intent: Optional[str]

    # node 1 produces
    document_tree: Optional[str]

    # node 2 produces
    page_range: Optional[str]

    # node 3 produces + updates
    content: Optional[str]
    pages_already_tried: List[str]

    # node 4 produces
    answer: Optional[str]

    # node 5 produces (if verification=True)
    answer_valid: Optional[bool]
    verification_feedback: Optional[str]
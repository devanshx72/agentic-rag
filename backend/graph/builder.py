from langgraph.graph import StateGraph, START, END

from graph.state import RAGState
from graph.nodes import (
    classify_intent,
    handle_general_query,
    retrieve_structure,
    route_query,
    fetch_pages,
    generate_answer,
    verify_answer
)
from graph.edges import (
    route_after_classification,
    route_after_generation,
    route_after_verification
)


def build_graph():
    """
    Assemble and compile the LangGraph agentic RAG pipeline.

    Flow:
    classify_intent
        → [general]  handle_general_query → END
        → [document] retrieve_structure
                         → route_query
                             → fetch_pages
                                 → generate_answer
                                     → [conditional] verify_answer or END
                                         → [conditional] route_query (retry) or END
    """

    graph = StateGraph(RAGState)

    # register nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_general_query", handle_general_query)
    graph.add_node("retrieve_structure", retrieve_structure)
    graph.add_node("route_query", route_query)
    graph.add_node("fetch_pages", fetch_pages)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)

    # entry point → intent classifier
    graph.add_edge(START, "classify_intent")

    # branch after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "handle_general_query": "handle_general_query",
            "retrieve_structure": "retrieve_structure",
        }
    )

    # general path exits immediately
    graph.add_edge("handle_general_query", END)

    # document RAG path
    graph.add_edge("retrieve_structure", "route_query")
    graph.add_edge("route_query", "fetch_pages")
    graph.add_edge("fetch_pages", "generate_answer")

    # conditional edge after generation
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {
            "verify_answer": "verify_answer",
            "end": END
        }
    )

    # conditional edge after verification
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verification,
        {
            "end": END,
            "route_query": "route_query"   # retry loop
        }
    )

    # compile
    compiled = graph.compile()

    return compiled
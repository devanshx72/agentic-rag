from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import RAGState
from indexer.indexer import (
    get_document_structure,
    get_page_content
)

llm = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0.3
)

llm_chat = ChatMistralAI(
    model="ministral-3b-latest",
    temperature=0.7  # slightly warmer for conversational replies
)


# Node 0a — Intent Classifier
async def classify_intent(state: RAGState) -> dict:
    """
    Quickly decide if the user's message is:
      - "general"  → greeting, chit-chat, thanks, general knowledge question
      - "document" → something that should be answered from the uploaded PDF
    """
    messages = [
        SystemMessage(content="""\
You are an intent classifier for a document Q&A assistant called PageMind.

Classify the user message into exactly one of these two categories:
  general  — greetings, thanks, chit-chat, general knowledge, anything not about a specific uploaded document
  document — a question that should be answered using the content of an uploaded PDF

Reply with ONLY one word: general  OR  document
Do NOT explain."""),
        HumanMessage(content=state["user_query"])
    ]

    response = await llm.ainvoke(messages)
    intent = response.content.strip().lower()

    # Normalise — if LLM returns anything unexpected, default to "document"
    if intent not in ("general", "document"):
        intent = "document"

    return {"query_intent": intent}


# Node 0b — General / Conversational Handler
async def handle_general_query(state: RAGState) -> dict:
    """
    Answer greetings, general knowledge, and off-topic messages
    in a friendly, helpful way — without touching the document.
    """
    messages = [
        SystemMessage(content="""\
You are PageMind, a friendly and helpful AI assistant that specialises in \
answering questions about uploaded PDF documents.

Right now the user is not asking about a document — they are making conversation \
or asking a general question. Respond naturally and helpfully.

Guidelines:
- Be warm, concise, and conversational.
- If they greet you, greet them back and briefly mention you can help them \
  explore their uploaded document.
- If they ask a general knowledge question you can answer it.
- If they ask something personal or impossible to know (e.g. "what is my \
  father's name"), politely say you don't know that.
- Keep replies short — 1-3 sentences unless more detail is genuinely needed."""),
        HumanMessage(content=state["user_query"])
    ]

    response = await llm_chat.ainvoke(messages)
    return {"answer": response.content.strip()}


# Node 1
async def retrieve_structure(state: RAGState) -> dict:
    """
    Fetch the PageIndex tree structure for the document.
    This is the 'look at table of contents' step.
    """
    tree = get_document_structure(state["doc_id"])

    return {"document_tree": tree}


# Node 2
async def route_query(state: RAGState) -> dict:
    """
    LLM reads the tree + user question and decides
    which page range is most relevant.
    """
    already_tried = state.get("pages_already_tried", [])
    feedback = state.get("verification_feedback")

    # build context about what was already tried
    retry_context = ""
    if already_tried:
        retry_context = f"""
Previous attempts retrieved these page ranges: {', '.join(already_tried)}.
They were insufficient because: {feedback or 'answer could not be verified'}.
Pick a DIFFERENT page range this time.
"""

    messages = [
        SystemMessage(content="""
You are a document navigation assistant.
You will be given a document tree structure and a user question.
Your job is to identify the most relevant page range to answer the question.

Rules:
- Respond with ONLY a page range in one of these formats:
  - Single page: "5"
  - Range: "5-8"
  - Multiple: "5,8"
- Do NOT explain your choice.
- Do NOT include any other text.
- Pick the most specific and relevant section.
- Avoid page ranges already tried.
"""),
        HumanMessage(content=f"""
Document tree:
{state['document_tree']}

User question:
{state['user_query']}

{retry_context}

Respond with only the page range:
""")
    ]

    response = await llm.ainvoke(messages)
    page_range = response.content.strip()

    return {"page_range": page_range}


# Node 3
async def fetch_pages(state: RAGState) -> dict:
    """
    Fetch actual text content of the selected page range.
    Track which pages have been tried to avoid repeating.
    """
    page_range = state["page_range"]

    content = get_page_content(state["doc_id"], page_range)

    # update the list of already tried page ranges
    already_tried = state.get("pages_already_tried", [])
    updated_tried = already_tried + [page_range]

    return {
        "content": content,
        "pages_already_tried": updated_tried
    }


# Node 4
async def generate_answer(state: RAGState) -> dict:
    """
    Generate an answer using the retrieved content.
    Strictly grounded — LLM must only use the provided context.
    """
    messages = [
        SystemMessage(content="""
You are a document question-answering assistant.

Rules:
- Answer ONLY from the provided document content.
- If the answer is not found in the content, respond with:
  "The answer to this question was not found in the provided document."
- Do NOT use any external knowledge.
- Be concise and precise.
- Cite the relevant section or page when possible.
"""),
        HumanMessage(content=f"""
Document content:
{state['content']}

User question:
{state['user_query']}

Answer:
""")
    ]

    response = await llm.ainvoke(messages)

    return {"answer": response.content.strip()}


# Node 5
async def verify_answer(state: RAGState) -> dict:
    """
    Verify that the generated answer is grounded
    in the retrieved content and actually answers the question.
    Only runs if verification=True.
    """
    messages = [
        SystemMessage(content="""
You are a strict answer verification assistant.

Your job is to check if a given answer:
1. Is actually supported by the provided document content
2. Directly addresses the user's question
3. Does not contain hallucinated information

Respond in this exact format:
VALID: true or false
FEEDBACK: one sentence explaining why the answer is valid or what is missing
"""),
        HumanMessage(content=f"""
User question:
{state['user_query']}

Document content used:
{state['content']}

Generated answer:
{state['answer']}

Verify:
""")
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()
    
    answer_valid = False
    feedback = raw

    for line in raw.splitlines():
        if line.startswith("VALID:"):
            value = line.replace("VALID:", "").strip().lower()
            answer_valid = value == "true"
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()

    return {
        "answer_valid": answer_valid,
        "verification_feedback": feedback
    }
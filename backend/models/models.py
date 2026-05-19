from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    doc_id: str


class QueryRequest(BaseModel):
    doc_id: str
    user_query: str
    verification: bool = False


class QueryResponse(BaseModel):
    llm_answer: str
    llm_feedback: Optional[str] = None


class CleanupResponse(BaseModel):
    success: bool
    message: str
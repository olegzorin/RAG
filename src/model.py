import logging
from typing import Optional, List

from pydantic import BaseModel


class RagDocument(BaseModel):
    url: str
    documentId: int
    contentType: Optional[str] = "application/pdf"


class RagDocumentResponse(BaseModel):
    documentId: int = 0
    status: int = 0
    errorMessage: str = None


class ActionRequest(BaseModel):
    questions: Optional[List[str]] = None
    documents: Optional[List[RagDocument]] = None
    logLevel: Optional[int] = logging.WARNING


class ActionResponse(BaseModel):
    success: bool = False
    answers: List[str] = None
    documents: List[RagDocumentResponse] = None
    errorMessage: str = None

import logging
from typing import Optional, List

from pydantic import BaseModel


class RagDocumentRequest(BaseModel):
    url: str
    documentId: int
    contentType: Optional[str] = "application/pdf"


class ActionRequest(BaseModel):
    questions: List[str]
    documents: List[RagDocumentRequest]
    logLevel: Optional[int] = logging.WARNING


class ActionResponse(BaseModel):
    success: bool = False
    answers: List[str] = None
    errorMessage: str = None

from typing import List

from pydantic import BaseModel


class ActionDocument(BaseModel):
    url: str
    documentId: int


class ActionRequest(BaseModel):
    questions: List[str]
    documents: List[ActionDocument]


class ActionResponse(BaseModel):
    success: bool = False
    answers: List[str] = None
    errorMessage: str = None

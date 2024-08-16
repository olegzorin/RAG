from typing import List, Optional, Any

from pydantic import BaseModel, Json


class ActionRequest(BaseModel):
    document_id: int
    url: str
    questions: List[str]
    template: Optional[Json[Any]] = None
    examples: Optional[Json[Any]] = None
    params: Optional[dict] = {}


class ActionResponse(BaseModel):
    success: bool = False
    answers: List[str] = None
    outputs: List[Json] = None
    errorMessage: str = None

import time
from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from conf import set_logging
from core import RagSearch
from main import Response, Executor
from reader import PdfDoc, read_pdf

set_logging()


class RagRequest(BaseModel):
    documentId: int
    url: str
    questions: Optional[List[str]] = None
    template: Optional[Any] = None
    examples: Optional[List[Any]] = None
    params: Optional[Dict] = {}


class RagResponse(Response):
    answers: List[str] = None
    outputs: List[Any] = None
    executionTime: int = 0


param_keys = [
    "chat_prompt_system_message",
    "reader.chunk_size",
    "reader.chunk_overlap",
    "search.method",
    "search.type",
    "embeddings.model",
    "embeddings.use_gpu",
    "ollama.model",
    "ollama.temperature",
    "ollama.seed",
    "ollama.top_k",
    "ollama.top_p",
    "ollama.num_ctx",
    "ollama.num_predict",
    "output.model",
    "output.use_gpu",
    "tokenizer.max_length",
    "keybert.model"
]


def _get_rag_search(params: Dict) -> Any:
    search_method = params.get("search.method", "vector")
    if search_method == 'vector':
        from vector import VectorSearch
        return VectorSearch(params)
    elif search_method == 'graph':
        from graph import GraphSearch
        return GraphSearch(params)
    elif search_method == 'keywords':
        from keywords import KeywordSearch
        return KeywordSearch(params)
    else:
        raise AttributeError(f"Invalid search method {search_method}")


class RagExecutor(Executor):

    def execute(
            self,
            request: RagRequest,
            response: RagResponse
    ) -> None:
        document: PdfDoc = read_pdf(
            document_id=request.documentId,
            source=request.url
        )

        start_time = time.time_ns()

        if request.questions:
            rag_search: RagSearch = _get_rag_search(request.params)
            answers = rag_search.get_answers(document, request.questions)

            response.outputs = rag_search.generate_outputs(
                texts=answers,
                template=request.template,
                examples=request.examples
            )
            response.answers = answers

        response.executionTime = (time.time_ns() - start_time) // 1000_000


if __name__ == "__main__":
    RagExecutor()(
        request_type=RagRequest,
        response=RagResponse()
    )

import base64
import json
import logging
import re
import sys
import time
from typing import List, Optional, Dict, Any

from pydantic import TypeAdapter, ValidationError, BaseModel

from conf import set_logging

set_logging()


class RagRequest(BaseModel):
    documentId: int
    url: str
    questions: Optional[List[str]] = None
    template: Optional[Any] = None
    examples: Optional[List[Any]] = None
    params: Optional[Dict] = {}


class RagResponse(BaseModel):
    success: bool = False
    answers: List[str] = None
    outputs: List[Any] = None
    errorMessage: str = None
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


def process_rag_request(
        request: RagRequest,
        response: RagResponse
) -> None:
    from core import RagSearch
    from reader import ExtractedDoc
    from reader import read_pdf

    document: ExtractedDoc = read_pdf(
        document_id=request.documentId,
        source=request.url
    )

    start_time = time.time_ns()

    if request.questions:
        rag_search: RagSearch = _get_rag_search(request.params)
        answers = rag_search.get_answers(document, request.questions)

        template = _get_non_empty_or_none(request.template)
        examples = _get_non_empty_or_none(request.examples)
        if template is not None or examples is not None:
            response.outputs = rag_search.generate_outputs(
                texts=answers,
                template=template,
                examples=examples
            )
        response.answers = answers

    response.executionTime = (time.time_ns() - start_time) // 1000_000
    response.success = True


def run(
        data: str,
        outpath: Optional[str] = None
) -> None:
    response = RagResponse()

    if not data:
        response.errorMessage = 'No input'
    else:
        try:
            decoded = base64.b64decode(data)
            request = TypeAdapter(RagRequest).validate_json(decoded)

            for k in request.params.keys():
                if not k in param_keys:
                    raise KeyError(f"Invalid param key: {k}")
            process_rag_request(request, response)

        except ValidationError as ve:
            response.errorMessage = f'Wrong action input, {ve}'
        except RuntimeError as ex:
            response.errorMessage = f"{ex}"
        except KeyError as ke:
            response.errorMessage = f"{ke}"
        except Exception as e:
            logging.exception(e)
            response.errorMessage = 'Unexpected exception'

        result = TypeAdapter(RagResponse).dump_json(response, exclude_none=True)

        if outpath is not None:
            with open(outpath, 'wb') as outfile:
                outfile.write(result)
        else:
            print(result)


def _get_non_empty_or_none(json_obj):
    is_empty = json_obj is None or not json_obj or not re.search('[a-zA-Z\\d]', json.dumps(json_obj))
    return None if is_empty else json_obj


if __name__ == "__main__":
    path = sys.argv[1]
    inp = sys.argv[2] if 2 < len(sys.argv) else None
    run(inp, path)

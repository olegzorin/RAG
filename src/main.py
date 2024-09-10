import base64
import json
import logging
import re
import sys
import warnings
from typing import List, Optional, Dict, Any

from pydantic import TypeAdapter, ValidationError, BaseModel

import conf
from core import RagSearch
from graph import GraphSearch
from vector import VectorSearch

logging.basicConfig(stream=sys.stderr, level=conf.get_log_level())

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="error", category=UserWarning)


class ActionRequest(BaseModel):
    documentId: int
    url: str
    questions: List[str]
    template: Optional[Any] = None
    examples: Optional[List[Any]] = None
    params: Optional[Dict] = {}


class ActionResponse(BaseModel):
    success: bool = False
    answers: List[str] = None
    outputs: List[Any] = None
    errorMessage: str = None


param_keys = [
    "chat_prompt_system_message",
    "reader.chunk_size",
    "reader.chunk_overlap",
    "search.method",
    "search.type",
    "embeddings.model",
    "ollama.model",
    "ollama.temperature",
    "ollama.seed",
    "ollama.top_k",
    "ollama.top_p",
    "ollama.num_ctx",
    "ollama.num_predict",
    "output.model"
]

def _get_rag_search(params: Dict) -> RagSearch:
    search_method = params.get("search.method", "vector")
    if search_method == 'vector':
        return VectorSearch(params)
    elif search_method == 'graph':
        return GraphSearch(params)
    else:
        raise AttributeError(f"Invalid search method {search_method}")


def run(data: str, outpath: Optional[str] = None):
    response = ActionResponse()

    if not data:
        response.errorMessage = 'No input'
    else:
        try:
            decoded = base64.b64decode(data)
            request = TypeAdapter(ActionRequest).validate_json(decoded)

            for k in request.params.keys():
                if not k in param_keys:
                    raise KeyError(f"Invalid param key: {k}")

            rag_search: RagSearch = _get_rag_search(request.params)

            answers = rag_search.get_answers(request.documentId, request.url, request.questions)

            template = _get_non_empty_or_none(request.template)
            examples = _get_non_empty_or_none(request.examples)
            if template is not None or examples is not None:
                response.outputs = rag_search.generate_outputs(texts=answers, template=template, examples=examples)

            response.answers = answers
            response.success = True

        except ValidationError as ve:
            response.errorMessage = f'Wrong action input, {ve}'
        except RuntimeError as ex:
            response.errorMessage = f"{ex}"
        except KeyError as ke:
            response.errorMessage = f"{ke}"
        except Exception as e:
            response.errorMessage = f'Unexpected exception: {e}'

        result = TypeAdapter(ActionResponse).dump_json(response, exclude_none=True)

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

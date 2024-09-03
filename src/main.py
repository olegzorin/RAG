import base64
import json
import logging
import re
import sys
import warnings
import conf
from typing import List, Optional, Dict, Any

from core import Ragger
from pydantic import TypeAdapter, ValidationError, BaseModel

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


def run(data: str, outpath: Optional[str] = None):
    response = ActionResponse()

    if not data:
        response.errorMessage = 'No input'
    else:
        try:
            decoded = base64.b64decode(data)
            request = TypeAdapter(ActionRequest).validate_json(decoded)

            ragger = Ragger(request.params)

            answers = ragger.get_answers(request.documentId, request.url, request.questions)

            template = _get_non_empty_or_none(request.template)
            examples = _get_non_empty_or_none(request.examples)
            if template is not None or examples is not None:
                response.outputs = ragger.generate_outputs(answers, template, examples)

            response.answers = answers
            response.success = True

        except ValidationError as ve:
            response.errorMessage = f'Wrong action input, {str(ve)}'
        except RuntimeError as ex:
            response.errorMessage = str(ex)
        except Exception as ex:
            response.errorMessage = f'Unexpected exception: {str(ex)}'

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

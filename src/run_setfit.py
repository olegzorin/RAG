import base64
import logging
import sys
import warnings
from typing import List, Optional
from pydantic import TypeAdapter, ValidationError, BaseModel
from setfit import SetFitModel

from conf import set_logging, model_source_dir, model_cache_dir

set_logging()

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
)


class SetFitRequest(BaseModel):
    phrases: Optional[List[str]] = None


class SetFitResponse(BaseModel):
    success: bool = False
    errorMessage: str = None
    scores: List[list[int]] = None

def execute(
        phrases: list[str]
) -> list[list[int]]:
    model = SetFitModel.from_pretrained(
        pretrained_model_name_or_path=f'{model_source_dir}/setfit',
        cache_dir=model_cache_dir,
        local_files_only=True
    )
    res = model.predict_proba(
        inputs=phrases,
        as_numpy=True
    )
    return [
        [int(round(s * 100, 0)) for s in score]
        for score in res
    ]


def run(data: str, outpath: Optional[str] = None):
    response = SetFitResponse()

    if not data:
        response.errorMessage = 'No input'
    else:
        try:
            decoded = base64.b64decode(data)
            request = TypeAdapter(SetFitRequest).validate_json(decoded)
            response.scores = execute(request.phrases)
            response.success = True
        except ValidationError as ve:
            response.errorMessage = f'Wrong input, {ve}'
        except Exception as e:
            logging.exception(e)
            response.errorMessage = 'Unexpected exception'

        result = TypeAdapter(SetFitResponse).dump_json(response, exclude_none=True)

        if outpath is not None:
            with open(outpath, 'wb') as outfile:
                outfile.write(result)
        else:
            print(result)


if __name__ == "__main__":
    path = sys.argv[1]
    inp = sys.argv[2] if 2 < len(sys.argv) else None
    run(inp, path)

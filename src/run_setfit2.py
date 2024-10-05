import warnings
from typing import List, Optional

from pydantic import BaseModel
from setfit import SetFitModel

from conf import set_logging, model_source_dir, model_cache_dir
from main2 import Response, Executor, run

set_logging()

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
)


class SetFitRequest(BaseModel):
    phrases: Optional[List[str]] = None


class SetFitResponse(Response):
    scores: List[list[int]] = None


class SetFitExecutor(Executor):
    def execute(
            self,
            request: SetFitRequest,
            response: SetFitResponse
    ) -> None:
        model = SetFitModel.from_pretrained(
            pretrained_model_name_or_path=f'{model_source_dir}/setfit',
            cache_dir=model_cache_dir,
            local_files_only=True
        )

        res = model.predict_proba(
            inputs=request.phrases,
            as_numpy=True
        )
        response.scores = [
            [int(round(s * 100, 0)) for s in score]
            for score in res
        ]


if __name__ == "__main__":
    run(
        request_type=SetFitRequest,
        response=SetFitResponse(),
        executor=SetFitExecutor
    )

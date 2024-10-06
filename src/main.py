import logging
import sys
from abc import ABC, abstractmethod
from typing import TypeVar, Type

from pydantic import BaseModel, TypeAdapter, ValidationError

from conf import set_logging

set_logging()


class Response(BaseModel):
    success: bool = False
    errorMessage: str = None


ResponseType = TypeVar(
    name='ResponseType',
    bound=Response
)


class Executor(ABC):

    @abstractmethod
    def execute(
            self,
            request: BaseModel,
            response: Response
    ) -> None:
        pass

    def __call__(
            self,
            request_type: Type,
            response: Response
    ) -> None:
        inpfile = sys.argv[1]
        outfile = sys.argv[2]

        with open(inpfile, 'rb') as f:
            data = f.read()

        if not data:
            response.errorMessage = 'No input'
        else:
            try:
                request = TypeAdapter(request_type).validate_json(data)
                self.execute(request=request, response=response)
                response.success = True

            except ValidationError as ve:
                response.errorMessage = f'Wrong action input, {ve}'
            except RuntimeError as ex:
                response.errorMessage = f"{ex}"
            except KeyError as ke:
                response.errorMessage = f"{ke}"
            except Exception as e:
                logging.exception(e)
                response.errorMessage = 'Unexpected exception'

            result = TypeAdapter(ResponseType).dump_json(response, exclude_none=True)

            with open(outfile, 'wb') as f:
                f.write(result)

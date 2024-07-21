import logging
import sys

from pydantic import TypeAdapter, ValidationError

from core import load_documents_and_answer_questions
from model import ActionResponse, ActionRequest


def run(data: str):
    response = ActionResponse()

    if data is None:
        response.errorMessage = 'No input'
    else:
        try:
            request = TypeAdapter(ActionRequest).validate_json(data)
            response.answers = load_documents_and_answer_questions(request.documents, request.questions)
            response.success = True
        except ValidationError as ve:
            response.errorMessage = f'Wrong action input, {str(ve)}'
        except Exception as ex:
            response.errorMessage = f'Unexpected exception, {ex=}, {type(ex)=}'

    print(TypeAdapter(ActionResponse).dump_json(response).decode('utf8'))


if __name__ == "__main__":
    inp = sys.argv[1] if 1 < len(sys.argv) else None
    run(inp)

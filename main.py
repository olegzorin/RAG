import sys

from pydantic import TypeAdapter, ValidationError

from core import load_documents, answer_questions
from model import ActionResponse, ActionRequest


def print_response(resp: ActionResponse):
    print(TypeAdapter(ActionResponse).dump_json(resp).decode('utf8'))


def print_error_message(error_message: str):
    response = ActionResponse()
    response.errorMessage = error_message
    print_response(response)


def main():
    if len(sys.argv) < 2:
        print_error_message('No action data')
        return

    try:
        action_request = TypeAdapter(ActionRequest).validate_json(sys.argv[1])

        if action_request.documents is not None:
            print_response(load_documents(action_request.documents, 'vector', 'Chunk'))
        elif action_request.questions is not None:
            print_response(answer_questions(action_request.questions, 'vector', 'Chunk'))
        else:
            print_error_message('Empty action request')

    except ValidationError as ve:
        print_error_message(f'Wrong action input, str{ve}')
    except Exception as ex:
        print_error_message(f'Unexpected exception, {ex=}, {type(ex)=}')


if __name__ == "__main__":
    main()

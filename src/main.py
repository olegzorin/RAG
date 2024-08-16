import sys
import base64

from PyPDF2.errors import PdfReadError
from pydantic import TypeAdapter, ValidationError

from core import process_request
from model import ActionResponse, ActionRequest


def run(data: str):
    response = ActionResponse()

    if not data:
        response.errorMessage = 'No input'
    else:
        try:
            decoded = base64.b64decode(data)
            request = TypeAdapter(ActionRequest).validate_json(decoded)
            process_request(request, response)
            response.success = True
        except ValidationError as ve:
            response.errorMessage = f'Wrong action input, {str(ve)}'
        except PdfReadError as re:
            response.errorMessage = str(re)
        except Exception as ex:
            response.errorMessage = f'Unexpected exception: {str(ex)}'

    print(TypeAdapter(ActionResponse).dump_json(response).decode('utf8'))


if __name__ == "__main__":
    inp = sys.argv[1] if 1 < len(sys.argv) else None
    run(inp)

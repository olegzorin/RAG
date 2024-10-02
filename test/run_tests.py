import logging
import shutil

import reader
from main import ActionResponse, ActionRequest, process_request
from test_utils import test_configs, questions, TestResult, DOCS_DIR

logging.basicConfig(
    level=logging.INFO,
    force=True
)

doc_names = ['CCR']  # , 'LHS', 'MHR']

for doc_name in doc_names:
    doc_path = f'{DOCS_DIR}/{doc_name}.pdf'
    results_path = f'{DOCS_DIR}/{doc_name}.results'
    open(results_path, 'w').write('')

    document_id = 1
    shutil.copy(
        src=f'{DOCS_DIR}/{doc_name}.json',
        dst=f'{reader.DOCS_FOLDER}/{document_id}.json'
    )

    chunk_sizes = ['1000', '500', '250']

    for conf in test_configs.configs[0:2]:
        for chunk_size in chunk_sizes[0:2]:
            request = ActionRequest(
                documentId=document_id,
                url=doc_path,
                questions=[question.Value for question in questions],
                params={
                    **test_configs.params,
                    **conf.params,
                    "reader.chunk_size": chunk_size
                }
            )
            response = ActionResponse()
            process_request(request, response)

            result = TestResult(
                doc_name=doc_name,
                conf_id=conf.id,
                chunk_size=chunk_size,
                answers=response.answers
            )
            open(results_path, 'a').write(f'{result.dump()}\n')

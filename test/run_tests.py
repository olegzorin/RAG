import logging
import shutil
from pathlib import Path

import reader
from run_rag import RagResponse, RagRequest, RagExecutor
from test_utils import test_configs, questions, TestResult, DOCS_DIR, doc_names

logging.basicConfig(
    level=logging.INFO,
    force=True
)

chunk_sizes = ['1000', '500', '250']

for doc_name in doc_names:
    doc_path = f'{DOCS_DIR}/{doc_name}.pdf'
    results_path = f'{DOCS_DIR}/{doc_name}.results'

    if not Path(results_path).exists():
        open(results_path, 'w').write('')

    document_id = 1
    shutil.copy(
        src=f'{DOCS_DIR}/{doc_name}.json',
        dst=f'{reader.DOCS_FOLDER}/{document_id}.json'
    )

    for conf in test_configs.configs:
        if conf.id != 'v4':
            continue
        for chunk_size in chunk_sizes:
            request = RagRequest(
                documentId=document_id,
                url=doc_path,
                questions=[question.Value for question in questions],
                params={
                    **test_configs.params,
                    **conf.params,
                    "reader.chunk_size": chunk_size
                }
            )
            response = RagResponse()
            RagExecutor().execute(request, response)

            result = TestResult(
                doc_name=doc_name,
                conf_id=conf.id,
                chunk_size=chunk_size,
                answers=response.answers
            )
            open(results_path, 'a').write(f'{result.dump()}\n')

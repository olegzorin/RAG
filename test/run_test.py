import base64
import json
import logging
import shutil
from typing import Any

import main
import reader

logging.basicConfig(
    level=logging.INFO,
    force=True
)

documents = ['CCR']  # , 'LHS', 'MHR']
chunk_sizes = ['1000', '500', '250']

test_conf: dict[str, Any] = json.loads(open('test_configs.json', 'r').read())

questions: list[dict[str, str]] = json.loads(open('questions.json', 'r').read())

common_params = test_conf['params']
tests = test_conf['tests'][0]['params']

results_path = 'results.json'
open(results_path, 'w').writelines(['['])

for document_id, doc_name in enumerate(documents, 1):
    shutil.copy(f'../docs/{doc_name}.json', f'{reader.DOCS_FOLDER}/{document_id}.json')
    for test in test_conf['tests'][0:1]:
        params = {**common_params, **test['params']}
        for chunk_size in chunk_sizes[0:1]:
            test_id = f"{doc_name}_{test['id']}_{chunk_size}"
            request = {
                "documentId": document_id,
                "url": f'../docs/{doc_name}.pdf',
                "questions": [question['Value'] for question in questions],
                "params": {**common_params, **test['params'], "reader.chunk_size": chunk_size}
            }
            # print(json.dumps(request))
            encoded = base64.b64encode(json.dumps(request).encode())
            outpath = f'../docs/{test_id}.output'
            main.run(encoded.decode('utf8'), outpath)
            response = open(outpath, 'r').read()
            open(results_path, 'a').write(response)


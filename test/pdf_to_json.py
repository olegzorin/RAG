import logging
import os
import shutil
from pathlib import Path

from conf import DOCS_CACHE_DIR
from pdf_document import PdfDoc, PdfFile

logging.basicConfig(level=logging.INFO)

DOCS_DIR = '../docs'

doc_names = [
    'MHR',
    'CCR',
    'LHS'
]

for document_id, doc_name in enumerate(doc_names, 1):
    local_json_file = f'{DOCS_DIR}/{doc_name}.json'
    cached_json_file = f'{DOCS_CACHE_DIR}/{document_id}.json'

    if Path(local_json_file).exists():
        os.remove(local_json_file)

    if Path(cached_json_file).exists():
        os.remove(cached_json_file)

    doc: PdfDoc = PdfFile.read_doc(
        document_id=document_id,
        source=f'{DOCS_DIR}/{doc_name}.pdf',
        no_cache=True
    )
    shutil.copy(
        src=cached_json_file,
        dst=local_json_file
    )

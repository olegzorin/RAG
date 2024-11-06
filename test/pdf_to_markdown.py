import logging
import os
import shutil
from pathlib import Path

from conf import DOCS_CACHE_DIR
from pdf_document import PdfDoc, PdfFile

logging.basicConfig(level=logging.WARN)

DOCS_DIR = '../docs'

doc_names = [
    'MHR',
    'CCR',
    'LHS'
]

for document_id, doc_name in enumerate(doc_names, 1):
    local_md_file = f'{DOCS_DIR}/{doc_name}.md'
    local_json_file = f'{DOCS_DIR}/{doc_name}.json'
    cached_json_file = f'{DOCS_CACHE_DIR}/{document_id}.json'

    if Path(local_json_file).exists():
        shutil.copy(
            src=local_json_file,
            dst=cached_json_file
        )
    elif Path(cached_json_file).exists():
        os.remove(cached_json_file)

    doc: PdfDoc = PdfFile.read_doc(
        document_id=document_id,
        source=f'{DOCS_DIR}/{doc_name}.pdf',
        no_cache=False
    )
    shutil.copy(
        src=cached_json_file,
        dst=local_json_file
    )

    Path(local_md_file).write_text(doc.get_content())

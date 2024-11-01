import logging

from pdf_document import PdfDoc, PdfFile

logging.basicConfig(level=logging.INFO)

DOCS_DIR = 'docs'
doc_name = 'LHS'
page = 33

doc: PdfDoc = PdfFile.read_doc(
    document_id=666,
    source=f'{DOCS_DIR}/{doc_name}.pdf',
    first_page=page,
    last_page=page,
    no_cache=True
)

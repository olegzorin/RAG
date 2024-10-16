from pymupdf import Document

from langchain_community.document_loaders import PyMuPDFLoader


pdf_path = 'docs/CCR.pdf'

loader = PyMuPDFLoader(str(pdf_path))
docs: list[Document] = loader.load()
print(len(docs))
doc: Document =docs[0]
print(doc.metadata)
# doc.p

from __future__ import annotations

import os
# os.environ['TESSDATA_PREFIX']='/opt/homebrew/share/tessdata'

import hashlib
import logging
from pathlib import Path
from re import match
from typing import Union, Any

import camelot
import pdf2image
import pytesseract
from PIL import Image
from camelot.core import TableList
from camelot.utils import is_url, download_url
from pandas import DataFrame
from pydantic import BaseModel, TypeAdapter

from conf import get_property, DOCS_CACHE_DIR

source = 'docs/CCR.pdf'

images: list[Image] = pdf2image.convert_from_path(
    pdf_path=source,
    first_page=16,
    last_page=16,
    output_folder='temp',
    paths_only=True,
    transparent=True,
    fmt='jpeg',
    dpi=300
)

# import fitz
#
# doc = fitz.open()  # new PDF
# for img in images:
#     imgdoc = fitz.open(img)  # open image as a document
#     pdfbytes = imgdoc.convert_to_pdf()  # make a 1-page PDF of it
#     imgpdf = fitz.open("pdf", pdfbytes)
#     doc.insert_pdf(imgpdf)  # insert the image PDF
# doc.save("allmyimages.pdf")
import pymupdf

# doc = pymupdf.Document(filename=images[0])
# page = doc.load_page(0)
# pixmap = page.get_pixmap(dpi=150)
# open("input-ocr.pdf", 'wb').write(
#     pixmap.pdfocr_tobytes(compress=False, tessdata='/opt/homebrew/share/tessdata')
# )

print(f'Images {len(images)}')

pdf_paths = []
for i, img in enumerate(images, 1):
    print(f'Img2pdf {i}: {img}')
    pdf = Path(img).with_suffix(".boxes")
    with open(pdf, 'w') as f:
        f.write(
            pytesseract.image_to_boxes(
                image=img,
                # extension='pdf',
                config='--psm 6'
            )
        )
    pdf_paths.append(pdf)

print(f'Pdfs {len(pdf_paths)}')
#
# from PyPDF2 import PdfMerger
#
# # set path files
# import os
#
pdf_path = "docs/CCR_1.pdf"

# merger = PdfMerger()
# for i, pdf in enumerate(pdf_paths, 1):
#     print(f'Merge pdf {i}')
#     merger.append(pdf)
#
# with open(pdf_path, "wb") as new_file:
#     print('Save merge')
#     merger.write(new_file)
#
import pymupdf4llm
#
md_text = pymupdf4llm.to_markdown(doc=pdf_paths[0], show_progress=False)
Path(pdf_paths[0]).with_suffix(".md").write_bytes(md_text.encode())

from __future__ import annotations

import json
import os
import shutil

import pdf2image

from reader import PdfPage

temp_dir = 'temp'
shutil.rmtree(temp_dir)
os.mkdir(temp_dir)

document_id = 1
source = 'docs/CCR.pdf'
n = 17
images: list = pdf2image.convert_from_path(
    pdf_path=source,
    first_page=n,
    last_page=n,
    output_folder=temp_dir,
    output_file=document_id,
    paths_only=True,
    fmt='png',
    dpi=150
)



pdf_page = PdfPage(page_no=1)
pdf_page.load_from_image(images[0])
print(json.dumps(pdf_page.model_dump(mode='json'), indent=4))

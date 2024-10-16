from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import PyPDF2
import camelot
import pdf2image
import pytesseract
from PIL import Image
from camelot.core import TableList
from pydantic import BaseModel

dpi = 150  # 600


class TextBlock(BaseModel):
    cols: int
    rows: int
    cells: list[list[str]]

    @classmethod
    def from_text(cls, text: str) -> TextBlock:
        return TextBlock(cols=1, rows=1, cells=[[text]])


class ExtractedPage(BaseModel):
    contents: list[TextBlock]

    @classmethod
    def from_image(cls, path: str) -> ExtractedPage:
        img: Image = Image.open(path)
        img = img.crop((0, 0, img.width, img.height * 0.936))  # cutoff footers

        pdf_path = Path(path).with_suffix(".pdf")
        pdf_path.write_bytes(
            pytesseract.image_to_pdf_or_hocr(
                image=img,
                extension='pdf',
                config='--psm 6'
            )
        )

        def extract_text(image: Image) -> str:
            return pytesseract.image_to_string(
                image=image,
                config='--psm 6'
            )

        table_list: TableList = camelot.read_pdf(
            filepath=pdf_path.as_posix(),
            pages='1',
            suppress_stdout=True,
            backend='ghostscript',
            resolution=dpi
        )

        if table_list.n == 0:
            text = extract_text(img)
            return ExtractedPage(
                contents=[TextBlock.from_text(text)] if text else []
            )

        # Start processing page with tables

        # Convertion of PDF coordinate system to pixels
        pdf = PyPDF2.PdfReader(pdf_path)
        mediabox = pdf.pages[0].mediabox
        h_scale = img.width / float(mediabox.width)
        v_scale = img.height / float(mediabox.height)

        contents: list[TextBlock] = []

        text_v_offset = 0

        for table in table_list:
            # Absolute positions of the rows and columns in pixels
            rows = [(img.height - v_scale * r[0], img.height - v_scale * r[1]) for r in table.rows]
            cols = [(h_scale * c[0], h_scale * c[1]) for c in table.cols]

            table_min_y = rows[0][0]
            table_max_y = rows[-1][1]

            # Read text preceding the table
            if text := extract_text(img.crop((0, text_v_offset, img.width, table_min_y))):
                contents.append(TextBlock.from_text(text))
            text_v_offset = table_max_y

            # Read table cells
            contents.append(
                TextBlock(
                    cols=len(table.cols),
                    rows=len(table.rows),
                    cells=[[
                        extract_text(img.crop((col[0], row[0], col[1], row[1])))
                        for col in cols
                    ] for row in rows]
                )
            )

        if text := extract_text(img.crop((0, text_v_offset, img.width, img.height))):
            contents.append(TextBlock.from_text(text))

        return ExtractedPage(contents=contents)


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
    dpi=dpi
)

extracted_page = ExtractedPage.from_image(images[0])
print(json.dumps(extracted_page.model_dump(mode='json'), indent=4))

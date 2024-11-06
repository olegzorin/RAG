import logging
import os
import re
import pdf2image
from pathlib import Path
from pydantic import BaseModel
import camelot as cm
from camelot.core import TableList, Table
import pytesseract as pt
from PIL import Image
import PyPDF2

from conf import get_tesseract_version, DOCS_CACHE_DIR

logger = logging.getLogger(__name__)

# https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy
# Note that you should use '--psm' or '-psm' depending on your tesseract version
TESSERACT_CONFIG = '' if not (v := get_tesseract_version()) else '-psm 6' if (v[0] < 4 and v[1] < 5) else '--psm 6'
print('TESSERACT_CONFIG=', TESSERACT_CONFIG)

DPI = 150


def convert_pdf_to_images(
        pdf_path: str,
        doc_name: str,
        *,
        first_page: int | None = None,
        last_page: int | None = None
) -> list[str]:
    temp_folder = Path(DOCS_CACHE_DIR, 'temp')
    os.makedirs(temp_folder, exist_ok=True)

    imgs = pdf2image.convert_from_path(
        pdf_path=pdf_path,
        first_page=first_page,
        last_page=last_page,
        output_folder=temp_folder,
        output_file=doc_name,
        fmt='png',
        paths_only=True,
        dpi=DPI,
        grayscale=True
    )
    return [f'{img}' for img in imgs]


class TableFrame(BaseModel):
    rows: list[tuple[int, int]]
    cols: list[tuple[int, int]]


def extract_tables_from_image(
        img: Image,
        pdf_path: Path
) -> list[TableFrame]:
    pdf_path.write_bytes(
        pt.image_to_pdf_or_hocr(
            image=img,
            extension='pdf',
            config=TESSERACT_CONFIG
        )
    )
    table_list: TableList = cm.read_pdf(
        filepath=pdf_path.as_posix(),
        pages='1',
        suppress_stdout=True,
        # backend='ghostscript',
        resolution=DPI,
        flavor='lattice',
        line_scale=40,
    )
    if table_list is None or table_list.n == 0:
        return []

    # Convertion of PDF coordinate system to pixels
    pdf = PyPDF2.PdfReader(pdf_path)
    mediabox = pdf.pages[0].mediabox
    h_scale = img.width / float(mediabox.width)
    v_scale = img.height / float(mediabox.height)

    # There are pages with a horizontally arranged set of tables,
    # each of which is rotated 90 degrees.
    # Sort the tables by vertical top position to properly compute
    # the bottom of preceding plain text if any.
    tables: list[Table] = [*table_list]
    tables.sort(key=lambda t: t.rows[0][0], reverse=True)

    table_frames: list[TableFrame] = []
    for table in tables:
        # Distances of the top and bottom borders of rows in pixels from the top of the page
        rows = [(img.height - int(v_scale * r[0]), img.height - int(v_scale * r[1])) for r in table.rows]
        # Distances of the left and right borders of the columns in pixels from the left edge of the page
        cols = [(int(h_scale * c[0]), int(h_scale * c[1])) for c in table.cols]

        table_frames.append(TableFrame(rows=rows, cols=cols))

    return table_frames


def extract_text_from_image(
        img: Image,
        box: tuple[float, float, float, float] = None,
        rotated: bool = False
) -> str:
    if rotated:
        im = img.crop(box=(box[1], box[0], box[3], box[2])) if box else img
        im = im.rotate(angle=90, expand=1)
    else:
        im = img.crop(box=(box[0], box[1], box[2], box[3])) if box else img

    text = pt.image_to_string(
        image=im.crop(box=(2, 0, im.width - 2, im.height)),
        config=TESSERACT_CONFIG
    )
    if text:
        # Fix OCR typos
        text = text.replace('|', 'I')
        text = re.sub(r'Il+(?![A-Za-z])', lambda x: x.group().replace('l', 'I'), text)
    return text

# from camelot.parsers import Lattice
# import shutil
#
# l: Lattice = Lattice()
# from camelot.core import Table
#
# filename1='/Users/oleg/Projects/PPC/home/ragagent/documents/temp/5d4efd8e-b3d7-4587-80ce-fcb23ad82f23-17.pdf'
# filename2='/Users/oleg/Projects/PPC/home/ragagent/documents/temp/page-17.pdf'
# shutil.copy(filename1, filename2)
# tables: list[Table] = l.extract_tables(filename=filename2, suppress_stdout=False)
# print(len(tables))

#
# # pip install opencv-contrib-python-headless
# from img2table.document import Image as Im
# import os

#
# src = '/Users/oleg/Projects/PPC/home/ragagent/documents/temp/17331d87-b735-4162-a328-adf930abd616-17.png'
#
# img = Im(src=src)
#
# # Table identification
# imgage_tables = img.extract_tables(min_confidence=40, implicit_rows=False, implicit_columns=True)
#
# # Result of table identification
# print(imgage_tables)
# #
# # src1 = '/Users/oleg/Projects/PPC/home/ragagent/documents/temp/17331d87-b735-4162-a328-adf930abd616-17.pdf'
# #
# # from img2table.document import PDF
# # pdf = PDF(src1, pages=[0,0])
# # print(pdf.extract_tables(min_confidence=40))§
#
# from transformers import MvpTokenizer, MvpForConditionalGeneration, TableTransformerModel
#
# tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp", cache_dir=MODEL_CACHE_DIR)
# model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text", cache_dir=MODEL_CACHE_DIR)
#
# table = '''
# | Service: | Reimbursement: |
# | :------ | :------ |
# | Drugs & Biologicals | 100% of ChoiceCare’s 201-544 fee schedule |
# | All other services | 95% of ChoiceCare’s 005-270 fee schedule |
# '''
#
# inputs = tokenizer.__call__(
#     "Convert the following table to text: " + table.replace('\n', ' '),
#     return_tensors="pt",
# )
# generated_ids = model.generate(**inputs)
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
#

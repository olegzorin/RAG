from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union, Any, Optional

import camelot
from camelot.core import TableList
import pdf2image
from PIL import Image
import PyPDF2
import pytesseract
from pydantic import BaseModel, TypeAdapter, ValidationError

import text_utils
from conf import DOCS_CACHE_DIR, get_property
from text_utils import reformat_paragraphs, table_to_html, fix_ocr_typos

# https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy
# Note that you should use '--psm' or '-psm' depending on your tesseract version
TESSERACT_CONFIG = get_property('tesseract.config', '-psm 6')
print(f'TESSERACT_CONFIG={TESSERACT_CONFIG}')

DPI: int = 150

logger = logging.getLogger(__name__)


def delete_file(
        path: Union[str, Path]
) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Error deleting {path}: {e}")


class TextBlock(BaseModel):
    cols: int
    rows: int
    data: list[list[str]]

    def get_content_len(self) -> int:
        return sum([sum([len(col) for col in row]) for row in self.data])

    @classmethod
    def from_text(cls, text: str) -> TextBlock:
        return TextBlock(cols=1, rows=1, data=[[text]])


class PdfPage(BaseModel):
    page_no: int
    contents: list[TextBlock] = []

    def get_content_len(self) -> int:
        return sum([blk.get_content_len() for blk in self.contents])

    def load_from_image(self, path: str) -> PdfPage:
        img: Image = Image.open(path)
        img = img.crop((0, 0, img.width, img.height * 0.936))  # cutoff footers

        pdf_path = Path(path).with_suffix(".pdf")
        pdf_path.write_bytes(
            pytesseract.image_to_pdf_or_hocr(
                image=img,
                extension='pdf',
                config=TESSERACT_CONFIG
            )
        )

        def _extract_text(
                image: Image,
                box: Optional[tuple[float, float, float, float]] = None,
                rotated: bool = False
        ) -> str:
            # print(f'box={box}')
            if rotated:
                im = image.crop(box=(box[1], box[0], box[3], box[2])) if box else image
                im = im.rotate(angle=90, expand=1)
            else:
                im = image.crop(box=(box[0], box[1], box[2], box[3])) if box else image

            txt = pytesseract.image_to_string(
                image=im.crop(box=(2, 0, im.width - 2, im.height)),
                config=TESSERACT_CONFIG
            )
            # print(f'txt={txt}')
            return txt

        table_list: TableList = camelot.read_pdf(
            filepath=pdf_path.as_posix(),
            pages='1',
            suppress_stdout=True,
            backend='ghostscript',
            resolution=DPI
        )

        if table_list is None or table_list.n == 0:
            if text := _extract_text(img):
                self.contents.append(TextBlock.from_text(text))
            logger.info(f"Found no tables")
            delete_file(pdf_path)
            return

        # Start processing page with tables
        logger.info(f"Found {table_list.n} tables")

        # Convertion of PDF coordinate system to pixels
        pdf = PyPDF2.PdfReader(pdf_path)
        mediabox = pdf.pages[0].mediabox
        h_scale = img.width / float(mediabox.width)
        v_scale = img.height / float(mediabox.height)

        # Vertical position of the top of the text
        text_v_top = 0

        # There are pages with a horizontally arranged set of tables,
        # each of which is rotated 90 degrees.
        # Sort the tables by vertical top position to properly compute
        # the bottom of preceding plain text if any.
        tables = list(table_list)
        tables.sort(key=lambda t: t.rows[0][0], reverse=True)

        for table in tables:
            # Absolute cordinates (start position and end positions) of the rows and columns in pixels
            rows = [(img.height - v_scale * r[0], img.height - v_scale * r[1]) for r in table.rows]
            cols = [(h_scale * c[0], h_scale * c[1]) for c in table.cols]

            # Vertical positions of the top and bottom of the table
            table_v_top = rows[0][0]
            table_v_btm = rows[-1][1]

            # Read text preceding the table, if any
            if (text_v_top < table_v_top) and (
                    text := _extract_text(image=img, box=(0, text_v_top, img.width, table_v_top))):
                self.contents.append(TextBlock.from_text(text))

            text_v_top = max(table_v_btm, text_v_top)

            # Read table cells
            avg_col_width = (cols[-1][1] - cols[0][0]) / len(cols)
            is_vertical_table = avg_col_width < 30
            print('Vertical:', is_vertical_table)
            if is_vertical_table:
                rows, cols = cols, rows
                rows.reverse()

            # print(f'cols={cols}')
            # print(f'rows={rows}')
            self.contents.append(
                TextBlock(
                    cols=len(cols),
                    rows=len(rows),
                    data=[[
                        _extract_text(image=img, box=(col[0], row[0], col[1], row[1]), rotated=is_vertical_table)
                        for col in cols
                    ] for row in rows]
                )
            )

        if text := _extract_text(image=img, box=(0, text_v_top, img.width, img.height)):
            self.contents.append(TextBlock.from_text(text))


class PdfDoc(BaseModel):
    pages: list[PdfPage] = []

    def get_content_len(self) -> int:
        return sum([page.get_content_len() for page in self.pages])

    def get_content(self) -> str:
        content = ''
        text = ''
        for page in self.pages:
            for text_block in page.contents:
                if text_block.cols > 1:
                    if text:
                        content += '\n\n' + reformat_paragraphs(text)
                        text = ''
                    content += '\n\n' + table_to_html(text_block.data)
                else:
                    text += '\n\n' + text_block.data[0][0]
        if text:
            content += '\n\n' + reformat_paragraphs(text)

        return fix_ocr_typos(content)

    def load_from_source(
            self,
            source: str,
            *,
            first_page: int | None = None,
            last_page: int | None = None
    ) -> None:

        temp_folder = Path(DOCS_CACHE_DIR, 'temp')
        os.makedirs(temp_folder, exist_ok=True)

        img_paths = pdf2image.convert_from_path(
            pdf_path=source,
            first_page=first_page,
            last_page=last_page,
            output_folder=temp_folder,
            fmt='png',
            paths_only=True,
            dpi=DPI
        )
        try:
            for i, path in enumerate(img_paths, 1):
                logger.info(f"Read page {i}")
                page = PdfPage(page_no=i)
                page.load_from_image(path)
                self.pages.append(page)
        finally:
            for path in img_paths:
                delete_file(path)

    def dump(self) -> str:
        import json
        outputs: list[str] = []
        for page in self.pages:
            outputs.append(f'<------ Page {page.page_no} ------>')
            for cnt in page.contents:
                if cnt.cols == 1:
                    outputs.extend([text_utils.reformat_paragraphs(row[0]) for row in cnt.data])
                else:
                    outputs.append(json.dumps(cnt.data, indent=4))
        return '\n'.join(outputs)

    def split_into_chucks(
            self,
            chunkers: list[Any],
            *,
            save_chunks: bool = False
    ) -> list[str]:
        text = self.get_content()
        chunks = []
        for chunker in chunkers:
            _chunks = chunker.split_text(text)
            logger.info(f'{chunker.__class__.__name__}: chunks={len(_chunks)}, maxlen={max([len(c) for c in _chunks])}')
            chunks.extend(_chunks)

        table_rows = self.get_table_rows()

        if save_chunks:
            # Save the chunks to a file for review
            filename = f'{self.id}.chunks'
            with open(Path(DOCS_CACHE_DIR, filename), 'w') as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f'\n\n<--------- text {i} ---------->\n')
                    f.write(chunk)
                for i, chunk in enumerate(table_rows, 1):
                    f.write(f'\n\n<--------- table {i} ---------->\n')
                    f.write(chunk)

        chunks.extend(table_rows)
        return list(map(lambda s: s.lower(), chunks))


class PdfFile(PdfDoc):
    checksum: str

    @staticmethod
    def read_doc(
            document_id: int,
            source: str,
            *,
            first_page: int | None = None,
            last_page: int | None = None,
            no_cache: bool = False
    ) -> PdfDoc:
        logger.info("Read PDF")

        delete_source = False

        try:
            from camelot.utils import is_url, download_url
            if is_url(source):
                source = download_url(source)
                delete_source = True

            import hashlib
            with open(source, "rb") as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            checksum = file_hash.hexdigest()

            doc_path: Path = Path(DOCS_CACHE_DIR, f'{document_id}.json')
            if not no_cache and doc_path.exists():
                try:
                    doc = TypeAdapter(PdfFile).validate_json(doc_path.read_bytes())
                    if doc.checksum == checksum:
                        return doc
                except ValidationError:
                    logger.warning(f"Exception reading cached document ID={document_id}")

            doc = PdfFile(checksum=checksum)
            doc.load_from_source(
                source=source,
                first_page=first_page,
                last_page=last_page
            )
            data = TypeAdapter(PdfFile).dump_json(doc)
            doc_path.write_bytes(data)

            return doc

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error reading document: {e}")
        finally:
            if delete_source:
                delete_file(source)


logging.basicConfig(
    level=logging.INFO,
    force=True
)
# name = 'CCR'
name = 'LHS'
doc_id = 1
page_no = 2

# PdfFile.read_doc(
#     document_id=doc_id,
#     source=f'../docs/{name}.pdf',
#     no_cache=True
#     # first_page=1, #page_no,
#     # last_page=page_no
# )
json_file = f'../docs/{name}.json'
# delete_file(json_file)
import shutil

# shutil.copy(f'{DOCS_CACHE_DIR}/{doc_id}.json', json_file)

# shutil.copy(json_file, f'{DOCS_CACHE_DIR}/{doc_id}.json')
doc: PdfDoc = PdfFile.read_doc(
    document_id=doc_id,
    source=f'../docs/{name}.pdf'
)
Path(f'../docs/{name}.txt').write_text(doc.dump())

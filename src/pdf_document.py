from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union, Any

import PyPDF2
import pdf2image

import camelot
import pytesseract
from PIL import Image
from camelot.core import TableList
from pydantic import BaseModel, TypeAdapter

import text_utils
from conf import DOCS_CACHE_DIR, get_property

# https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy
# Note that you should use '--psm' or '-psm' depending on your tesseract version
TESSERACT_CONFIG = get_property('tesseract.config', '-psm 6')

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
    cells: list[list[str]]

    def get_content_len(self) -> int:
        return sum([sum([len(col) for col in row]) for row in self.cells])

    @classmethod
    def from_text(cls, text: str) -> TextBlock:
        return TextBlock(cols=1, rows=1, cells=[[text]])


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
                config='--psm 6'
            )
        )

        def _extract_text(image: Image) -> str:
            return pytesseract.image_to_string(
                image=image,
                config='--psm 6'
            )

        table_list: TableList = camelot.read_pdf(
            filepath=pdf_path.as_posix(),
            pages='1',
            suppress_stdout=True,
            backend='ghostscript',
            resolution=DPI
        )

        if table_list.n == 0:
            if text := _extract_text(img):
                self.contents.append(TextBlock.from_text(text))
            return

        # Start processing page with tables

        # Convertion of PDF coordinate system to pixels
        pdf = PyPDF2.PdfReader(pdf_path)
        mediabox = pdf.pages[0].mediabox
        h_scale = img.width / float(mediabox.width)
        v_scale = img.height / float(mediabox.height)

        # Vertical position of the top of the text
        text_v_top = 0

        for table in table_list:
            # Absolute cordinates (start position and end positions) of the rows and columns in pixels
            rows = [(img.height - v_scale * r[0], img.height - v_scale * r[1]) for r in table.rows]
            cols = [(h_scale * c[0], h_scale * c[1]) for c in table.cols]

            # Vertical positions of the top and bottom of the table
            table_v_top = rows[0][0]
            table_v_btm = rows[-1][1]

            # Read text preceding the table, if any
            if text := _extract_text(img.crop((0, text_v_top, img.width, table_v_top))):
                self.contents.append(TextBlock.from_text(text))

            text_v_top = table_v_btm

            # Read table cells
            self.contents.append(
                TextBlock(
                    cols=len(table.cols),
                    rows=len(table.rows),
                    cells=[[
                        _extract_text(img.crop((col[0], row[0], col[1], row[1])))
                        for col in cols
                    ] for row in rows]
                )
            )

        if text := _extract_text(img.crop((0, text_v_top, img.width, img.height))):
            self.contents.append(TextBlock.from_text(text))


class PdfDoc(BaseModel):
    pages: list[PdfPage] = []

    def get_content_len(self) -> int:
        return sum([page.get_content_len() for page in self.pages])

    def get_content(self) -> str:
        texts = []
        for page in self.pages:
            texts.append(page.content)
        return text_utils.reformat_paragraphs('\n'.join(texts))

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
            paths_only=True,
            dpi=DPI
        )
        try:
            for i, path in enumerate(img_paths, 1):
                page = PdfPage(page_no=i)
                page.load_from_image(path)
                self.pages.append(page)
        finally:
            for path in img_paths:
                delete_file(path)
                delete_file(Path(path).with_suffix(".pdf"))

    def dump(
            self
    ) -> str:
        outputs: list[str] = []
        for page in self.pages:
            outputs.append(f'<------ Page {page.page_no} ------>')
            outputs.append(text_utils.reformat_paragraphs(page.content))
            if page.tables:
                outputs.append("Tables:")
                for i, table in enumerate(page.tables, 1):
                    outputs.append(f"Table {i}")
                    for row in table.rows:
                        outputs.append(row)
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
                except Exception as e:
                    logger.warning(f"Exception reading document ID={document_id}: {e}")

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



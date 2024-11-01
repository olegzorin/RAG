from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import PyPDF2
import camelot
import pdf2image
from PIL import Image
from camelot.core import TableList, Table
from pydantic import BaseModel, TypeAdapter, ValidationError

from conf import DOCS_CACHE_DIR
from text_utils import text_to_markdown, table_to_markdown

logger = logging.getLogger(__name__)

TEMP_FOLDER = Path(DOCS_CACHE_DIR, 'temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

DPI: int = 150


def delete_file(path: Union[str, Path]):
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

    def load_from_image(self, path: str) -> None:
        img: Image = Image.open(path)
        img = img.crop((0, 0, img.width, img.height * 0.938))  # cutoff footers

        pdf_path = Path(path).with_suffix(".pdf")

        from ocr_utils import convert_image_to_pdf, extract_text_from_image

        pdf_path.write_bytes(convert_image_to_pdf(img=img))

        try:
            table_list: TableList = camelot.read_pdf(
                filepath=pdf_path.as_posix(),
                pages='1',
                suppress_stdout=True,
                # backend='ghostscript',
                resolution=DPI,
                flavor='lattice',
                line_scale=40,
            )

            if table_list is None or table_list.n == 0:
                if text := extract_text_from_image(img):
                    self.contents.append(TextBlock.from_text(text))
                logger.info(f"Found no tables")
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
            tables: list[Table] = [*table_list]
            tables.sort(key=lambda t: t.rows[0][0], reverse=True)

            for table in tables:
                # print('Table data: ', table.data)

                # Absolute cordinates (start position and end positions) of the rows and columns in pixels
                rows = [(img.height - v_scale * r[0], img.height - v_scale * r[1]) for r in table.rows]
                cols = [(h_scale * c[0], h_scale * c[1]) for c in table.cols]

                # Vertical positions of the top and bottom of the table
                table_v_top = rows[0][0]
                table_v_btm = rows[-1][1]

                # Read text preceding the table, if any
                if (text_v_top < table_v_top) and (
                        text := extract_text_from_image(img=img, box=(0, text_v_top, img.width, table_v_top))):
                    self.contents.append(TextBlock.from_text(text))

                text_v_top = max(table_v_btm, text_v_top)

                # Read table cells
                avg_col_width = (cols[-1][1] - cols[0][0]) / len(cols)
                is_vertical_table = avg_col_width < 30
                # print('Vertical:', is_vertical_table)
                if is_vertical_table:
                    rows, cols = cols, rows
                    rows.reverse()

                # print(f'cols={cols}')
                # print(f'rows={rows}')

                # Filter out too narrow rows and columns generated by OCR
                rows = [row for row in rows if row[1] - row[0] > 4]
                cols = [col for col in cols if col[1] - col[0] > 4]

                # print(f'cols={cols}')
                # print(f'rows={rows}')

                self.contents.append(
                    TextBlock(
                        cols=len(cols),
                        rows=len(rows),
                        data=[[
                            extract_text_from_image(
                                img=img,
                                box=(col[0], row[0], col[1], row[1]),
                                rotated=is_vertical_table
                            )
                            for col in cols
                        ] for row in rows]
                    )
                )

            if text := extract_text_from_image(img=img, box=(0, text_v_top, img.width, img.height)):
                self.contents.append(TextBlock.from_text(text))
        finally:
            delete_file(pdf_path)


class PdfDoc(BaseModel):
    pages: list[PdfPage] = []

    def get_content_len(self) -> int:
        return sum([page.get_content_len() for page in self.pages])

    def get_content(self, show_page_numbers: bool = False) -> str:
        content = ''
        plain_text = ''
        for i, page in enumerate(self.pages, 1):
            if show_page_numbers:
                content += '\n\n<--------- ' + str(i) + ' ---------->\n\n'
            for text_block in page.contents:
                if text_block.cols > 1:
                    if plain_text:
                        content += '\n\n' + text_to_markdown(plain_text)
                        plain_text = ''
                    content += '\n\n' + table_to_markdown(text_block.data)
                else:
                    plain_text += text_block.data[0][0]
        if plain_text:
            content += '\n\n' + text_to_markdown(plain_text)

        return content

    def load_from_source(
            self,
            source_path: str,
            *,
            first_page: int | None = None,
            last_page: int | None = None
    ) -> None:

        doc_name = Path(source_path).stem

        img_paths = pdf2image.convert_from_path(
            pdf_path=source_path,
            first_page=first_page,
            last_page=last_page,
            output_folder=TEMP_FOLDER,
            fmt='png',
            paths_only=True,
            dpi=DPI
        )
        try:
            for i, path in enumerate(img_paths, 1):
                logger.info(f"{doc_name}: read page {i}")
                page = PdfPage(page_no=i)
                page.load_from_image(f'{path}')
                self.pages.append(page)
        finally:
            for path in img_paths:
                delete_file(f'{path}')


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

        delete_source = False

        try:
            from camelot.utils import is_url, download_url
            if is_url(source):
                source = download_url(source)
                delete_source = True

            logger.info(f"Source={Path(source).stem}")

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
                source_path=source,
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

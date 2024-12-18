from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

from pydantic import BaseModel, TypeAdapter, ValidationError

from conf import DOCS_CACHE_DIR
from text_utils import text_to_markdown, table_to_text

logger = logging.getLogger(__name__)

DPI: int = 150

SAVE_TEMP_FILES = True


def delete_temp_file(path: Union[str, Path]):
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
    contents: list[TextBlock]

    def get_content_len(self) -> int:
        return sum([blk.get_content_len() for blk in self.contents])

    def last_table_index(self) -> int | None:
        idx = None
        for i, c in enumerate(self.contents):
            if c.cols > 1:
                idx = i
        return idx

    def load_from_image(self, path: str) -> None:

        from ocr_utils import extract_tables_from_image, extract_text_from_image

        from PIL import Image

        img = Image.open(path)

        # Vertical position of the top of the page content
        text_v_top = 0
        text_v_bottom = img.height * 0.938  # cutoff footers

        tables = extract_tables_from_image(
            img=img,
            pdf_path=Path(path).with_suffix(".pdf")
        )

        if not tables:
            if text := extract_text_from_image(img, box=(0, text_v_top, img.width, text_v_bottom)):
                self.contents.append(TextBlock.from_text(text))
            logger.info(f"Found no tables")
            return

        # Start processing page with tables
        logger.info(f"Found {len(tables)} tables")

        for table in tables:
            rows = table.rows
            cols = table.cols

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

        if text := extract_text_from_image(img=img, box=(0, text_v_top, img.width, text_v_bottom)):
            self.contents.append(TextBlock.from_text(text))


class PdfDoc(BaseModel):
    pages: list[PdfPage]

    def get_content_len(self) -> int:
        return sum([page.get_content_len() for page in self.pages])

    def get_content(self) -> str:
        content = ''
        plain_text = ''
        for i, page in enumerate(self.pages, 1):
            for text_block in page.contents:
                if text_block.cols > 1:
                    if plain_text:
                        content += '\n\n' + text_to_markdown(plain_text)
                        plain_text = ''
                    content += '\n\n' + table_to_text(text_block.data)
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

        from ocr_utils import convert_pdf_to_images

        img_paths: list[str] = convert_pdf_to_images(
            pdf_path=source_path,
            doc_name=doc_name,
            first_page=first_page,
            last_page=last_page
        )
        try:
            for i, path in enumerate(img_paths, 1):
                logger.info(f"{doc_name}: read page {i}")
                page = PdfPage(page_no=i, contents=[])
                page.load_from_image(path)
                self.pages.append(page)
        finally:
            if not SAVE_TEMP_FILES:
                for path in img_paths:
                    delete_temp_file(path)
                    delete_temp_file(Path(path).with_suffix('.pdf'))


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

            doc = PdfFile(checksum=checksum, pages=[])
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
                delete_temp_file(source)

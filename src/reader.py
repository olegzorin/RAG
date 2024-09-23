from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Union

import camelot
import pdf2image
import pytesseract
from PIL import Image
from camelot.core import TableList
from camelot.utils import is_url, download_url
from pandas import DataFrame
from pydantic import BaseModel, TypeAdapter

from conf import resolve_path

DOCS_FOLDER = f'{resolve_path("documents.root", "documents")}'
os.makedirs(DOCS_FOLDER, exist_ok=True)

WORK_FOLDER = Path(DOCS_FOLDER, 'temp')
os.makedirs(WORK_FOLDER, exist_ok=True)

DPI: int = 300
PDF_HEADER_OFFSET = 200
PDF_FOOTER_OFFSET = 250

logging.getLogger("pytesseract").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ExtractedTable(BaseModel):
    rows: list[str]


class ExtractedPage(BaseModel):
    page_no: int
    content: str
    tables: list[ExtractedTable]


class ExtractedDoc(BaseModel):
    checksum: str
    pages: list[ExtractedPage]

    def get_content_length(
            self
    ) -> int:
        return sum([len(page.content) for page in self.pages])

    def split_into_chucks(
            self,
            chunker
    ) -> list[str]:
        texts = []
        table_rows = []
        for page in self.pages:
            texts.append(page.content)
            for table in table_rows:
                table_rows.extend(table.rows)

        chunks = chunker('\n\n'.join(texts))
        chunks.extend(table_rows)
        return chunks

    def save(
            self,
            path: Path
    ) -> None:
        data = TypeAdapter(ExtractedDoc).dump_json(self)
        open(path, 'wb').write(data)

    @staticmethod
    def load(
            path: Path
    ) -> ExtractedDoc:
        data = open(path, 'rb').read()
        return TypeAdapter(ExtractedDoc).validate_json(data)

    def dump(
            self
    ) -> str:
        outputs: list[str] = []
        for page in self.pages:
            outputs.append(f'Page {page.page_no}')
            outputs.append(page.content)
            if page.tables:
                outputs.append("Tables:")
                for i, table in enumerate(page.tables, 1):
                    outputs.append(f"Table {i}")
                    for row in table.rows:
                        outputs.append(row)
        return '\n\n'.join(outputs)

def _clean_text(
        text: str
) -> str:
    return re.sub('[\0\\s]+', ' ', text).strip()


def _delete_file(
        filepath: Union[str, Path]
) -> None:
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning(f"Error deleting {filepath}: {e}")

def _get_checksum(
        source: str
) -> str:
    with open(source, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            # noinspection PyTypeChecker
            file_hash.update(chunk)
    return file_hash.hexdigest()

def _get_table_rows(
        df: DataFrame
) -> list[str]:
    df = df.dropna(how="all").loc[:, ~df.columns.isin(['', ' '])]
    df = df.apply(lambda x: x.str.replace("\n", " "))
    df = df.rename(columns=df.iloc[0]).drop(df.index[0]).reset_index(drop=True)

    if df.shape[0] <= 2 or df.eq("").all(axis=None):
        return []

    df["summary"] = df.apply(
        lambda x: " ".join([f"{col}: {val}, " for col, val in x.items()]),
        axis=1
    )
    return [row["summary"] for ind, row in df.iterrows()]


def _convert_pdf_to_images(
        source: str
) -> list[str]:
    images = pdf2image.convert_from_path(
        pdf_path=source,
        output_folder=WORK_FOLDER,
        paths_only=True,
        fmt="ppm",
        dpi=DPI
    )
    return [f"{image}" for image in (images or [])]


def _extract_page_from_image(
        page_no: int,
        img_path: str
) -> ExtractedPage:
    logger.info(f"Extracting page {page_no}")

    page = ExtractedPage(
        page_no=page_no,
        content='',
        tables=[]
    )
    try:
        img = Image.open(img_path)
        img = img.crop((0, PDF_HEADER_OFFSET, img.width, img.height - PDF_FOOTER_OFFSET))
        page.content = pytesseract.image_to_string(
            image=img
        )
    except Exception as e:
        raise RuntimeError(f"Error recognizing text from image on page {page_no}: {e}")

    pdf_path = Path(img_path).with_suffix(".pdf").as_posix()

    # Convert image to PDF for processing by camelot
    try:
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(
                pytesseract.image_to_pdf_or_hocr(
                    image=img,
                    extension='pdf'
                )
            )
    except Exception as e:
        raise RuntimeError(f"Error converting image to pdf on page {page_no}: {e}")

    # Extract tables from PDF
    try:
        table_list: TableList = camelot.read_pdf(
            filepath=pdf_path,
            pages='1',
            suppress_stdout=True,
            backend='ghostscript',
            resolution=DPI
        )
        logger.info(f"Found {table_list.n} tables")

        for table in table_list:
            table_rows = _get_table_rows(table.df)
            page.tables.append(
                ExtractedTable(rows=table_rows)
            )
    except Exception as e:
        raise RuntimeError(f"Table extraction error on page {page_no}: {e}")
    finally:
        _delete_file(pdf_path)

    return page


def _load_doc_from_source(
        checksum: str,
        source: str
) -> ExtractedDoc:
    doc = ExtractedDoc(
        checksum=checksum,
        pages=[]
    )
    img_paths = _convert_pdf_to_images(source)
    logger.info(f"Document contains {len(img_paths)} pages")
    try:
        for i, img_path in enumerate(img_paths, 1):
            page = _extract_page_from_image(i, img_path)
            doc.pages.append(page)
    finally:
        for img_path in img_paths:
            _delete_file(img_path)
    return doc


def read_pdf(
        document_id: int,
        source: str,
        *,
        no_cache: bool = False
) -> ExtractedDoc:

    logger.info("read_pdf")

    doc_path = Path(DOCS_FOLDER, f'{document_id}.json')

    delete_source = False

    try:
        if is_url(source):
            source = download_url(source)
            delete_source = True

        checksum = _get_checksum(source)

        if not no_cache and doc_path.exists():
            doc = ExtractedDoc.load(doc_path)
            if doc.checksum == checksum:
                return doc

        doc = _load_doc_from_source(checksum, source)
        doc.save(doc_path)
        return doc

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading document: {e}")
    finally:
        if delete_source and not is_url(source):
            _delete_file(source)

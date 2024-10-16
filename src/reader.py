from __future__ import annotations

import hashlib
import logging
import os
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

from conf import get_property, docs_cache_dir

DOCS_FOLDER = docs_cache_dir
WORK_FOLDER = Path(DOCS_FOLDER, 'temp')
os.makedirs(WORK_FOLDER, exist_ok=True)

DPI: int = 300
PDF_HEADER_OFFSET = 200
PDF_FOOTER_OFFSET = 250

SAVE_CHUNKS: bool = True

# https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy
# Note that you should use '--psm' or '-psm' depending on your tesseract version
TESSERACT_CONFIG = get_property('tesseract.config', '-psm 6')

logging.getLogger("pytesseract").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logger = logging.getLogger()


def _reformat_paragraphs(
        text: str,
        paragraph_endline_threshold: float = 0.8
) -> str:
    lines = [l.rstrip() for l in text.strip().splitlines()]
    max_line_len = max([len(l) for l in lines])

    def _is_paragraph_endline(l: str):
        return len(l) < paragraph_endline_threshold * max_line_len and l.endswith(('.', ':', '!', '?'))

    def _is_list_item(l: str) -> bool:
        return (match('^\\(?[0-9.]+\\)?\\s+', l) is not None
                or match('^\\(?[a-zA-Z][.)]\\s+', l) is not None
                or match('^[oe*]\\s+', l) is not None)

    class Block(BaseModel):
        lines: list[str] = []

    # Extract blocks separated by empty lines
    blocks: list[Block] = []
    block_lines = []
    for line in lines:
        if line:
            block_lines.append(line)
        elif block_lines:
            blocks.append(Block(lines=block_lines))
            block_lines = []
    if block_lines:
        blocks.append(Block(lines=block_lines))

    blocks1: list[Block] = []
    block = blocks[0]
    for i, blk in enumerate(blocks[1:]):
        if blk.lines[0][1].islower():
            block.lines.extend(blk.lines)
        else:
            blocks1.append(block)
            block = blk
    blocks1.append(block)

    # Extract paragraphs from the blocks
    pars: list[Block] = []
    for blk in blocks1:
        par_lines = []
        for l in blk.lines:
            if _is_list_item(l):
                if par_lines:
                    pars.append(Block(lines=par_lines))
                par_lines = [l]
            else:
                par_lines.append(l)
                if _is_paragraph_endline(l):
                    pars.append(Block(lines=par_lines))
                    par_lines = []
        if par_lines:
            pars.append(Block(lines=par_lines))

    # Compose the text
    lines = []
    for par in pars:
        lines.extend([*par.lines, ''])
    return '\n'.join(lines[:-1])


class ExtractedTable(BaseModel):
    rows: list[str]


class ExtractedPage(BaseModel):
    page_no: int
    content: str
    tables: list[ExtractedTable]


class ExtractedDoc(BaseModel):
    id: int = 0
    checksum: str
    pages: list[ExtractedPage]

    def get_content_length(self) -> int:
        return sum([len(page.content) for page in self.pages])

    def get_table_rows(self) -> list[str]:
        table_rows = []
        for page in self.pages:
            for _table in page.tables:
                table_rows.extend(_table.rows)
        return table_rows

    def get_content(self) -> str:
        texts = []
        for page in self.pages:
            texts.append(page.content)
        return _reformat_paragraphs('\n'.join(texts))

    def split_into_chucks(
            self,
            chunkers: list[Any]
    ) -> list[str]:
        text = self.get_content()
        chunks = []
        for chunker in chunkers:
            _chunks = chunker.split_text(text)
            logger.info(f'{chunker.__class__.__name__}: chunks={len(_chunks)}, maxlen={max([len(c) for c in _chunks])}')
            chunks.extend(_chunks)

        table_rows = self.get_table_rows()

        if SAVE_CHUNKS:
            # Save the chunks to a file for review
            filename = f'{self.id}.chunks'
            with open(Path(DOCS_FOLDER, filename), 'w') as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f'\n\n<--------- text {i} ---------->\n')
                    f.write(chunk)
                for i, chunk in enumerate(table_rows, 1):
                    f.write(f'\n\n<--------- table {i} ---------->\n')
                    f.write(chunk)

        chunks.extend(table_rows)
        return list(map(lambda s: s.lower(), chunks))

    def save(self) -> None:
        data = TypeAdapter(ExtractedDoc).dump_json(self)
        filename = f'{self.id}.json'
        open(Path(DOCS_FOLDER, filename), 'wb').write(data)

    @staticmethod
    def load(
            document_id: int
    ) -> ExtractedDoc | None:
        path = Path(DOCS_FOLDER, f'{document_id}.json')
        if not path.exists():
            return None
        data = open(path, 'rb').read()
        doc: ExtractedDoc = TypeAdapter(ExtractedDoc).validate_json(data)
        doc.id = document_id
        return doc

    def dump(
            self
    ) -> str:
        outputs: list[str] = []
        for page in self.pages:
            outputs.append(f'<------ Page {page.page_no} ------>')
            outputs.append(_reformat_paragraphs(page.content))
            if page.tables:
                outputs.append("Tables:")
                for i, table in enumerate(page.tables, 1):
                    outputs.append(f"Table {i}")
                    for row in table.rows:
                        outputs.append(row)
        return '\n'.join(outputs)


# End of ExtractedDoc class


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
        source: str,
        first_page: int | None,
        last_page: int | None
) -> list[str]:
    images = pdf2image.convert_from_path(
        pdf_path=source,
        first_page=first_page,
        last_page=last_page,
        output_folder=WORK_FOLDER,
        paths_only=True,
        dpi=DPI
    )
    return [f"{image}" for image in (images or [])]


def _extract_page_from_image(
        page_no: int,
        img_path: str,
        skip_tables: bool
) -> ExtractedPage:
    logger.info(f"Extracting page {page_no}")

    page = ExtractedPage(
        page_no=page_no,
        content='',
        tables=[]
    )
    try:
        img = Image.open(img_path)
        # img = img.crop((0, PDF_HEADER_OFFSET, img.width, img.height - PDF_FOOTER_OFFSET))
        text: str = pytesseract.image_to_string(
            image=img,
            config=TESSERACT_CONFIG
        )
        if text and (text := text.strip()):
            page.content = text
            if text.endswith(('.', ':', '!', '?')):
                page.content += '\n'
    except Exception as e:
        raise RuntimeError(f"Error recognizing text from image on page {page_no}: {e}")

    if skip_tables:
        return page

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
        document_id: int,
        checksum: str,
        source: str,
        first_page: int | None,
        last_page: int | None,
        skip_tables: bool
) -> ExtractedDoc:
    doc = ExtractedDoc(
        id=document_id,
        checksum=checksum,
        pages=[]
    )
    img_paths = _convert_pdf_to_images(
        source=source,
        first_page=first_page,
        last_page=last_page
    )
    logger.info(f"Extracted {len(img_paths)} pages")
    try:
        for i, img_path in enumerate(img_paths, first_page or 1):
            page = _extract_page_from_image(
                page_no=i,
                img_path=img_path,
                skip_tables=skip_tables
            )
            doc.pages.append(page)
    finally:
        for img_path in img_paths:
            _delete_file(img_path)
    return doc


def read_pdf(
        document_id: int,
        source: str,
        *,
        first_page: int | None = None,
        last_page: int | None = None,
        no_cache: bool = False,
        skip_tables: bool = False
) -> ExtractedDoc:
    logger.info("Read PDF")

    delete_source = False

    try:
        if is_url(source):
            source = download_url(source)
            delete_source = True

        checksum = _get_checksum(source)

        if not no_cache and (doc := ExtractedDoc.load(document_id)):
            if doc.checksum == checksum:
                return doc

        doc = _load_doc_from_source(
            document_id=document_id,
            checksum=checksum,
            source=source,
            first_page=first_page,
            last_page=last_page,
            skip_tables=skip_tables
        )
        doc.save()
        return doc

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading document: {e}")
    finally:
        if delete_source and not is_url(source):
            _delete_file(source)

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from re import match
from typing import Any

from camelot.utils import is_url, download_url
from pydantic import BaseModel, TypeAdapter

from conf import DOCS_CACHE_DIR
from pdf_document import PdfDoc, delete_file

SAVE_CHUNKS: bool = True

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



class PdfDoc2(PdfDoc):
    checksum: str



    def save(self) -> None:
        data = TypeAdapter(PdfDoc2).dump_json(self)
        filename = f'{self.id}.json'
        open(Path(DOCS_CACHE_DIR, filename), 'wb').write(data)

    @staticmethod
    def load(
            document_id: int
    ) -> PdfDoc | None:
        path = Path(DOCS_CACHE_DIR, f'{document_id}.json')
        if not path.exists():
            return None
        data = open(path, 'rb').read()
        doc: PdfDoc = TypeAdapter(PdfDoc2).validate_json(data)
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


def _get_checksum(
        source: str
) -> str:
    with open(source, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def read_pdf(
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
        if is_url(source):
            source = download_url(source)
            delete_source = True

        checksum = _get_checksum(source)

        if not no_cache and (doc := PdfDoc.load(document_id)):
            if doc.checksum == checksum:
                return doc

        doc = PdfDoc2(id=document_id, checksum=checksum)
        doc.load_from_source(
            source=source,
            first_page=first_page,
            last_page=last_page
        )
        doc.save()
        return doc

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading document: {e}")
    finally:
        if delete_source and not is_url(source):
            delete_file(source)

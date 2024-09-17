import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Union

import camelot
import pdf2image
import pytesseract
from camelot.core import TableList
from camelot.utils import is_url, download_url
from langchain_community.document_loaders import PyMuPDFLoader
from pandas import DataFrame


from conf import resolve_path

DOCS_FOLDER = f'{resolve_path("documents.root", "documents")}'
os.makedirs(DOCS_FOLDER, exist_ok=True)

WORK_FOLDER = Path(DOCS_FOLDER, 'temp')
os.makedirs(WORK_FOLDER, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DPI: int = 300

logging.getLogger("pytesseract").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)

def _clean_text(text: str):
    return re.sub('[\0\\s]+', ' ', text).strip()


def _replace_file_ext(path: str, new_ext: str):
    return Path(path).with_suffix(new_ext).as_posix()


def _delete_file(filepath: Union[str, Path]):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning(f"Error deleting {filepath}: {e}")


def _extract_table_rows(df: DataFrame) -> Union[list[str], None]:
    df = df.dropna(how="all").loc[:, ~df.columns.isin(['', ' '])]
    df = df.apply(lambda x: x.str.replace("\n", " "))
    df = df.rename(columns=df.iloc[0]).drop(df.index[0]).reset_index(drop=True)

    if df.shape[0] < 2 or df.eq("").all(axis=None):
        return None

    df["summary"] = df.apply(
        lambda x: " ".join([f"{col}: {val}, " for col, val in x.items()]),
        axis=1
    )
    return [row["summary"] for ind, row in df.iterrows()]


def _extract_tables_with_tesseract_ocr(image_paths: list[str], table_rows: list[str]):
    from img2table.ocr import TesseractOCR
    from img2table.document import Image

    ocr = TesseractOCR(
        n_threads=1,
        lang="eng"
    )

    for i, image_path in enumerate(image_paths, 1):
        try:
            page_image = Image(image_path)
            extracted_tables = page_image.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=False,
                min_confidence=10
            )
            table_count = len(extracted_tables or [])
            logger.debug(f"Found {table_count} tables ")
            if table_count > 0:
                for table in extracted_tables:
                    table_rows.extend(_extract_table_rows(table.df))
        except Exception as e:
            raise RuntimeError(f"Error extracting tables on page {i}: {e}")


class ExtractedDoc:
    contents: list[str] = []
    table_rows: list[str] = []

    def split_into_chucks(self, text_chunker):
        text = "\n".join(self.contents)
        chunks = text_chunker(text)
        if self.table_rows:
            chunks.extend(self.table_rows)
        return chunks


    def _process_page(self, page_no: int, page):
        logger.debug(f"Start processing page {page_no}")

        ppm_path = f"{page}"
        pdf_path = _replace_file_ext(ppm_path, ".pdf")
        try:

            # Convert image to PDF
            try:
                pdf_data = pytesseract.image_to_pdf_or_hocr(
                    image=ppm_path,
                    extension='pdf'
                )
                open(pdf_path, 'wb').write(pdf_data)
                logger.debug("Pdf created from image")
            except Exception as e:
                raise RuntimeError(f"Error converting image to pdf on page {page_no}: {e}")

            # Read PDF content
            try:
                docs = PyMuPDFLoader(
                    file_path=pdf_path
                ).load()

                if not docs:
                    logger.warning(f"No data on page {page_no}")
                    return

                self.contents.append(docs[0].page_content)
                logger.debug("Page content loaded")
            except Exception as e:
                raise RuntimeError(f"Error reading pdf content of page {page_no}: {e}")

            # Extract tables from PDF
            try:
                table_list: TableList = camelot.read_pdf(
                    filepath=pdf_path,
                    pages='1',
                    suppress_stdout=True,
                    backend='ghostscript',
                    resolution=DPI
                )
                logger.debug(f"Found {table_list.n} tables")

                for table in table_list:
                    rows = _extract_table_rows(table.df)
                    if rows and len(rows) > 0:
                        self.table_rows.extend(rows)

            except Exception as e:
                raise RuntimeError(f"Table extraction error: {e}")

        finally:
            _delete_file(ppm_path)
            _delete_file(pdf_path)


    def load_from_source(self, source: str):
        # Convert PDF's pages to images
        pages = pdf2image.convert_from_path(
            pdf_path=source,
            output_folder=WORK_FOLDER,
            paths_only=True,
            fmt="ppm",
            dpi=DPI
        )
        page_count = len(pages or [])
        logger.debug(f"Document contains {page_count} pages")
        if page_count >  0:
            for i, page in enumerate(pages, 1):
                self._process_page(i, page)


    def load_from_cache(self, content_path: Path, tables_path: Path):
        text = open(content_path, 'r').read()
        self.contents.append(text)
        if tables_path.exists():
            with open(tables_path) as file:
                while row := file.readline():
                    self.table_rows.append(row.rstrip())

    def save_to_cache(self, content_path: Path, tables_path: Path):
        open(content_path, 'w').write("\n".join(self.contents))
        if self.table_rows:
            open(tables_path, 'w').write("\n".join(self.table_rows))
        else:
            _delete_file(tables_path)


def read_pdf(
        document_id: int,
        source: str,
        *,
        no_cache: bool = False

) -> ExtractedDoc:

    logger.info("read_pdf")

    checksum_file = Path(DOCS_FOLDER, f'{document_id}.checksum')
    content_file = Path(DOCS_FOLDER, f'{document_id}.content')
    tables_file = Path(DOCS_FOLDER, f'{document_id}.tables')

    delete_source = False

    try:
        if is_url(source):
            source = download_url(source)
            delete_source = True

        checksum = hashlib.md5(open(source, 'rb').read()).hexdigest()

        use_cache: bool = (
                not no_cache
                and content_file.exists()
                and checksum_file.exists()
                and checksum == open(checksum_file, 'r').readline()
        )

        extracted_doc = ExtractedDoc()

        if use_cache:
            extracted_doc.load_from_cache(content_file, tables_file)
        else:
            extracted_doc.load_from_source(source)
            extracted_doc.save_to_cache(content_file, tables_file)
            open(checksum_file, 'w').write(checksum)

        return extracted_doc

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading document: {e}")
    finally:
        if delete_source and not is_url(source):
            _delete_file(source)

import logging
import re

from camelot.core import Table
from camelot.utils import is_url, download_url

logger = logging.getLogger(__name__)


def _clean_text(text: str):
    return re.sub('[\0\\s]+', ' ', text).strip()


def _get_table_rows(table: Table) -> list[str] | None:
    df_table = table.df.dropna(how="all").loc[:, ~table.df.columns.isin(['', ' '])]
    df_table = df_table.apply(lambda x: x.str.replace("\n", " "))
    df_table = df_table.rename(columns=df_table.iloc[0]).drop(df_table.index[0]).reset_index(drop=True)

    if df_table.shape[0] <= 3 or df_table.eq("").all(axis=None):
        return None

    df_table["summary"] = df_table.apply(
        lambda x: " ".join([f"{col}: {val}, " for col, val in x.items()]),
        axis=1
    )
    return [row["summary"] for ind, row in df_table.iterrows()]


def read_pdf(
        document_id: int,
        source: str,
        text_chunker,
        *,
        read_tables: bool = True,
        convert_pdf_to_image: bool = True
) -> list[str]:

    chunks: list[str] = []
    try:
        if is_url(source):
            source = download_url(source)

        pages = []
        if convert_pdf_to_image:
            import pdf2image
            import pytesseract

            pages.extend(pdf2image.convert_from_path(pdf_path=source, dpi=300))
            for page_no, page in enumerate(pages, 1):
                try:
                    text = pytesseract.image_to_string(page)
                    chunks.extend(text_chunker(text))
                except UserWarning as w:
                    raise Exception(f"Problem reading text in document_id={document_id}, page_no={page_no}: {str(w)}")
        else:
            from langchain_community.document_loaders import PyMuPDFLoader

            pages.extend(PyMuPDFLoader(source).load())
            for page in pages:
                chunks.extend(text_chunker(page.page_content))

        if read_tables:
            from camelot import read_pdf

            for page_no in range(1, len(pages)):
                try:
                    table_list = read_pdf(filepath=source, pages=str(page_no), suppress_stdout=False, backend='ghostscript')
                    for table in table_list:
                        rows = _get_table_rows(table)
                        if rows is not None and len(rows) > 0:
                            chunks.extend(rows)
                except UserWarning as w:
                    raise Exception(f"Problem reading tables in document_id={document_id}, page_no={page_no}: {str(w)}")

        return list(map(lambda txt: _clean_text(txt), chunks))
    except Exception as e:
        raise RuntimeError(f"Error reading content of documentId={document_id}: {str(e)}")

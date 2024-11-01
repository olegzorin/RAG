from re import match, split, search

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from conf import MODEL_CACHE_DIR

END_OF_SENTENCE = ('.', ':', '!', '?')
END_OF_LIST_ITEM = (';', '.', ':')

model = SentenceTransformer(
    model_name_or_path="all-MiniLM-L6-v2",
    cache_folder=MODEL_CACHE_DIR
)


def _flat_map(inp: list[str], func) -> list[str]:
    out: list[str] = []
    for x in inp:
        out.extend(func(x))
    return out


def compute_average_similarity(data: list[list[str]]) -> float:
    similarities = []
    for row in data:
        row = [text for text in row if text and text.strip()]  # Filter out empty strings
        row_len = len(row)
        if row_len > 1:
            embeddings = model.encode(row)
            similarity_matrix = cosine_similarity(embeddings)
            # Calculate the average similarity of non-diagonal elements
            avg_similarity = (similarity_matrix.sum() - similarity_matrix.trace()) / (row_len * (row_len - 1))
            similarities.append(avg_similarity)

    return np.mean(similarities) if similarities else 0


# Function to detect table alignment based on similarity
def detect_table_alignment(data: list[list[str]]):
    table_rows = data[1:]
    table_columns = [[data[j][i] for j in range(len(data[1:]))] for i in range(len(data[0]))]

    # Compute average similarity for rows and columns
    avg_row_similarity = compute_average_similarity(table_rows)
    avg_column_similarity = compute_average_similarity(table_columns)

    # Determine reading direction
    alignment = 'horizontal (read table_rows)' if avg_column_similarity > avg_row_similarity else 'vertical (read table_column)'

    return alignment  # , avg_row_similarity, avg_column_similarity


def _split_by_empty_lines(text: str) -> list[str]:
    return split(r'\s*\n\s*\n\s*', text)


# List items often lack the end-of-sentence character (.;).
# Add it to make _join_paragraphs work correctly.
def _split_by_numeration(text: str) -> list[str]:
    pars: list[str] = []
    last_par_lines = []
    item_end = ''
    for line in text.splitlines():
        if (match(r'\(?[0-9.]+\)?\s+', line) is not None
                or match(r'\(?[a-zA-Z][.)]\s+', line) is not None):
            if last_par_lines:
                par = '\n'.join(last_par_lines)
                if item_end and not par.endswith(END_OF_LIST_ITEM):
                    par += item_end
                pars.append(par)
                last_par_lines = []
            item_end = '.'
        last_par_lines.append(line)
    if last_par_lines:
        par = '\n'.join(last_par_lines)
        if item_end and not par.endswith(END_OF_LIST_ITEM):
            par += item_end
        pars.append(par)
    return pars


# List items often lack the end-of-sentence character (.;).
# Add it to make _join_paragraphs work correctly.
def _split_by_items(text: str) -> list[str]:
    pars: list[str] = []
    last_par_lines = []
    item_end = ''
    for line in text.splitlines():
        m = match(r'[oe«¢*]\s+', line)
        if m is not None:
            if last_par_lines:
                par = '\n'.join(last_par_lines)
                if item_end and not par.endswith(END_OF_LIST_ITEM):
                    par += item_end
                pars.append(par)
                last_par_lines = []
            line = '* ' + line[m.end():]
            item_end = '.'
        last_par_lines.append(line)
    if last_par_lines:
        par = '\n'.join(last_par_lines)
        if item_end and not par.endswith(END_OF_LIST_ITEM):
            par += item_end
        pars.append(par)
    return pars


# Short line with a sentence end mark at the end
# is most likely the last line of a paragraph.
def _split_by_short_lines(text: str, threshold: float = 0.8) -> list[str]:
    pars: list[str] = []
    lines = text.splitlines()
    max_line_len = max([len(l) for l in lines])
    last_par_lines = []
    for line in lines:
        last_par_lines.append(line)
        if len(line) < threshold * max_line_len and line.endswith(END_OF_SENTENCE):
            pars.append('\n'.join(last_par_lines))
            last_par_lines = []
    if last_par_lines:
        pars.append('\n'.join(last_par_lines))
    return pars


# All-caps text lines are section headins
def _split_by_headings(text: str) -> list[str]:
    pars: list[str] = []
    last_par_lines = []
    for line in text.splitlines():
        if search(r'[A-Z]+', line) and (search(r'[a-z]+', line) is None):
            if last_par_lines:
                pars.append('\n'.join(last_par_lines))
                last_par_lines = []
            prefix = '### ' if match(r'\(?[A-Z0-9][0-9]?[.)]', line) else '## '
            pars.append(prefix + line)
        else:
            last_par_lines.append(line)
    if last_par_lines:
        pars.append('\n'.join(last_par_lines))
    return pars


def _join_paragraphs(pars: list[str]):
    text = pars[0]
    for i, par in enumerate(pars[1:], 1):
        if search(r'[a-z,]$', pars[i - 1]) and match(r'[A-Za-z][a-z]+', par):
            # parts of a single paragraph: do not separate by empty line
            text += '\n' + par
        else:
            text += '\n\n' + par
    return text


def text_to_markdown(text: str) -> str:
    pars = _split_by_empty_lines(text)
    pars = _flat_map(pars, _split_by_short_lines)
    pars = _flat_map(pars, _split_by_headings)
    pars = _flat_map(pars, _split_by_items)
    pars = _flat_map(pars, _split_by_numeration)
    return _join_paragraphs(pars)


def _is_table_with_headers(rows: list[list[str]]) -> bool:
    # Check number of rows
    if len(rows) < 2:
        return False

    # Check number of columns
    if len(rows[0]) > 3:
        return True

    # Check num of lines in the cells of the first row
    h_lines = [len(cell.splitlines()) for cell in rows[0]]
    if min(h_lines) == 0:  # headers must not be empty
        return False
    if max(h_lines) == 1:  # most likely the headers row
        return True

    max_lines_in_first_column = max([len(row[0].splitlines()) for row in rows[1:]])
    max_lines_in_second_column = max([len(row[1].splitlines()) for row in rows[1:]])
    return max_lines_in_first_column <= 1 or max_lines_in_second_column <= 1


def _table_cell_to_markdown(text: str) -> str:
    if not text.strip():
        return ''
    pars = _split_by_empty_lines(text)
    pars = _flat_map(pars, _split_by_short_lines)
    pars = _flat_map(pars, _split_by_items)
    pars = _flat_map(pars, _split_by_numeration)
    return _join_paragraphs(pars).replace('\n', '<br/>')


def table_to_markdown(rows: list[list[str]]):
    with_headers = _is_table_with_headers(rows)

    n_cols = len(rows[0])

    for i in range(len(rows)):
        for j in range(n_cols):
            rows[i][j] = _table_cell_to_markdown(rows[i][j])

    if not with_headers:
        # Add empty headers
        rows.insert(0, ['     '] * n_cols)

    # Insert ruler
    rows.insert(1, [':------'] * n_cols)

    return ''.join(['| ' + ' | '.join(row) + ' |\n' for row in rows])


def split_text(text: str, chunker, max_chunk_size: int = 0):
    chunks: list[str] = chunker.split_text(text)
    large_count: int
    print('count: ', len(chunks))
    if max_chunk_size == 0 or (large_count := len([c for c in chunks if len(c) > max_chunk_size])) == 0:
        return chunks

    while large_count > 0:
        large_count = 0
        new_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                new_chunks.append(chunk)
            else:
                large_count += 1
                sub_chunks = chunker.split_text(chunk)
                if len(sub_chunks) == 1:
                    raise Exception("Cannot reduce chunk: " + chunk)
                new_chunks.extend(sub_chunks)
        chunks = new_chunks
        print('count: ', len(chunks), ', large: ', large_count)

    return chunks

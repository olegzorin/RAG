import re
from re import match, sub, split, search

END_OF_SENTENCE = ('.', ':', '!', '?')

def _fix_ocr_typos(text: str) -> str:
    text = text.replace('|', 'I')
    return sub(r'Il+(?![A-Za-z])', lambda x: x.group().replace('l', 'I'), text)


def _flat_map(inp: list[str], func) -> list[str]:
    out: list[str] = []
    for x in inp:
        out.extend(func(x))
    return out


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
                if item_end and not par.endswith((';', '.')):
                    par += item_end
                pars.append(par)
                last_par_lines = []
            item_end = '.'
        last_par_lines.append(line)
    if last_par_lines:
        par = '\n'.join(last_par_lines)
        if item_end and not par.endswith((';', '.')):
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
                if item_end and not par.endswith((';', '.')):
                    par += item_end
                pars.append(par)
                last_par_lines = []
            line = '* ' + line[m.end():]
            item_end = '.'
        last_par_lines.append(line)
    if last_par_lines:
        par = '\n'.join(last_par_lines)
        if item_end and not par.endswith((';', '.')):
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
    text = pars[0] + '\n'
    for i, par in enumerate(pars[1:], 1):
        if search(r'[a-z,]$', pars[i - 1]) and match(r'[A-Za-z][a-z]+', par):
            # parts of a single paragraph: do not separate by empty line
            text += par + '\n'
        else:
            text += '\n' + par + '\n'
    return text


def text_to_markdown(text: str) -> str:
    text = _fix_ocr_typos(text)
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
    if len(rows[0]) > 2:
        return True

    # Check max num of lines in the cells of the first row
    max_lines_in_first_row = max([len(cell.splitlines()) for cell in rows[0]])
    if max_lines_in_first_row != 1:
        return False

    max_lines_in_first_column = max([len(row[0].splitlines()) for row in rows[1:]])
    max_lines_in_second_column = max([len(row[1].splitlines()) for row in rows[1:]])
    return max_lines_in_first_column <= 1 or max_lines_in_second_column <= 1


def table_to_html(data: list[list[str]]):
    return ('<table border="1px"><tr><td>'
            + '</td></tr><tr><td>'.join(['</td><td>'.join(row) for row in data])
            + '</td></tr></table>')


def table_to_markdown(rows: list[list[str]]):
    n_rows, n_cols = (len(rows), len(rows[0]))

    for i in range(n_rows):
        for j in range(n_cols):
            rows[i][j] = _fix_ocr_typos(rows[i][j]).replace('\n', '<br/>')

    if not _is_table_with_headers(rows):
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

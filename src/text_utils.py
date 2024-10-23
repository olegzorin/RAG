from re import match, sub, split, search


def _flat_map(inp: list[str], func) -> list[str]:
    out: list[str] = []
    for x in inp:
        out.extend(func(x))
    return out


def _split_by_empty_lines(text: str) -> list[str]:
    return split(r'\s*\n\s*\n\s*', text)


def _split_by_numeration(text: str) -> list[str]:
    pars: list[str] = []
    last_par = ''
    for line in text.splitlines():
        if (match(r'\(?[0-9.]+\)?\s+', line) is not None
                or match(r'\(?[a-zA-Z][.)]\s+', line) is not None
                or match(r'[oe*]\s+', line) is not None):
            if last_par:
                pars.append(last_par)
                last_par = ''
        last_par += line + '\n'
    if last_par:
        pars.append(last_par)
    return pars


def _split_by_short_lines(text: str, threshold: float = 0.8) -> list[str]:
    pars: list[str] = []
    lines = text.splitlines()
    max_line_len = max([len(l) for l in lines])
    last_par = ''
    for line in lines:
        last_par += line + '\n'
        if len(line) < threshold * max_line_len and line.endswith(('.', ':', '!', '?')):
            pars.append(last_par)
            last_par = ''
    if last_par:
        pars.append(last_par)
    return pars


def _split_by_headings(text: str) -> list[str]:
    pars: list[str] = []
    last_par = ''
    for line in text.splitlines():
        if search(r'[A-Z]+', line) and (search(r'[a-z]+', line) is None):
            if last_par:
                pars.append(last_par)
                last_par = ''
            prefix = '### ' if match(r'\(?[A-Z][.)]', line) else '## '
            pars.append(prefix + line + '\n')
        else:
            last_par += line + '\n'
    if last_par:
        pars.append(last_par)
    return pars


def _fix_ocr_typos(text: str) -> str:
    text = text.replace('|', 'I')
    return sub(r'Il+(?![A-Za-z])', lambda x: x.group().replace('l', 'I'), text)


def text_to_markdown(text: str) -> str:
    text = _fix_ocr_typos(text)
    pars = _split_by_empty_lines(text)
    pars = _flat_map(pars, _split_by_numeration)
    pars = _flat_map(pars, _split_by_short_lines)
    pars = _flat_map(pars, _split_by_headings)
    return ''.join(pars)


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
    if not _is_table_with_headers(rows):
        rows.insert(0, ['     ' for _ in rows[0]])
    text = '| ' + ' | '.join(rows[0]).replace('\n', '<br/>') + ' |\n'
    text += ''.join(['|:------' for _ in rows[0]]) + '|\n'
    for row in rows[1:]:
        text += '| ' + ' | '.join(row).replace('\n', '<br/>') + ' |\n'
    return text


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

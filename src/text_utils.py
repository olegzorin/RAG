
def reformat_paragraphs(
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

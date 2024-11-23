import re
from pathlib import Path

from pydantic import TypeAdapter, BaseModel

from pdf_document import PdfFile, PdfDoc, TextBlock, PdfPage
from re import search, match, compile

from text_utils import table_to_markdown

doc_path = Path("docs/CCR.json")

doc: PdfDoc = TypeAdapter(PdfFile).validate_json(doc_path.read_bytes())


class Table(BaseModel):
    data: list[list[str]]


content: list[str | Table] = []

for page in doc.pages:
    for tb in page.contents:
        if tb.cols > 1:
            content.append(Table(data=tb.data))
        else:
            x = '\n\n'.join([row[0] for row in tb.data])
            if content and isinstance(content[-1], str):
                content[-1] += '\n' + x
            else:
                content.append(x)

items = [x if isinstance(x, str) else table_to_markdown(x.data) for x in content]
Path('CCR.txt').write_text('\n\n'.join(items))


def is_heading_line(line: str):
    return (len(line) > 0 and search(r'[A-Z]+', line)
            and (search(r'[a-z]+', line) is None)
            and (search(r' {2,}', line) is None))


def split_by_headings(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        line: str = line.strip()
        heading_line = is_heading_line(line)
        if lines[-1].startswith('### '):
            if len(line) == 0 or heading_line:
                lines[-1] += line
            else:
                lines.append(line)
        elif heading_line:
            lines.append('### ' + line)
        else:
            lines.append(line)


sections: list[PdfDoc] = []

# text_blocks: list[TextBlock] = []
# for page in doc.pages:
#     for text_block in page.text_blocks:
#         if text_block.cols > 1:
#             text_blocks.append(text_block)
#         else:

txt = '''
Super

ATTACHMENT.

PROVIDER REIMBURSEMENT.
Simple text

SKILLED NURSING FACILITY ("PCT")

COMMERCIAL AND MEDICARE ADVANTAGE PLANS:

Proba pera

Advantage Network.
ARTICLE II
SERVICES/OBLIGATIONS
2.1 Participation-Medicare Advantage. As a participant in Plan's Med
'''

# out: list[tuple[str,str]] = [()]
#
pat = compile(r'\n([^a-z ]+ )+[^a-z ]+\n')
pos = 0
while (match := pat.search(txt, pos)) is not None:
    start = match.start()

    res = txt[match.start(): match.end()].strip()
    print('(' + re.sub(r'\s+', ' ', txt[pos:match.start()]).strip() + ')')
    print('[' + re.sub(r'\s+', ' ', res) + ']')
    pos = match.end()

if pos < len(txt):
    print('(' + txt[pos:].strip() + ')')

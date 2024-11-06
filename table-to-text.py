from pdf_document import TextBlock

data = [
    [
        "Service:\n",
        "Reimbursement:\n"
    ],
    [
        "Drugs & Biologicals\n",
        "100% of ChoiceCare’s 201-544 fee schedule\n"
    ],
    [
        "All other services\n",
        "95% of ChoiceCare’s 005-270 fee schedule\n"
    ]
]


def clean(s: str) -> str:
    return s.strip('\n\r\t: ').replace('\n', ' ')


text = ''
for row in data[1:]:
    text += 'If ' + clean(data[0][0]) + ' is ' + clean(row[0]) + ' then '
    text += ' and '.join([clean(data[0][i]) + ' is ' + clean(row[i]) for i in range(1, len(row))])
    text += '.\n'

# print(text)

print(isinstance('', str))
xs = [
    TextBlock(cols=1, rows=1, data=[['Proba\n']]),
    TextBlock(cols=1, rows=2, data=[['novogo\n'],['pera\n']]),
    TextBlock(cols=2, rows=2, data=[['one\n', 'two\n'], ['three\n', 'four\n']]),
    TextBlock(cols=1, rows=1, data=[['The end\n']])
]

blocks: list[str | TextBlock] = ['']
for x in xs:
    was_text = isinstance(blocks[-1], str)
    if x.cols == 1:
        text = ''.join([r[0] for r in x.data])
        if was_text:
            blocks[-1] += text
        else:
            blocks.append(text)
    else:
        blocks.append(x)
print(blocks)




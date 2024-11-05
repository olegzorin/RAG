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

print(text)

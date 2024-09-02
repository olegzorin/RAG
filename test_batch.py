from typing import Dict

from core import Ragger

params: Dict = {
    "embeggingsModel": "all-MiniLM-L6-v2",
    "output.model": "numind/NuExtract",
    "ollama.model": "llama3.1:70b",
    "ollama.temperature": "0",
    "ollama.seed": "2",
    "ollama.top_k": "80",
    "ollama.top_p": "0.9",
    "ollama.num_ctx": "4096",
    "ollama.num_predict": "-2"
}

names = ['MHR', 'CCR', 'LHS']

question_keys: list[str] = []
questions: list[str] = []
with open('questions.txt') as qf:
    for line in qf.read().splitlines(False):
        kv = line.split(';')
        question_keys.append(kv[0])
        questions.append(kv[1])

ragger = Ragger(params)

for documentId, name in enumerate(names):
    print(f'start {name}')
    answers = ragger.get_answers(documentId, f'/Users/oleg/Downloads/{name}.pdf', questions)
    with open(f'{name}_70b.out', 'w') as o:
        for i, answer in enumerate(answers):
            o. write(question_keys[i] + ':\n' + answer.replace('\n', '\\n') + '\n\n')
    print(f'end {name}')



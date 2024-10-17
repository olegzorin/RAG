from pathlib import Path

import pymupdf4llm

from reader import PdfDoc, ExtractedPage
from vector import VectorSearch

pdf_path = 'docs/CCR.pdf'

md_text = pymupdf4llm.to_markdown(
    doc=pdf_path,
    show_progress=False
)

Path(pdf_path).with_suffix(".md").write_bytes(md_text.encode())

vector = VectorSearch(
    params={
        "search.type": "similarity",
        "embeddings.use_gpu": "False",
        "output.use_gpu": "False",
        "ollama.temperature": "0.0",
        "ollama.seed": "2",
        "ollama.top_k": "10",
        "ollama.top_p": "0.3",
        "ollama.num_ctx": "4096",
        "ollama.num_predict": "-2",
        "tokenizer.max_length": "3900",
        "search.method": "vector",
        "embeddings.model": "all-MiniLM-L6-v2",
        "ollama.model": "llama3.1"
    }
)

doc = PdfDoc(
    id=1,
    checksum='123',
    pages=[
        ExtractedPage(page_no=i, content=text.strip(), tables=[]) for i, text in enumerate(md_text.split('\n\n-----\n\n'), 1)
    ]
)

questions = [
    "If leveled plan, what are the per diem rates for each level?",
    "If level plan, what levels are included and what is the name of each level?",
    "If level plan, what are the revenue codes for each level?",
    "Please list all included services for each level and provide the level definition."
]

answers = vector.get_answers(
    document=doc,
    questions=questions
)
[print(f'{qa[0]}:\n{qa[1]}\n\n-----\n\n') for qa in zip(questions, answers)]

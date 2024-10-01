import logging
import os
import shutil

from ragatouille import RAGPretrainedModel

import reader
from reader import ExtractedDoc

os.environ['CC'] = '/usr/bin/gcc'
os.environ['CXX'] = '/usr/bin/g++'
# need gcc at least version 10

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


RAG: RAGPretrainedModel = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


logging.basicConfig(
    level=logging.INFO,
    force=True
)

document_id = 1
document_name = 'CCR 1'

shutil.copy(f'../docs/{document_name}.json', f'{reader.DOCS_FOLDER}/{document_id}.json')

questions = {
    'AGR_DATE': "What is the effective date of the agreement?",
    'AGR_TERM_MONTHS': "What is the term of the agreement in months?",
    'RNW_PERIOD': "What is the autorenewal period in months?",
    'TRM_DAYS': "How many days notice is needed to terminate without cause?",
    'CLM_SUB_DAYS': "What is the claims submission period in days?",
    'CLM_APP_DAYS': "How many days does the provider have to appeal a claim?",
    'CLM_TIMELINE': "What is the claims payment timeline?",
    'BIZ_LINES': "What lines of business are included within the agreement?",
    'RMB_STRUCT': "What is the reimbursement structure?",
    'PDPM_PERC': "If PDPM (Medicare allowable), what percentage is paid per day?",
    'PLAN_LEVELS_RATES': "If leveled plan, what are the per diem rates for each level?",
    'PLAN_LEVELS': "If level plan, what levels are included and what is the name of each level?",
    'PLAN_REV_CODES': "If level plan, what are the revenue codes for each level?",
    'SERV_LEVELS': "Please list all included services for each level and provide the level definition.",
    'EXCLUSIONS': "Are there any full contract exclusions?",
    'HIGH_COSTS': "Please list any high cost medication provisions found in the agreement.",
    'HIGH_COSTS_CLAIMS': "Please list all necessary items needed for the claim for high cost medications to be reimbursed.",
    'FACILITIES': "What facilities are included on the agreement?",
    'CLM_ADDRESS': "What is the claims submission address?"
}

doc: ExtractedDoc = reader.read_pdf(
    document_id=document_id,
    source=f'../docs/{document_name}.pdf'
)

content: str = doc.get_content()
tables: list[str] = doc.get_table_rows()
RAG.index(
    collection=[content],
    document_ids=[document_name],
    document_metadatas=[{"entity": "PCT", "source": document_name}],
    index_name="documents",
    max_document_length=512,
    split_documents=True
)
RAG.add_to_index(
    new_collection=tables,
    index_name="documents",
    split_documents=False
)

k = 3  # How many documents you want to retrieve, defaults to 10, we set it to 3 here for readability
for question in list(questions.values()):
    results: list[dict] = RAG.search(
        query=question,
        k=k
    )
    print(f'\nQ: {question}')
    for res in results:
        print(f"R: rank={res['rank']}, score={res['score']}")
        print(f"A: {res['content']}")

import base64
import json
import logging
import shutil
import time

import main
import reader

logging.basicConfig(
    level=logging.INFO,
    force=True
)

document_id = 1
document_name = 'MHR'

shutil.copy(f'docs/{document_name}.json', f'{reader.DOCS_FOLDER}/{document_id}.json')

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

def run():
    req = {
        "documentId": document_id,
        "url": f'docs/{document_name}.pdf',
        "questions": list(questions.values()),

        #         "template": {"Outputs": [{"Key": "", "Value": "", "Type": ""}]},
        #         "examples": [
        #             {"Output":{"Key":"Level 1 Rate","Value":"100","Type":"INT"}},
        #             {"Output":{"Key":"Level 2 Rate","Value":"150","Type":"INT"}},
        #             {"Output": {"Key": "Level 3 Rate", "Value": "200", "Type": "INT"}}
        #         ],
        "params": {
            "embeddings.use_gpu": "True",
            "output.use_gpu": "False",
            "reader.chunk_size": "500",
            "embeddings.model": "BAAI/bge-m3",  # replaces previously used "all-MiniLM-L6-v2"
            "search.method": "keywords",  # vector",  # "vector" or "graph" (aka parent/child)
            "search.type": "similarity",
            # only for search method "vector". Comma-separated list of bm25, similarity, mmr
            # "ollama.model": "llama3.1",   # to be installed later: "eas/dragon-yi-v0"
            "ollama.model": "eas/dragon-yi-v0",
            "ollama.temperature": "0.0",
            "ollama.seed": "2",
            "ollama.top_k": "10",
            "ollama.top_p": "0.3",
            "ollama.num_ctx": "4096",
            "ollama.num_predict": "-2",
            "tokenizer.max_length": "3900",
            "chat_prompt_system_message": "Please give me precise information. Don't be verbose."
        }
    }
    encoded = base64.b64encode(json.dumps(req).encode())
    outpath = f'docs/{document_name}.answers'
    main.run(encoded.decode('utf8'), outpath)
    resp = json.loads(open(outpath, 'r').read())

    question_keys = list(questions.keys())
    for i, answer in enumerate(resp['answers']):
        print(f'{i+1}. {question_keys[i]}: {answer}')

print(time.ctime())
run()
print(time.ctime())


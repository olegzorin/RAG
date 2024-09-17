import base64
import json
import time

import main


def run():
    req = {
        "documentId": 1,
        "url": '/Users/oleg/Downloads/CCR 1.pdf',
        # "url": 'https://s3.amazonaws.com/sbox.ragdoc/208?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240912T083501Z&X-Amz-SignedHeaders=host&X-Amz-Credential=AKIAJPAZDO6JTXZ6FQSQ%2F20240912%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Expires=43200&X-Amz-Signature=289844399c6135026e37117ba5628273474dc99af242ab996e1682c5ba79c05f',
        # "url": '/opt/greenxserver/ragagent/python-test/CCR_1.pdf',
        "questions": ['What is the EFFECTIVE DATE of this agreement?'],
# Right answer is 2022-01-04 (PDF Pg 12)?'],
#         "template": {"Outputs": [{"Key": "", "Value": "", "Type": ""}]},
#         "examples": [
#             {"Output":{"Key":"Level 1 Rate","Value":"100","Type":"INT"}},
#             {"Output":{"Key":"Level 2 Rate","Value":"150","Type":"INT"}},
#             {"Output": {"Key": "Level 3 Rate", "Value": "200", "Type": "INT"}}
#         ],
        "params": {
            "reader.chunk_size": "384",
            "reader.chunk_overlap": "48",
            "reader.stop_on_table_errors": "false",
            "embeddings.model": "BAAI/bge-m3",  # replaces previously used "all-MiniLM-L6-v2"
            "search.method": "vector",    # "vector" or "graph" (aka parent/child)
            "search.type": "similarity",  # only for search method "vector". Comma-separated list of bm25, similarity, mmr
            "ollama.model": "llama3.1",   # to be installed later: "eas/dragon-yi-v0"
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
    # print(encoded)
    # print(time.ctime())
    main.run(encoded.decode('utf8'))  #, '/tmp/llama31_70b.out')
    # print(time.ctime())
    # subprocess.call(["./run.sh", encoded.decode()])


run()


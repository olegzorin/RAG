import base64
import json
import time

import main


def run():
    req = {
        "documentId": 1,
        # "url": '/Users/oleg/Downloads/LHS 1.pdf',
        "url": '/Users/oleg/Downloads/CCR 1.pdf',
        "questions": ['What is the EFFECTIVE DATE of this agreement?'],
# Right answer is 2022-01-04 (PDF Pg 12)?'],
#         "template": {"Outputs": [{"Key": "", "Value": "", "Type": ""}]},
#         "examples": [
#             {"Output":{"Key":"Level 1 Rate","Value":"100","Type":"INT"}},
#             {"Output":{"Key":"Level 2 Rate","Value":"150","Type":"INT"}},
            # {"Output": {"Key": "Level 3 Rate", "Value": "200", "Type": "INT"}}
        # ],
        "params": {
            "reader.convertPdf2Image": "true",
            "reader.read_tables" : "false",
            "ollama.model": "llama3.1",
            "ollama.temperature": "0.0",
            "ollama.seed": "2",
            "ollama.top_k": "10",
            "ollama.top_p": "0.3",
            "ollama.num_ctx": "4096",
            "ollama.num_predict": "-2"
        }
    }

    encoded = base64.b64encode(json.dumps(req).encode())
    print(time.ctime())
    main.run(encoded.decode('utf8'))  #, '/tmp/llama31_70b.out')
    print(time.ctime())
    # subprocess.call(["./run.sh", encoded.decode()])


run()


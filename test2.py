import json
import time
from typing import Any
from xml.etree.ElementTree import indent

from core import generate_outputs
from pydantic import Json

print(time.ctime())

question = 'What is the contracted rate for level 1?'
answers = [
    "Level 1: $180.24 Per Diem\nLevel 2: $275.75 Per Diem\nLevel 3: $427.09 Per Diem\nLevel 4: $478.94 Per Diem",
    "Level 1: $18 Per Diem\nLevel 2: $27 Per Diem\nLevel 3: $427 Per Diem"
]

schema: Json[Any] = {"Outputs": [{
        "Key": "",
        "Value": "",
        "Unit": ""
    }]}

example: Json[Any] =[{"Outputs": [
    {"Key": "Level 1", "Value": "$000", "Unit": "Per Diem"}
]}]



print(json.dumps(generate_outputs(answers, schema, example), indent=4))

# def predict_NuExtract(model, tokenizer, text, schema, example):
#     schema = json.dumps(schema, indent=4)
#     input_llm = "<|input|>\n### Template:\n" + schema + "\n"
#     if example != "":
#         input_llm += "### Examples:\n" + json.dumps(example, indent=4) + "\n"
#
#     input_llm += "### Text:\n" + text + "\n<|output|>\n"
#     # input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
#     input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000)
#
#     output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
#     return "prediction: "+output.split("<|output|>")[1].split("<|end-output|>")[0]
#
#
# model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
#
# # model.to("cuda")
#
# model.eval()
#
#
# print(example)
#
# prediction = predict_NuExtract(model, tokenizer, answer, schema, example)
# # prediction = predict_NuExtract(model, tokenizer, output, schema)#
# print(prediction)
#
# print(time.ctime())

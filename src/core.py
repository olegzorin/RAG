import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

import conf

class Chunker:
    s: SemanticChunker
    t: TokenTextSplitter
    r: RecursiveCharacterTextSplitter

    def __init__(self, embeddings: Embeddings, chunk_size: int, chunk_overlap: int):
        self.s = SemanticChunker(embeddings)
        self.t = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        self.r = RecursiveCharacterTextSplitter(chunk_size=chunk_size*3/4, chunk_overlap=chunk_overlap, length_function=len)

    def __call__(self, text: str) -> list[str]:
        return self.s.split_text(text) + self.t.split_text(text) + self.r.split_text(text)


class RagSearch(ABC):
    params: Dict
    embeddings_model: Embeddings
    llm: ChatOllama

    def __init__(self, params: Dict):

        self.params = params

        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=params.get("embeggingsModel", 'all-MiniLM-L6-v2'),
            cache_folder=conf.resolve_path('embeddingsModel.cacheDir', 'caches/embeddings').as_posix()
        )

        self.llm = ChatOllama(
            base_url=conf.get_property('ollama.url'),
            streaming=True,
            model=params.get('ollama.model', 'llama3.1'),
            temperature=float(params.get('ollama.temperature', 0.0)),
            # Increasing the temperature will make the model answer more creatively. (Default: 0.8)
            seed=int(params.get('ollama.seed', 2)),
            # seed should be set for consistent responses
            top_k=int(params.get('ollama.top_k', 10)),
            # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=float(params.get('ollama.top_p', 0.3)),
            # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=int(params.get('ollama.num_ctx', 3072)),
            # Sets the size of the context window used to generate the next token.
            num_predict=int(params.get('ollama.num_predict', -2))
            # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
        )

    def get_text_splitter(self, chunk_size: int, chunk_overlap: int):
        return Chunker(self.embeddings_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @abstractmethod
    def get_answers(self, document_id: int, source: str, questions: list[str]) -> list[str]:
        pass

    def generate_outputs(self, texts: List[str], template: Any, examples: List[Any]) -> List[Any]:

        model_name = self.params.get("output.model", "numind/NuExtract-tiny")

        model_cache_dir = conf.resolve_path('outputsModel.cacheDir', 'caches/outputs')

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=model_cache_dir
            )
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True
            )
            model.generation_config.pad_token_id = tokenizer.eos_token_id

            input_llm: list[str] = ["<|input|>"]
            if template is not None:
                input_llm.extend(["### Template:", json.dumps(template, indent=4)])

            if examples is not None:
                for example in examples:
                    input_llm.extend(["### Example:", json.dumps(example, indent=4)])

            input_llm.extend(["### Text:", "...", "<|output|>", ""])

            outputs = []
            max_length = int(self.params.get('tokenizer.max_length', 4000))

            for text in texts:
                input_llm[-3] = text
                input_ids = tokenizer("\n".join(input_llm), return_tensors="pt", truncation=True, max_length=max_length)
                output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
                output = output.split("<|output|>")[1].split("<|end-output|>")[0].strip()
                try:
                    outputs.append({} if len(output) == 0 else json.loads(output))
                except Exception as e:
                    outputs.append({"error": str(e), "raw": output})

            return outputs

        except Exception as e:
            raise RuntimeError(f"Error generating outputs: {str(e)}")




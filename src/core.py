import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import torch
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

import conf
from reader import ExtractedDoc

cache_folder = conf.resolve_path('model.cacheDir', 'caches').as_posix()

DEFAULT_EMBEDDINGS_MODEL = "BAAI/bge-m3"  # "all-MiniLM-L6-v2"
DEFAULT_GENERATIVE_MODEL = "llama3.1"  # "eas/dragon-yi-v0"
DEFAULT_OUTPUTS_MODEL = "numind/NuExtract-tiny"

CUDA_PARAMS = {}  # {'device': 'cuda'}


def _empty_cache():
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()


class RagSearch(ABC):
    params: Dict
    embeddings_model: Embeddings
    embeddings_dimension: int
    llm: ChatOllama

    def __init__(self, params: Dict):

        self.params = params

        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=params.get("embeddings.model", DEFAULT_EMBEDDINGS_MODEL),
            model_kwargs=CUDA_PARAMS,
            encode_kwargs=encode_kwargs,
            cache_folder=cache_folder
        )

        self.embeddings_dimension = len(self.embeddings_model.embed_query("foo"))

        self.llm = ChatOllama(
            base_url=conf.get_property('ollama.url'),
            streaming=True,
            model=params.get('ollama.model', DEFAULT_GENERATIVE_MODEL),
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

    @abstractmethod
    def get_answers(
            self,
            document: ExtractedDoc,
            questions: list[str]
    ) -> list[str]:
        pass

    def _retrieve_answers(
            self,
            retriever: BaseRetriever,
            questions: list[str]
    ) -> list[str]:

        system_message = self.params.get(
            'chat_prompt_system_message',
            "Please give me precise information. Don't be verbose."
        )
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", """<context>
                    {context}
                    </context>

                    Answer the following question: 

                    {question}"""),
            ]
        )

        qa_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | self.llm
                | StrOutputParser()
        )

        answers = []
        for i, question in enumerate(questions):
            try:
                answers.append(qa_chain.invoke(question))
            except Exception as e:
                raise RuntimeError(f"Error getting an answer to question #{i}: {e}")

        return answers

    def generate_outputs(
            self,
            texts: List[str],
            template: Any,
            examples: List[Any]
    ) -> (int, List[Any]):

        start_time = time.time_ns()

        model_name = self.params.get("output.model", DEFAULT_OUTPUTS_MODEL)

        print("model name: " + model_name)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_folder
            )
            # model.to('cuda')
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
            print(f'max length = {max_length}')

            for text in texts:
                input_llm[-3] = text
                input_ids = tokenizer("\n".join(input_llm), return_tensors="pt", truncation=True, max_length=max_length)
                output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
                _empty_cache()
                output = output.split("<|output|>")[1].split("<|end-output|>")[0].strip()
                try:
                    outputs.append({} if len(output) == 0 else json.loads(output))
                except Exception as e:
                    outputs.append({"error": str(e), "raw": output})

            end_time = time.time_ns()
            elapsed_time_ms = (end_time - start_time) // 1000_000

            return elapsed_time_ms, outputs

        except Exception as e:
            raise RuntimeError(f"Error generating outputs: {str(e)}")

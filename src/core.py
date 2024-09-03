import json
import logging
import os
import time

import conf
import reader

from typing import List, Any, Dict, Optional

import torch
from langchain_community.chat_models import ChatOllama
from langchain.retrievers import EnsembleRetriever

from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

from db import DB

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class SearchType:
    BASIC = "similarity"
    BM25 = "bm25"
    MMR = "mmr"
    values = [BASIC, BM25, MMR]


class Ragger:

    params: Dict
    search_types: list[str]
    embeddingsModel: Embeddings
    llm: ChatOllama

    def __init__(self, params: Dict):
        self.params = params

        self.search_types = params.get("search_types", SearchType.BASIC).split(',')
        for search_type in self.search_types:
            if search_type not in SearchType.values:
                raise TypeError(f"Invalid search type '{search_type}', valid types: {SearchType.values}")

        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=params.get("embeggingsModel", 'all-MiniLM-L6-v2'),
            cache_folder=conf.resolve_path('embeddingsModel.cacheDir', 'caches/embeddings').as_posix()
        )

        logger.info(f"loaded embeddings: {time.ctime()}")

        self.llm = ChatOllama(
            base_url=conf.get_property('ollama.url'),
            streaming=True,
            model=params.get('ollama.model', 'llama2'),
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
        logger.info(f"loaded LLM: {time.ctime()}")


    def get_answers(self, document_id: int, url: str, questions: List[str]) -> List[str]:
        chunks = reader.read_pdf(document_id, url, SemanticChunker(embeddings=self.embeddings_model))

        logger.info(f"read document: {time.ctime()}")

        vectorstore: Optional[DB] = None

        try:

            retrievers = []
            for search_type in self.search_types:
                if search_type == SearchType.BM25:
                    from langchain_community.retrievers import BM25Retriever
                    bm25_retriever = BM25Retriever.from_texts(chunks)
                    bm25_retriever.k = 3
                    retrievers.append(bm25_retriever)
                else:
                    if vectorstore is None:
                        vectorstore = DB.get_instance(
                            name=self.params.get("vectorstore", DB.NEO4J),
                            embeddings_model=self.embeddings_model
                        )
                        vectorstore.add_texts(chunks)

                    retrievers.append(vectorstore.as_retriever(search_type=search_type))

            ensemble_retriever = EnsembleRetriever(retrievers=retrievers)

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
                    {"context": ensemble_retriever, "question": RunnablePassthrough()}
                    | rag_prompt
                    | self.llm
                    | StrOutputParser()
            )

            return [qa_chain.invoke(question).strip() for question in questions]

        finally:
            if vectorstore is not None:
                vectorstore.close()


    def generate_outputs(self, texts: List[str], template: Any, examples: List[Any]) -> List[Any]:
        output_model = self.params.get("output.model", "numind/NuExtract-tiny")
        output_model_cache_dir = conf.resolve_path('outputsModel.cacheDir', 'caches/outputs')

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=output_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=output_model_cache_dir
            )
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=output_model,
                trust_remote_code=True
            )
            model.generation_config.pad_token_id = tokenizer.eos_token_id

            input_llm = ["<|input|>"]
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
                    logger.exception(e)
                    outputs.append({"error": str(e), "raw": output})

            return outputs

        except Exception as e:
            logger.exception(e)
            raise RuntimeError(f"Error generating outputs: {str(e)}")

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

import torch
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker

from conf import get_property, MODEL_CACHE_DIR
from pdf_document import PdfDoc

DEFAULT_EMBEDDINGS_MODEL = "BAAI/bge-m3"
# DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GENERATIVE_MODEL = "llama3.1"  # "eas/dragon-yi-v0"
DEFAULT_OUTPUTS_MODEL = "numind/NuExtract-tiny"

logger = logging.getLogger(__name__)

gpu_device = 'cuda' if torch.cuda.is_available() \
    else 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
    else 'cpu'


def _empty_gpu_cache(device: str):
    if device == 'cuda':
        import gc
        gc.collect()
        torch.cuda.empty_cache()


MAX_CHUNK_SIZE = 3000


class ParagraphSplitter:
    @staticmethod
    def split_text(text: str) -> list[str]:
        chunks = []
        par = []
        for line in text.splitlines():
            if line:
                par.append(line)
            elif par:
                chunks.append('\n'.join(par))
                par = []
        if par:
            chunks.append('\n'.join(par))
        return chunks


class SemanticSplitter(SemanticChunker):
    def __init__(
            self,
            embeddings: Embeddings,
            chunk_size: Optional[int] = None
    ):
        super().__init__(
            embeddings=embeddings
        )
        self.chunk_size = chunk_size

    def split_text(
            self,
            text: str
    ) -> list[str]:
        self.number_of_chunks = len(text) // self.chunk_size if self.chunk_size else None
        return super().split_text(text)


class RagSearch(ABC):

    def _get_computing_device(self, param_key: str, default_value: bool) -> str:
        no_gpu = self.params.get(param_key, str(default_value)).lower() == 'false'
        return 'cpu' if no_gpu else gpu_device

    def __init__(self, params: Dict):

        self.params = params

        embeddings_model = params.get("embeddings.model", DEFAULT_EMBEDDINGS_MODEL)
        logger.info(f"Embeddings model: {embeddings_model}")

        computing_device = self._get_computing_device("embeddings.use_gpu", True)
        logger.info(f"Embeddings computing device: {computing_device}")
        _empty_gpu_cache(computing_device)

        model_kwargs = {'device': computing_device}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=embeddings_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=MODEL_CACHE_DIR
        )

        self.embeddings_dimension = len(self.embeddings_model.embed_query("foo"))

        self.llm = ChatOllama(
            base_url=get_property('ollama.url'),
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
            document: PdfDoc,
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
            template: Optional[str] = None,
            examples: List[str] = None
    ) -> List[Any]:

        str_template = json.dumps(template, indent=4) if template else None
        str_examples = [json.dumps(example, indent=4) for example in examples if
                        example is not None] if examples else None
        if not str_template and not str_examples:
            return None

        device = self._get_computing_device("output.use_gpu", True)
        logger.info(f"Output computing device: {device}")

        model_name = self.params.get("output.model", DEFAULT_OUTPUTS_MODEL)

        logger.info("model name: " + model_name)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE_DIR
            )
            model.to(device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True
            )
            model.generation_config.pad_token_id = tokenizer.eos_token_id

            input_llm: list[str] = ["<|input|>"]
            if str_template is not None:
                input_llm.extend(["### Template:", str_template])

            if str_examples is not None:
                for str_example in str_examples:
                    input_llm.extend(["### Example:", str_example])

            input_llm.extend(["### Text:", "...", "<|output|>", ""])

            outputs = []
            max_length = int(self.params.get('tokenizer.max_length', 4000))

            for text in texts:
                input_llm[-3] = text
                input_ids = tokenizer(
                    "\n".join(input_llm),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(device)
                output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
                _empty_gpu_cache(device)
                output = output.split("<|output|>")[1].split("<|end-output|>")[0].strip()
                try:
                    outputs.append({} if len(output) == 0 else json.loads(output))
                except Exception as e:
                    outputs.append({"error": str(e), "raw": output})

            return outputs

        except Exception as e:
            raise RuntimeError(f"Error generating outputs: {e}")

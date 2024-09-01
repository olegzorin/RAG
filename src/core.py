import json
import os
import warnings
from typing import List, Any, Dict

import camelot
import pdf2image
import pytesseract
import torch
from PIL.Image import Image
from camelot.core import Table
from camelot.utils import is_url, download_url
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_property, clean_text, get_logger, resolve_path

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="error", category=UserWarning)

logger = get_logger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _get_table_rows(table: Table) -> list[str] | None:
    df_table = table.df.dropna(how="all").loc[:, ~table.df.columns.isin(['', ' '])]
    df_table = df_table.apply(lambda x: x.str.replace("\n", " "))
    df_table = df_table.rename(columns=df_table.iloc[0]).drop(df_table.index[0]).reset_index(drop=True)

    if df_table.shape[0] <= 3 or df_table.eq("").all(axis=None):
        return None

    df_table["summary"] = df_table.apply(
        lambda x: " ".join([f"{col}: {val}, " for col, val in x.items()]),
        axis=1
    )
    return [row["summary"] for ind, row in df_table.iterrows()]


def _read_document(document_id: int, source: str, text_chunker) -> list[str]:
    chunks: list[str] = []
    try:
        if is_url(source):
            source = download_url(source)

        pages: list[Image] = pdf2image.convert_from_path(pdf_path=source, dpi=300)

        for page_no, page in enumerate(pages, 1):
            try:
                text = pytesseract.image_to_string(page)
                chunks.extend(text_chunker.split_text(text=text))
            except UserWarning as w:
                logger.warning(f"Problem reading text in document_id={document_id}, page_no={page_no}: {str(w)}")

        for page_no in range(1, len(pages)):
            try:
                table_list = camelot.read_pdf(filepath=source, pages=str(page_no), suppress_stdout=False, backend='ghostscript')
                for table in table_list:
                    rows = _get_table_rows(table)
                    if rows is not None and len(rows) > 0:
                        chunks.extend(rows)
            except UserWarning as w:
                logger.warning(f"Problem reading tables in document_id={document_id}, page_no={page_no}: {str(w)}")

        return list(map(lambda txt: clean_text(txt), chunks))
    except Exception as e:
        raise RuntimeError(f"Error reading content of documentId={document_id}: {str(e)}")


class CloseableVectorStore(Neo4jVector):

    def __init__(
            self,
            embeddings_model: Embeddings):

        super().__init__(
            url=get_property('neo4j.url'),
            password=get_property('neo4j.password'),
            username=get_property('neo4j.username'),
            embedding=embeddings_model,
            logger=logger
        )

    def reset(self):
        """
        Delete existing data and create a new index
        """
        from neo4j.exceptions import DatabaseError

        self.query(
            f"MATCH (n:`{self.node_label}`) "
            "CALL { WITH n DETACH DELETE n } "
            "IN TRANSACTIONS OF 10000 ROWS;"
        )
        try:
            self.query(f"DROP INDEX {self.index_name}")
        except DatabaseError:  # Index didn't exist yet
            pass

        self.create_new_index()


    def close(self):
        self._driver.close()


class Ragger:
    params: Dict
    llm: ChatOllama
    embeddingsModel: Embeddings
    vectorstore: CloseableVectorStore

    def __init__(self, params: Dict):
        self.params = params

        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=params.get("embeggingsModel", 'all-MiniLM-L6-v2'),
            cache_folder=resolve_path('embeddingsModel.cacheDir', 'caches/embeddings').as_posix()
        )

        self.vectorstore = CloseableVectorStore(self.embeddings_model)

        self.llm = ChatOllama(
            base_url=get_property('ollama.url'),
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


    def get_answers(self, document_id: int, url: str, questions: List[str]) -> List[str]:
        chunks = _read_document(document_id, url, SemanticChunker(embeddings=self.embeddings_model))

        try:
            self.vectorstore.reset()
            self.vectorstore.add_texts(chunks)

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
                    {"context": self.vectorstore.as_retriever(), "question": RunnablePassthrough()}
                    | rag_prompt
                    | self.llm
                    | StrOutputParser()
            )

            return [qa_chain.invoke(question).strip() for question in questions]

        finally:
            self.vectorstore.close()


    def generate_outputs(self, texts: List[str], template: Any, examples: List[Any]) -> List[Any]:
        output_model = self.params.get("output.model", "numind/NuExtract-tiny")
        output_model_cache_dir = resolve_path('outputsModel.cacheDir', 'caches/outputs')

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

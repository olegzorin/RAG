import io
import json
import warnings
from typing import List, Any

import camelot
import torch
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from camelot.core import Table
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import Requests
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from model import ActionRequest, ActionResponse
from pydantic import Json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_int_property, get_property, get_float_property, clean_text, get_logger, get_non_empty_or_none

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="error", category=UserWarning)

logger = get_logger(__name__)


class CloseableVectorStore(Neo4jVector):
    def close(self):
        self._driver.close()


def get_embeddings_model() -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=get_property('ppc.ragagent.embeddings.modelName'),
        cache_folder=get_property('ppc.ragagent.embeddings.cacheDir')
    )


def get_vectorstore(embeddings_model: Embeddings) -> CloseableVectorStore:
    vectorstore = CloseableVectorStore(
        url=get_property('ppc.ragagent.neo4j.url'),
        password=get_property('ppc.ragagent.neo4j.password'),
        username=get_property('ppc.ragagent.neo4j.username'),
        embedding=embeddings_model,
        logger=logger,
        pre_delete_collection=True
    )

    dimension = vectorstore.retrieve_existing_index()
    if dimension is None or dimension[0] is None:
        logger.info(f'Create index name={vectorstore.index_name}')
        vectorstore.create_new_index()

    return vectorstore


def get_table_rows(table: Table) -> list[str] | None:
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


def load_document(document_id: int, url: str, text_chunker) -> list[str]:
    chunks: list[str] = []
    try:
        data = Requests().get(url).content
        pdf_reader = PdfReader(io.BytesIO(data))
        for page_no, page in enumerate(pdf_reader.pages, 1):
            chunks.extend(text_chunker.split_text(text=page.extract_text()))
            try:
                table_list = camelot.read_pdf(url, pages=str(page_no), suppress_stdout=False, backend='ghostscript')
                for table in table_list:
                    rows = get_table_rows(table)
                    if rows is not None and len(rows) > 0:
                        chunks.extend(rows)
            except UserWarning as e:
                logger.warning(f"Problem reading tables in document_id={document_id}, page_no={page_no}: {str(e)}")

            return list(map(clean_text, chunks))
    except Exception as e:
        raise PdfReadError(f"Error reading content of documentId={document_id}: {str(e)}")


def get_llm():
    return ChatOllama(
        base_url=get_property('ppc.ragagent.ollama.url'),
        model=get_property('ppc.ragagent.ollama.model'),
        streaming=True,
        temperature=get_float_property('ppc.ragagent.ollama.temperature', 0.8),
        # Increasing the temperature will make the model answer more creatively. (Default: 0.8)
        seed=get_int_property('ppc.ragagent.ollama.seed', 2),
        # seed should be set for consistent responses
        top_k=get_int_property('ppc.ragagent.ollama.topK', 10),
        # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=get_float_property('ppc.ragagent.ollama.topP', 0.3),
        # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=get_int_property('ppc.ragagent.ollama.numCtx', 3072),
        # Sets the size of the context window used to generate the next token.
        num_predict=get_int_property('ppc.ragagent.ollama.numPredict', 128)
        # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
    )


def generate_outputs(answers: List[str], template: Json[Any], examples: Json[Any]) -> list[str]:
    model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)

    input_llm: list[str] = ["<|input|>"]
    if template is not None:
        input_llm.extend(["### Template:", json.dumps(template, indent=4)])

    if examples is not None:
        input_llm.extend(["### Examples:", json.dumps(examples, indent=4)])

    input_llm.extend(["### Text:", "", "<|output|>"])

    outputs = []
    for answer in answers:
        input_llm[-2] = answer
        input_ids = tokenizer("\n".join(input_llm), return_tensors="pt", truncation=True, max_length=4000)
        output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
        outputs.append(json.loads(output.split("<|output|>")[1].split("<|end-output|>")[0]))

    return outputs


def process_request(req: ActionRequest, resp: ActionResponse):
    if req.documents is None or not req.documents:
        raise Exception('No documents to load')
    if req.questions is None or not req.questions:
        raise Exception('No questions to answer')

    embeddings_model: Embeddings = get_embeddings_model()
    chunks = load_document(req.document_id, req.url, SemanticChunker(embeddings=embeddings_model))

    llm = get_llm()

    vectorstore = get_vectorstore(embeddings_model)
    try:
        vectorstore.add_texts(chunks)

        system_message = req.params.get(
            'chat_prompt_system_message',
            "You are an assistant for question-answering tasks. Do not make up information."
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
                {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
        )

        answers = [qa_chain.invoke(question).strip() for question in req.questions]

        template = get_non_empty_or_none(req.template)
        examples = get_non_empty_or_none(req.examples)

        if template is not None or examples is not None:
            try:
                resp.outputs = generate_outputs(answers, req.template, req.examples)
            except Exception as e:
                raise Exception(f"Error generating outputs: {str(e)}")

        resp.answers = answers

    finally:
        vectorstore.close()

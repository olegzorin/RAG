import io
import logging
import os
import sys
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from jproperties import Properties, PropertyTuple
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.utilities import Requests
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document

from model import RagDocument, RagDocumentResponse, ActionResponse

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

properties = Properties()
props_path = Path(os.environ['PPC_HOME'], 'config', 'properties', 'ragagent.properties')
with open(props_path, 'rb') as props_file:
    properties.load(props_file)


def get_property(key: str) -> str:
    return properties.get(key, PropertyTuple(data=None, meta=None)).data


def get_int_property(key: str, default_value: int) -> int:
    return int(properties.get(key, PropertyTuple(data=default_value, meta=None)).data)


def get_float_property(key: str, default_value: float) -> float:
    return float(properties.get(key, PropertyTuple(data=default_value, meta=None)).data)


def _get_vectorstore(index_name, node_label):
    embedding_model = SentenceTransformerEmbeddings(
        model_name=get_property('ppc.ragagent.embeddings.modelName'),
        cache_folder=get_property('ppc.ragagent.embeddings.cacheDir')
    )

    vectorstore = Neo4jVector(
        url=get_property('ppc.ragagent.neo4j.url'),
        password=get_property('ppc.ragagent.neo4j.password'),
        username=get_property('ppc.ragagent.neo4j.username'),
        index_name=(index_name or 'vector'),
        node_label=(node_label or 'Chunk'),
        embedding=embedding_model
    )

    dimension = vectorstore.retrieve_existing_index()
    if dimension is None:
        logger.info(f'Create index name={vectorstore.index_name}')
        vectorstore.create_new_index()

    return vectorstore


def load_document(rag_document: RagDocument, text_splitter: TextSplitter, vectorstore: Neo4jVector) -> RagDocumentResponse:
    logger.info(f'Load document_id={rag_document.documentId}')

    document_response = RagDocumentResponse()
    document_response.documentId = rag_document.documentId

    try:
        data = Requests().get(rag_document.url).content
        pdf_reader = PdfReader(io.BytesIO(data))

        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()
        chunks = text_splitter.split_text(text=content)

        entries: [Document] = []
        for index, chunk in enumerate(chunks):
            entry = Document(chunk)
            entry.metadata = {'document_id': rag_document.documentId, 'chunk_no': str(index)}
            entries.append(entry)

        vectorstore.add_documents(entries)
        document_response.status = 1

    except PdfReadError as e:
        document_response.errorMessage = f'Error reading document_id={rag_document.documentId}: {str(e)}'

    return document_response


def load_documents(rag_documents: List[RagDocument], index_name, node_label) -> ActionResponse:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=get_int_property('ppc.ragagent.chunkSize', 1000),
        chunk_overlap=get_int_property('ppc.ragagent.chunkOverlap', 200),
        length_function=len
    )

    vectorstore = _get_vectorstore(index_name, node_label)

    response = ActionResponse()
    response.documents = [load_document(rag_document, text_splitter, vectorstore) for rag_document in rag_documents]
    response.success = True
    return response


def answer_questions(questions: List[str], index_name, node_label) -> ActionResponse:
    logger.info("Get question-answering")

    vectorstore = _get_vectorstore(index_name, node_label)

    llm = ChatOllama(
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

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    response = ActionResponse()
    response.answers = [qa.run(question) for question in questions]
    response.success = True
    return response

import json
import logging
import os

import torch

import conf

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter

from transformers import AutoModelForCausalLM, AutoTokenizer

from camelot.utils import is_url, download_url

from neo4j.exceptions import DatabaseError, ClientError
from operator import itemgetter
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Grapher:

    params: Dict
    graph: Neo4jGraph

    def __init__(self, params: Dict):
        self.params = params

        """
        Delete existing data
        """
        self.graph = Neo4jGraph(
            url=conf.get_property('neo4j.url'),
            password=conf.get_property('neo4j.password'),
            username=conf.get_property('neo4j.username'),
            refresh_schema=True
        )
        self.graph.query(
            f"MATCH (n) "
            "CALL { WITH n DETACH DELETE n } "
            "IN TRANSACTIONS OF 10000 ROWS;"
        )
        try:
            self.graph.query(f"DROP INDEX parent_document")
        except DatabaseError:  # Index didn't exist yet
            pass
        try:
            self.graph.query(f"DROP INDEX typical_rag")
        except DatabaseError:  # Index didn't exist yet
            pass


    def get_answers(self, source: str, questions: list[str]):
        if is_url(source):
            source = download_url(source)

        embeddings_model = HuggingFaceEmbeddings(
            model_name=self.params.get("embeggingsModel", 'all-MiniLM-L6-v2'),
            cache_folder=conf.resolve_path('embeddingsModel.cacheDir', 'caches/embeddings').as_posix()
        )
        embedding_dimension = 384

        parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        child_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=24)

        parent_documents = PyMuPDFLoader(source).load_and_split(parent_splitter)

        for i, parent in enumerate(parent_documents):
            child_documents = child_splitter.split_documents([parent])
            params = {
                "parent_text": parent.page_content,
                "parent_id": i,
                "parent_embedding": embeddings_model.embed_query(parent.page_content),
                "children": [
                    {
                        "text": c.page_content,
                        "id": f"{i}-{ic}",
                        "embedding": embeddings_model.embed_query(c.page_content),
                    }
                    for ic, c in enumerate(child_documents)
                ],
            }
            # Ingest data
            self.graph.query(query=
                """
            MERGE (p:Parent {id: $parent_id})
            SET p.text = $parent_text
            WITH p
            CALL db.create.setVectorProperty(p, 'embedding', $parent_embedding)
            YIELD node
            WITH p 
            UNWIND $children AS child
            MERGE (c:Child {id: child.id})
            SET c.text = child.text
            MERGE (c)<-[:HAS_CHILD]-(p)
            WITH c, child
            CALL db.create.setVectorProperty(c, 'embedding', child.embedding)
            YIELD node
            RETURN count(*)
            """,
               params=params
            )
            # Create vector index for child
            try:
                self.graph.query(
                    "CALL db.index.vector.createNodeIndex('parent_document', "
                    "'Child', 'embedding', $dimension, 'cosine')",
                    {"dimension": embedding_dimension},
                )
            except ClientError:  # already exists
                pass
            # Create vector index for parents
            try:
                self.graph.query(
                    "CALL db.index.vector.createNodeIndex('typical_rag', "
                    "'Parent', 'embedding', $dimension, 'cosine')",
                    {"dimension": embedding_dimension},
                )
            except ClientError:  # already exists
                pass

        parent_query = """
        MATCH (node)<-[:HAS_CHILD]-(parent)
        WITH parent, max(score) AS score // deduplicate parents
        RETURN parent.text AS text, score, {} AS metadata LIMIT 1
        """

        parent_vectorstore = Neo4jVector.from_existing_index(
            embedding=embeddings_model,
            url=conf.get_property('neo4j.url'),
            password=conf.get_property('neo4j.password'),
            username=conf.get_property('neo4j.username'),
            index_name="parent_document",
            retrieval_query=parent_query,
        )

        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        retriever = parent_vectorstore.as_retriever()

        llm = ChatOllama(
            base_url=conf.get_property('ollama.url'),
            streaming=True,
            model=self.params.get('ollama.model', 'llama2'),
            temperature=float(self.params.get('ollama.temperature', 0.0)),
            # Increasing the temperature will make the model answer more creatively. (Default: 0.8)
            seed=int(self.params.get('ollama.seed', 2)),
            # seed should be set for consistent responses
            top_k=int(self.params.get('ollama.top_k', 10)),
            # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=float(self.params.get('ollama.top_p', 0.3)),
            # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=int(self.params.get('ollama.num_ctx', 3072)),
            # Sets the size of the context window used to generate the next token.
            num_predict=int(self.params.get('ollama.num_predict', -2))
            # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            RunnableParallel(
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        return [chain.invoke({'question':question}) for question in questions]


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

import logging
import os
from operator import itemgetter
from typing import Dict

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter
from neo4j.exceptions import DatabaseError, ClientError

import conf
import reader
from core import RAG

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class GraphSearch(RAG):

    graph: Neo4jGraph

    def __init__(
            self,
            params: Dict
        ):
        super().__init__(params)

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

        embedding_dimension = 384
        # Create vector index for child
        try:
            self.graph.query(
                "CALL db.index.vector.createNodeIndex('parent_document', "
                "'Child', 'embedding', $dimension, 'cosine')",
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            raise

        # Create vector index for parents
        try:
            self.graph.query(
                "CALL db.index.vector.createNodeIndex('typical_rag', "
                "'Parent', 'embedding', $dimension, 'cosine')",
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            raise


    def get_answers(self, document_id: int, source: str, questions: list[str]) -> list[str]:

        parent_chunkers = [SemanticChunker(embeddings=self.embeddings_model), TokenTextSplitter(chunk_size=512, chunk_overlap=24)]
        child_chunkers = [SemanticChunker(embeddings=self.embeddings_model), TokenTextSplitter(chunk_size=100, chunk_overlap=24)]

        convert_pdf_to_image = self.params.get("reader.convertPdf2Image", "false") == "true"
        print(f'convert_pdf_to_image={convert_pdf_to_image}')
        read_tables = self.params.get("reader.read_tables", "false") == "true"
        print(f'read_tables={read_tables}')
        parent_chunks = reader.read_pdf(document_id, source, parent_chunkers, read_tables=read_tables, convert_pdf_to_image=convert_pdf_to_image)

        print(f"parent chunks {len(parent_chunks)}")

        # with open('CCR1.txt', 'w') as f:
        #     f.write("\n\n".join([d.page_content for d in parent_documents]))

        child_counts: [int] = []
        for i, parent_chunk in enumerate(parent_chunks):
            child_chunks = []
            for chunker in child_chunkers:
                child_chunks.extend(chunker.split_text(parent_chunk))

            child_counts.append(len(child_chunks))
            params = {
                "parent_text": parent_chunk,
                "parent_id": i,
                "parent_embedding": self.embeddings_model.embed_query(parent_chunk),
                "children": [
                    {
                        "text": child_chunk,
                        "id": f"{i}-{ic}",
                        "embedding": self.embeddings_model.embed_query(child_chunk),
                    }
                    for ic, child_chunk in enumerate(child_chunks)
                ],
            }
            # Ingest data
            self.graph.query(query=
                """
            MERGE (p:Parent {id: $parent_id})
            SET p.text = $parent_text
            WITH p
            CALL db.create.setNodeVectorProperty(p, 'embedding', $parent_embedding)
            WITH p 
            UNWIND $children AS child
            MERGE (c:Child {id: child.id})
            SET c.text = child.text
            MERGE (c)<-[:HAS_CHILD]-(p)
            WITH c, child
            CALL db.create.setNodeVectorProperty(c, 'embedding', child.embedding)
            RETURN count(*)
            """,
               params=params
            )

        print(f'child counts: {child_counts}')

        parent_query = """
        MATCH (node)<-[:HAS_CHILD]-(parent)
        WITH parent, max(score) AS score // deduplicate parents
        RETURN parent.text AS text, score, {} AS metadata LIMIT 1
        """

        parent_vectorstore = Neo4jVector.from_existing_index(
            embedding=self.embeddings_model,
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
            | self.llm
            | StrOutputParser()
        )

        answers = []
        for question in questions:
            answers.append(chain.invoke({'question':question}))

        return answers

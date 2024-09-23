import logging
import os
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter
from neo4j.exceptions import DatabaseError, ClientError

import conf
from core import RagSearch, MAX_CHUNK_SIZE
from reader import ExtractedDoc

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class ParentChunker:
    s: SemanticChunker

    def __init__(
            self,
            embeddings: Embeddings,
            number_of_chunks: int
    ):
        self.s = SemanticChunker(
            embeddings=embeddings,
            number_of_chunks=number_of_chunks
        )

    def __call__(self, text: str) -> list[str]:
        return [chunk for chunk in self.s.split_text(text) if len(chunk) <= MAX_CHUNK_SIZE]


class ChildChunker:
    t: TokenTextSplitter
    r: RecursiveCharacterTextSplitter

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.t = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        self.r = RecursiveCharacterTextSplitter(chunk_size=(chunk_size * 3) // 4, chunk_overlap=chunk_overlap, length_function=len)

    def __call__(self, text: str) -> list[str]:
        chunks = []
        chunks.extend([chunk for chunk in self.t.split_text(text) if len(chunk) <= MAX_CHUNK_SIZE])
        chunks.extend([chunk for chunk in self.r.split_text(text) if len(chunk) <= MAX_CHUNK_SIZE])
        return chunks


class GraphSearch(RagSearch):
    graph: Neo4jGraph

    def __init__(
            self,
            params: Dict
    ):
        super().__init__(params)

        logger.info('GraphSearch!')

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

    def _create_indexes(self):
        # Create vector index for child
        try:
            self.graph.query(
                "CALL db.index.vector.createNodeIndex('parent_document', "
                "'Child', 'embedding', $dimension, 'cosine')",
                {"dimension": self.embeddings_dimension},
            )
        except ClientError:  # already exists
            pass

        # Create vector index for parents
        try:
            self.graph.query(
                "CALL db.index.vector.createNodeIndex('typical_rag', "
                "'Parent', 'embedding', $dimension, 'cosine')",
                {"dimension": self.embeddings_dimension},
            )
        except ClientError:  # already exists
            pass

    def get_answers(
            self,
            document: ExtractedDoc,
            questions: list[str]
    ) -> list[str]:

        chunk_size = int(self.params.get("reader.chunk_size", 384))
        chunk_overlap = int(self.params.get("reader.chunk_overlap", chunk_size // 8))

        parent_chunker = ParentChunker(
            embeddings=self.embeddings_model,
            number_of_chunks=document.get_content_length() // chunk_size
        )
        child_chunker = ChildChunker(
            chunk_size=chunk_size // 3,
            chunk_overlap=chunk_overlap // 3
        )

        parent_chunks = document.split_into_chucks(parent_chunker)
        logger.debug(f"parent chunks {len(parent_chunks)}")

        child_counts: [int] = []
        for i, parent_chunk in enumerate(parent_chunks):
            child_chunks = child_chunker(parent_chunk)

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
            self.graph.query(
                query="""
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

            self._create_indexes()

        logger.debug(f'child counts: {child_counts}')

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

        return self._retrieve_answers(
            retriever=parent_vectorstore.as_retriever(),
            questions=questions
        )

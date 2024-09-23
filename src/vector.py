import logging
import os
from typing import Dict

from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter

from core import RagSearch, MAX_CHUNK_SIZE
from db import Neo4jDB
from reader import ExtractedDoc

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class Chunker:
    s: SemanticChunker
    t: TokenTextSplitter
    r: RecursiveCharacterTextSplitter

    def __init__(
            self,
            embeddings: Embeddings,
            text_size: int,
            chunk_size: int,
            chunk_overlap: int
    ):
        self.s = SemanticChunker(
            embeddings=embeddings,
            number_of_chunks=text_size//chunk_size
        )
        self.t = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.r = RecursiveCharacterTextSplitter(
            chunk_size=(chunk_size * 3) // 4,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def __call__(
            self,
            text: str
    ) -> list[str]:
        chunks = []
        chunks.extend([chunk for chunk in self.s.split_text(text) if len(chunk) <= MAX_CHUNK_SIZE])
        chunks.extend([chunk for chunk in self.t.split_text(text) if len(chunk) <= MAX_CHUNK_SIZE])
        chunks.extend([chunk for chunk in self.r.split_text(text) if len(chunk) <= MAX_CHUNK_SIZE])
        return chunks


class SearchType:
    SIMILARITY = "similarity"
    BM25 = "bm25"
    MMR = "mmr"
    values = [SIMILARITY, BM25, MMR]


class VectorSearch(RagSearch):

    search_types: list[str]

    def __init__(self, params: Dict):
        super().__init__(params)

        logger.info('VectorSearch!')

        self.search_types = params.get("search.type", SearchType.SIMILARITY).split(',')
        logger.info(f'Search types: {self.search_types}')

        for search_type in self.search_types:
            if search_type not in SearchType.values:
                raise TypeError(f"Invalid search type '{search_type}', valid types: {SearchType.values}")



    def get_answers(
            self,
            document: ExtractedDoc,
            questions: list[str]
    ) -> list[str]:

        text_size = document.get_content_length()
        chunk_size = int(self.params.get("reader.chunk_size", 384))
        chunk_overlap = int(self.params.get("reader.chunk_overlap", chunk_size//8))

        chunker = Chunker(
            embeddings=self.embeddings_model,
            text_size=text_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = document.split_into_chucks(chunker)

        vectorstore = None

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
                        vectorstore = Neo4jDB(embeddings_model=self.embeddings_model)
                        logging.getLogger("neo4j.pool").setLevel(logging.WARNING)
                        vectorstore.add_texts(chunks)

                    retrievers.append(vectorstore.as_retriever(search_type=search_type))

            return self._retrieve_answers(
                retriever=EnsembleRetriever(retrievers=retrievers),
                questions=questions
            )
        except Exception as e:
            logger.exception(e)
            raise e

        finally:
            if vectorstore is not None:
                vectorstore.close()

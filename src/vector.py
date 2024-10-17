import logging
import os
from typing import Dict

from langchain.retrievers import EnsembleRetriever

from core import RagSearch, SemanticSplitter, ParagraphSplitter
from db import Neo4jDB
from reader import PdfDoc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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
            document: PdfDoc,
            questions: list[str]
    ) -> list[str]:

        chunk_size = int(self.params.get("reader.chunk_size", 384))

        chunks = document.split_into_chucks(
            chunkers=[
                SemanticSplitter(
                    embeddings=self.embeddings_model,
                    chunk_size=chunk_size
                )#,
                # ParagraphSplitter()
            ]
        )

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

                    retrievers.append(
                        vectorstore.as_retriever(
                            search_type=search_type,
                            search_kwargs={'k': 3, 'fetch_k': 30}
                        )
                    )

            return self._retrieve_answers(
                retriever=EnsembleRetriever(retrievers=retrievers),
                questions=list(map(lambda s: s.lower(), questions))
            )
        except Exception as e:
            logger.exception(e)
            raise e

        finally:
            if vectorstore is not None:
                vectorstore.close()

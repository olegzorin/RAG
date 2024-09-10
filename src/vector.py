import logging
import os
from typing import List, Dict

from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import reader
from core import RagSearch
from db import Neo4jDB

logger = logging.getLogger(__name__)

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

        print('VectorSearch!')

        self.search_types = params.get("search.type", SearchType.BM25).split(',')
        print(f'Search types: ', self.search_types)
        for search_type in self.search_types:
            if search_type not in SearchType.values:
                raise TypeError(f"Invalid search type '{search_type}', valid types: {SearchType.values}")



    def get_answers(self, document_id: int, url: str, questions: List[str]) -> list[str]:
        chunk_size = int(self.params.get("reader.chunk_size", 384))
        chunk_overlap = int(self.params.get("reader.chunk_overlap", chunk_size//8))
        chunks = reader.read_pdf(
            document_id,
            source=url,
            text_chunker=self.get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            read_tables=True,
            convert_pdf_to_image=True
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

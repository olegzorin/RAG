import logging
import os
from typing import Dict

from keybert import KeyBERT
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from core import RagSearch, SemanticSplitter
from db import Neo4jDB
from reader import ExtractedDoc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class KeywordSearch(RagSearch):
    def __init__(
            self,
            params: Dict
    ) -> None:
        super().__init__(params)
        self.kw_model = KeyBERT(self.params.get('keybert.model', 'all-MiniLM-L6-v2'))
        logger.info('KeywordSearch!')

    def get_answers(
            self,
            document: ExtractedDoc,
            questions: list[str]
    ) -> list[str]:
        def preprocessing_upper_func(
                text: str
        ) -> list[str]:
            return text.upper().split()

        chunker = SemanticSplitter(
            embeddings=self.embeddings_model,
            chunk_size=500
        )

        chunks = document.split_into_chucks([chunker])

        # initialize the bm25 retriever and faiss retriever
        bm25_retriever = BM25Retriever.from_texts(
            texts=chunks,
            preprocess_func=preprocessing_upper_func
        )
        bm25_retriever.k = 20

        from langchain.prompts import ChatPromptTemplate

        template = """<human>: Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':
        
        ### CONTEXT
        {context}
        
        ### QUESTION
        Question: {question}
        
        \n
        
        <bot>:
        """

        prompt_qa = ChatPromptTemplate.from_template(template)

        answers = []
        for question in questions:
            logger.info(f'Q: {question}')
            keywords = self.kw_model.extract_keywords(
                docs=str(question),
                keyphrase_ngram_range=(1, 1),
                stop_words='english',
                use_mmr=True,
                diversity=0.7
            )

            sanitized_query = ' '.join([key.upper() for key, _ in keywords])
            logger.info(f'Keywords: {sanitized_query}')

            retrieved_docs: list[Document] = bm25_retriever.get_relevant_documents(sanitized_query)

            retrieved_texts = [doc.page_content for doc in retrieved_docs]

            # print("retrieved_docs=",retrieved_docs)

            retrieved_texts.extend(document.get_table_rows())

            # Initialize vector store and document store
            vectorstore = Neo4jDB(
                embeddings_model=self.embeddings_model
            )
            try:
                vectorstore.add_texts(retrieved_texts)

                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

                chain_ensemble_with_dragon = (
                        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
                        | prompt_qa
                        | self.llm
                        | StrOutputParser()
                )

                ensemble_with_dragon_result = chain_ensemble_with_dragon.invoke({'question': question})
            finally:
                if vectorstore is not None:
                    vectorstore.close()

            answers.append(str(ensemble_with_dragon_result))

        return answers

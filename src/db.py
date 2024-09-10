import logging

from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings

from conf import get_property


# import chromadb.config


# class VectorDB(VectorStore, ABC):
#
#     CHROMA: str = 'chroma'
#     NEO4J: str = 'neo4j'
#
#     @abstractmethod
#     def close(self):
#         pass
#
#     @classmethod
#     def get_instance(cls, name: str, embeddings_model: Embeddings):
#         if name == cls.CHROMA:
#             from langchain_community.vectorstores import Chroma
#             return ChromaDB(embeddings_model)
#         elif name == cls.NEO4J:
#             return Neo4jDB(embeddings_model)
#         else:
#             raise Exception("Wrong RagDB name")
#

class Neo4jDB(Neo4jVector):
    def __init__(
            self,
            embeddings_model: Embeddings
    ) -> None:

        super().__init__(
            url=get_property('neo4j.url'),
            password=get_property('neo4j.password'),
            username=get_property('neo4j.username'),
            embedding=embeddings_model,
            pre_delete_collection=True,
            logger=logging.getLogger('Neo4jDB')
        )
        self.create_new_index()

    def close(self):
        self._driver.close()


# class ChromaDB(Chroma, VectorDB):
#
#     def __init__(
#             self,
#             embeddings_model: Embeddings
#     ) -> None:
#         super().__init__(
#             embedding_function=embeddings_model,
#             client_settings=chromadb.config.Settings(
#                 anonymized_telemetry=False,
#                 is_persistent=False
#             )
#         )
#
#     def close(self):
#         pass

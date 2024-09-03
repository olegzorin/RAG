import logging
from abc import ABC, abstractmethod

import chromadb.config
from langchain_community.vectorstores import Neo4jVector, Chroma

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from conf import get_property


class DB(VectorStore, ABC):

    CHROMA: str = 'chroma'
    NEO4J: str = 'neo4j'

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @classmethod
    def get_instance(cls, name: str, embeddings_model: Embeddings):
        if name == cls.CHROMA:
            return ChromaDB(embeddings_model)
        elif name == cls.NEO4J:
            return Neo4jDB(embeddings_model)
        else:
            raise Exception("Wrong RagDB name")


class Neo4jDB(Neo4jVector, DB):
    def __init__(
            self,
            embeddings_model: Embeddings
    ) -> None:

        super().__init__(
            url=get_property('neo4j.url'),
            password=get_property('neo4j.password'),
            username=get_property('neo4j.username'),
            embedding=embeddings_model,
            logger=logging.getLogger('Neo4jDB')
        )

    def reset(self):
        """
        Delete existing data and create a new index
        """
        from neo4j.exceptions import DatabaseError

        self.query(
            f"MATCH (n:`{self.node_label}`) "
            "CALL { WITH n DETACH DELETE n } "
            "IN TRANSACTIONS OF 10000 ROWS;"
        )
        try:
            self.query(f"DROP INDEX {self.index_name}")
        except DatabaseError:  # Index didn't exist yet
            pass

        self.create_new_index()

    def close(self):
        self._driver.close()


class ChromaDB(Chroma, DB):

    def __init__(
            self,
            embeddings_model: Embeddings
    ) -> None:
        super().__init__(
            embedding_function=embeddings_model,
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=False
            )
        )

    def reset(self):
        pass

    def close(self):
        pass

import logging

import chromadb.config
from langchain_community.vectorstores import Neo4jVector, Chroma
from langchain_core.embeddings import Embeddings

from utils import get_property


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
            logger=logging.getLogger(__name__)
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


class ChromaDB(Chroma):
    def __init__(
            self,
            embeddings_model: Embeddings
    ) -> None:
        super().__init__(
            embedding_function=embeddings_model,
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=False,
                persist_directory="./chroma"
            )
        )

    def reset(self):
        pass

    def close(self):
        pass

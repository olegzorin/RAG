from langchain_community.vectorstores import Neo4jVector
from utils import get_property
from langchain_core.embeddings import Embeddings


class VStore(Neo4jVector):

    def __new__(cls, *args, **kwargs):
        pass


    def __init__(self, embeddings_model: Embeddings, logger):
        super(VStore, self).__init__(
            url=get_property('ppc.ragagent.neo4j.url'),
            password=get_property('ppc.ragagent.neo4j.password'),
            username=get_property('ppc.ragagent.neo4j.username'),
            embedding=embeddings_model,
            logger=logger,
            pre_delete_collection=False
        )


from typing import Optional, LiteralString, cast, Any

import neo4j
from neo4j.exceptions import CypherSyntaxError
from conf import get_property

url = get_property('neo4j.url')
password = get_property('neo4j.password')
username = get_property('neo4j.username')

driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
database="neo4j"
driver.verify_connectivity()

def create_database():
    script = open('create_movie.query', 'r').read()
    queries = [f'CREATE {clause}' for clause in script.split('CREATE ') if len(clause.strip()) > 0]
    with driver.session(database=database) as session:
        for query in queries:
            print(query)
            session.execute_write(
                lambda tx: tx.run(query))

def run_query(query: str, params: Optional[dict] = None) -> list[Any]:
    with driver.session(database=database) as session:
        try:
            data: neo4j.Result = session.run(cast(LiteralString, query), params)
            return [r.data() for r in data]
        except CypherSyntaxError as e:
            raise ValueError(f"Cypher Statement is not valid\n{e}")


# create_database()

# run_query("CREATE (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix)")
# print(run_query("MATCH (m:Movie {title:'The Matrix'})<-[r:ACTED_IN]-(p:Person) RETURN m.title,  p.name"))
# print(run_query("MATCH (p:Person) return p.name"))

res = run_query("match (p:Person) return p.name order by p.name")
for i, r in enumerate(res):
    print(i, r)

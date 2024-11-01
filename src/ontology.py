import shutil

import nest_asyncio
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

import re
from conf import MODEL_CACHE_DIR, get_property, DOCS_CACHE_DIR
from pdf_document import PdfDoc

nest_asyncio.apply()

## ontology
from typing import Literal
from llama_index.llms.ollama import Ollama

# best practice to use upper-case
# - contract
# - business_line
# - facility
# - level
# - claim
# - reimbursement
# - medication
entities = Literal[
    "CONTRACT", "BUSINESS_LINE", "ORGANIZATION", "FACILITY", "LEVEL", "CLAIM", "REIMBURSEMENT",
    "MEDICATION", "EFFECTIVE_DATE", "DURATION", "NOTICE_PERIOD", "REVENUE_CODE", "SERVICES",
    "SERVICE_EXCLUSION", "ADDRESS"
]
enitites_names = [
    "CONTRACT", "BUSINESS_LINE", "ORGANIZATION", "FACILITY", "LEVEL", "CLAIM", "REIMBURSEMENT",
    "MEDICATION", "EFFECTIVE_DATE", "DURATION", "NOTICE_PERIOD", "REVENUE_CODE", "SERVICES",
    "SERVICE_EXCLUSION", "ADDRESS"
]

# - has_effective_date
# - has_term
# - has_autorenewal_period
# - has_termination_notice
# - has_claims_submission_period
# - has_claims_appeal_period
# - has_claims_payment_timeline
# - includes_business_line
# - includes_facility
# - has_reimbursement_structure
# - has_level
# - has_revenue_code
# - has_services
# - excludes
# - has_high_cost_medication_provision
# - requires_for_reimbursement

relations = Literal[
    "HAS_EFFECTIVE_DATE", "HAS_TERM", "HAS_AUTORENEWAL_PERIOD", "HAS_TERMINATION_NOTICE",
    "HAS_CLAIM_SUBMISSION_PERIOD", "HAS_CLAIM_APPEAL_PERIOD", "HAS_CLAIMS_PAYMENT_TIMELINE",
    "INCLUDES_BUSINESS_LINE", "INCLUDES_FACILITY", "HAS_REIMBURSEMENT_STRUCTURE", "HAS_LEVEL",
    "HAS_REVENUE_CODE", "HAS_SERVICES", "EXCLIDES", "HAS_HIGH_COST_MEDICATION_PROVISION",
    "REQUIRES_FOR_REIMBURSEMENT"
]
relations_names = [
    "HAS_EFFECTIVE_DATE", "HAS_TERM", "HAS_AUTORENEWAL_PERIOD", "HAS_TERMINATION_NOTICE",
    "HAS_CLAIM_SUBMISSION_PERIOD", "HAS_CLAIM_APPEAL_PERIOD",
    "HAS_CLAIMS_PAYMENT_TIMELINE", "INCLUDES_BUSINESS_LINE", "INCLUDES_FACILITY",
    "HAS_REIMBURSEMENT_STRUCTURE", "HAS_LEVEL",
    "HAS_REVENUE_CODE", "HAS_SERVICES", "EXCLIDES", "HAS_HIGH_COST_MEDICATION_PROVISION",
    "REQUIRES_FOR_REIMBURSEMENT"
]

validation_schema = [
    ("CONTRACT", "HAS_EFFECTIVE_DATE", "EFFECTIVE_DATE"),
    ("CONTRACT", "HAS_TERM", "DURATION"),
    ("CONTRACT", "HAS_AUTORENEWAL_PERIOD", "DURATION"),
    ("CONTRACT", "HAS_TERMINATION_NOTICE", "NOTICE_PERIOD"),
    ("CLAIM", "HAS_CLAIMS_SUBMISSION_PERIOD", "DURATION"),
    ("CLAIM", "HAS_CLAIMS_APPEAL_PERIOD", "DURATION"),
    ("CLAIM", "HAS_CLAIMS_PAYMENT_TIMELINE", "DURATION"),
    ("CONTRACT", "INCLUDES_BUSINESS_LINE", "BUSINESS_LINE"),
    ("CONTRACT", "INCLUDES_FACILITY", "FACILITY"),
    ("CONTRACT", "HAS_REIMBURSEMENT_STRUCTURE", "REIMBURSEMENT"),
    ("REIMBURSEMENT", "HAS_LEVEL", "LEVEL"),
    ("LEVEL", "HAS_REVENUE_CODE", "REVENUE_CODE"),
    ("LEVEL", "HAS_SERVICES", "SERVICES"),
    ("CONTRACT", "EXCLUDES", "SERVICE_EXCLUSION"),
    ("CONTRACT", "HAS_HIGH_COST_MEDICATION_PROVISION", "MEDICATION"),
    ("MEDICATION", "REQUIRES_FOR_REIMBURSEMENT", "REQUIREMENTS"),
    ("CONTRACT", "INCLUDES_FACILITY", "FACILITY"),
    ("CONTRACT", "HAS_CLAIMS_SUBMISSION_ADDRESS", "ADDRESS"),
]

# best so far
llm = Ollama(
    model='mistral:7b-instruct-v0.3-q2_K',
    context_window=2048,
    temperature=0.1,
    request_timeout=300.0,
    additional_kwargs={"num_predict": 256, "num_ctx": 2048}
)
# llm = Ollama(model="llama3:instruct",  request_timeout=3600)
# llm = Ollama(model="eas/dragon-yi-v0:latest", json_mode=True, request_timeout=3600)
# llm = Ollama( model='vaibhav-s-dabhade/phi-3.5-mini-instruct', context_window=2048, temperature=0.1, request_timeout=300.0,additional_kwargs={"num_predict": 256, "num_ctx": 2048})
# llm = Ollama( model='eas/dragon-mistral-v0:latest', context_window=2048, temperature=0.1, request_timeout=300.0,additional_kwargs={"num_predict": 256, "num_ctx": 2048})


from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)
from typing import List, Tuple
import json


def parse_dynamic_triplets(
        llm_output: str
) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.
    This function is flexible and can handle various output formats.

    Args:
        llm_output (str): The output from the LLM, which may be JSON-like or plain text.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of triplets.
    """
    triplets = []
    print("llm_output = ", llm_output)

    try:
        # Attempt to parse the output as JSON
        data = json.loads(llm_output)
        for item in data:
            head = item.get("head", "").strip()  # Strip whitespace
            head_type = item.get("head_type", "").strip()
            relation = item.get("relation", "").strip()
            tail = item.get("tail", "").strip()  # Strip whitespace
            tail_type = item.get("tail_type", "").strip()

            # Ensure none of the key components are empty after stripping
            if head and head_type and relation and tail and tail_type:
                head_node = EntityNode(name=head, label=head_type)
                tail_node = EntityNode(name=tail, label=tail_type)
                relation_node = Relation(
                    source_id=head_node.id, target_id=tail_node.id, label=relation
                )
                triplets.append((head_node, relation_node, tail_node))

    except json.JSONDecodeError:
        # Flexible pattern to match the key-value pairs for head, head_type, relation, tail, and tail_type
        pattern = r'[\{"\']head[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']head_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']relation[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']tail[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']tail_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']'

        # Find all matches in the output
        matches = re.findall(pattern, llm_output)

        for match in matches:
            head, head_type, relation, tail, tail_type = match

            # Strip whitespace from the values
            head = head.strip()
            head_type = head_type.strip()
            relation = relation.strip()
            tail = tail.strip()
            tail_type = tail_type.strip()

            # Ensure none of the key components are empty after stripping
            if head and head_type and relation and tail and tail_type:
                head_node = EntityNode(name=head, label=head_type)
                tail_node = EntityNode(name=tail, label=tail_type)
                relation_node = Relation(
                    source_id=head_node.id, target_id=tail_node.id, label=relation
                )
                triplets.append((head_node, relation_node, tail_node))

    return triplets


kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    parse_fn=parse_dynamic_triplets,
    max_triplets_per_chunk=20,
    num_workers=4,
    allowed_entity_types=enitites_names,
    allowed_relation_types=relations_names,
)

graph_store = Neo4jPropertyGraphStore(
    url=get_property('neo4j.url'),
    password=get_property('neo4j.password'),
    username=get_property('neo4j.username')
)

vec_store = None
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device='mps',
    cache_folder=MODEL_CACHE_DIR  # device='cuda')
)

#########
from llama_index.core.node_parser import SemanticSplitterNodeParser

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)

document_id = 1
shutil.copy(
    src='../docs/CCR.json',
    dst=f'{DOCS_CACHE_DIR}/{document_id}.json'
)

from llama_index.core.schema import Document

document = Document(text=PdfDoc.load(document_id=document_id).get_content())

nodes = splitter.get_nodes_from_documents([document])

index = PropertyGraphIndex(
    nodes,
    kg_extractors=[kg_extractor],
    embed_model=embed_model,
    property_graph_store=graph_store,
    vector_store=vec_store,
    show_progress=True,
)

# Close the neo4j connection explicitly.
graph_store.close()

################


from llama_index.core.indices.property_graph import VectorContextRetriever

from llm_custom_synonym import LLMCustomSynonymRetriever
from llama_index.llms.ollama import Ollama

prompt = (
    "Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
    "Note, result should be in one-line, separated by '^' symbols."
    "----\n"
    "QUERY: {query_str}\n"
    "----\n"
    "KEYWORDS: "
)


def parse_fn(llm_output: str) -> list:
    # Use a regular expression to extract the part with ^ delimiters
    print("llm_output =", llm_output)

    pattern = r'([^\n]+(?:\^[^\n]+)+)'
    match = re.search(pattern, llm_output)

    if match:

        # Extract the matched portion inside the quotes (group 1)
        keywords_str = match.group(1).replace('"', '')

        # Split the string by ^, capitalize each part, and replace spaces with underscores
        keywords_list = [keyword.strip().upper().replace(" ", "_") for keyword in keywords_str.split("^") if
                         keyword.strip()]

        print('keywords_list=', keywords_list)

        return keywords_list
    else:
        # Return an empty list if no pattern matches
        return []


synonym_retriever = LLMCustomSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    # include source chunk text with retrieved paths
    include_text=True,
    synonym_prompt=prompt,
    output_parsing_fn=parse_fn,
    max_keywords=10,
    # the depth of relations to follow after node retrieval
    path_depth=3,
)

retriever = index.as_retriever(sub_retrievers=[synonym_retriever])
nodes = retriever.retrieve("What is the effective date of the agreement?")
for node in nodes:
    print(node.text)

retriever = index.as_retriever(sub_retrievers=[synonym_retriever])
nodes = retriever.retrieve("What is the effective date of the agreement?")
i = 0
for node in nodes:
    i += 1
    print("node#=", i)
    print(node.text)

# llm_synonym = LLMSynonymRetriever(
#     index.property_graph_store,
#     llm=llm,
#     include_text=False,
# )
vector_context = VectorContextRetriever(
    index.property_graph_store,
    embed_model=embed_model,
    include_text=True,
)

print("Vector retriever")

retriever = index.as_retriever(sub_retrievers=[vector_context])
nodes = retriever.retrieve("What is the effective date of the agreement?")
i = 0
for node in nodes:
    i += 1
    print("node#=", i)
    print(node.text)

retriever = index.as_retriever(
    sub_retrievers=[
        synonym_retriever,
        vector_context,
    ]
)

from llama_index.core import PromptTemplate

text_qa_template_str = """<human>: Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context_str}

### QUESTION
Question: {query_str}

\n

<bot>:
"""
# text_qa_template_str = (
#     "Context information is"
#     " below.\n---------------------\n{context_str}\n---------------------\nUsing"
#     " both the context information and also using your own knowledge, answer"
#     " the question: {query_str}\nIf the context isn't helpful, you can also"
#     " answer the question on your own.\n"
# )

text_qa_template = PromptTemplate(text_qa_template_str)

llm_query = Ollama(
    model="eas/dragon-yi-v0:latest",
    request_timeout=3600
)
query_engine = index.as_query_engine(
    text_qa_template=text_qa_template,
    sub_retrievers=[
        synonym_retriever,
        vector_context,
    ],
    llm=llm_query,
)

# query_engine = index.as_query_engine(
#     sub_retrievers=[
#         llm_synonym,
#         vector_context,
#     ],
#     llm=Ollama(model="llama3", request_timeout=3600),
# )

print("Final retriever")
questions = [
    "What is the effective date of the agreement?",
    "How many days notice is needed to terminate without cause?",
    "What is the claims submission period in days?",
    "What is the claims payment timeline?",
    "What lines of business are included within the agreement?",
    "What is the reimbursement structure?",
    "If leveled plan, what are the per diem rates for each level?"
]
outputs = {}
for question in questions:

    print("Combined retriever")

    nodes = retriever.retrieve(question)

    for node in nodes:
        print(node.text)

    response = query_engine.query(question)
    # print("final response =",str(response))
    outputs[question] = str(response)

# Print the final outputs dictionary
print("final outputs =", outputs)

# Close the neo4j connection explicitly.
graph_store.close()

'''
llm_output =  if_leveled_plan^per_diem^rates^for_each_level^per_level^payment^wage^compensation^rate^remuneration^salary^stipend^wages^hourly_rate^daily_rate
keywords_list= ['IF_LEVELED_PLAN', 'PER_DIEM', 'RATES', 'FOR_EACH_LEVEL', 'PER_LEVEL', 'PAYMENT', 'WAGE', 'COMPENSATION', 'RATE', 'REMUNERATION', 'SALARY', 'STIPEND', 'WAGES', 'HOURLY_RATE', 'DAILY_RATE']
final outputs = {'What is the effective date of the agreement?': 'Not Found.', 'How many days notice is needed to terminate without cause?': 'Not Found', 'What is the claims submission period in days?': 'Not Found.', 'What is the claims payment timeline?': 'Not Found.', 'What lines of business are included within the agreement?': 'This Agreement shall apply only to Health Care Services provided by Provider, pursuant to this agreement between ChoiceCare and Provider.', 'What is the reimbursement structure?': "It depends on whether ChoiceCare has authorized a specific service per Humana's Provider Manual or its successor manual.", 'If leveled plan, what are the per diem rates for each level?': 'Not Found.'}

'''

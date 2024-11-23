import os
import re

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

content = '''
LEVEL 1 $395 Revenue Code: 191.

Basic Services - Level I services include but are not limited to:  A. Room and board  B. Twenty-four (24) hour nursing care  C. Prepared meals with/without special diet  D. Laboratory  E. Radiology  F. Medication administration  G. Medications unless otherwise excluded*  H. IV administration and maintenance, including PICC lines, solutions, equipment, pumps, and supplies  I. Medical/disposable supplies  J. Discharge planning  K. Therapy; up to 1.5 hours of combined physical therapy (PT), Occupational Therapy (OT), and Speech Therapy (ST) per day at a minimum of six days a week; all therapy services shall be performed by a registered Therapist or an Assistant Therapist as appropriate L. Stage 2 wound care  M. Decomposition in functional status due to chronic illness COPD, CHF  N. Contact Isolation.

LEVEL 3 $640 Revenue Code: 193.

Level III - services include but not be limited to the following:  A. Ventilator care (chronic) and supplies, including all medical supplies, specialty laboratory test (e.g. blood gas) pulse oximetry, pulmonary testing, and pulmonary rehab  B. Dialysis Services and Supplies, Hemodialysis and/or Peritoneal Dialysis when provided by facility C. Wound care for stage 4 wounds  D. Respiratory Isolation - droplet precautions and negative pressure room  E. Plus Therapy: any combination of Physical Therapy, Occupational Therapy and Speech.

LEVEL 2 $465 Revenue Code: 192.

Subacute Care — Level II services include but are not limited to:  A. Telemetry  B. Patient Controlled Analgesia Pump  C. Tracheostomy Care  D. Central Access Intravenous Lines  E. Total Parenteral Nutrition = (“TPN”) Administration* F. In-House Dialysis provided by an external vendor G. Therapy: >1.5 hours of combined Physical Therapy, Occupational Therapy or Speech Therapy per day at a minimum of six days/wk. Therapies provided by licensed/registered therapist or an assistant (PT/OT) as appropriate  H. Complex Neurological patient requiring PT, OT and/or ST services  I. Wound care for stage 3  J. General post op surgical recovery i.e. patient - new ostomy care and education or joint replacement/ repair due to fall  K. G-Tube, J-Tube  L. Includes all services in Level I.

LEVEL 4 $740 Revenue Code: 194.

Level IV- services to include but not be limited to the following:  A. Ventilator care (weanable) and supplies, including all medical supplies, specialty laboratory tests (ie. blood gases), pulse oximetry, pulmonary testing, and pulmonary rehabilitation B. Extensive nursing and technical intervention defined as vital signs every two hours, neurological checks every two hours, and/or 1:1 extensive nursing care. Recommend addition of time limited protocols with Medical Director oversite.

Therapy as clinically indicated. Frequency and intensity determined based on medical condition of the patient  F. Includes all services in Levels I and II.

C. Plus Therapy: any combination of Physical Therapy, Occupational Therapy and Speech Therapy as Clinically indicated. Frequency and intensity determined based on medical condition of the patient  D. Complex Bariatric Care >400 LBS - with multiple comorbidities  E. New organ transplant  F. Includes all services in Levels I, II and III.
'''

embeddings_model = HuggingFaceBgeEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder='/Users/oleg/Projects/PPC/home/ragagent/caches'
)
#
# chunker: SemanticChunker = SemanticChunker(
#     embeddings=embeddings_model
# )
#
# chunks: list[str] = chunker.split_text(content)

chunks = [content] #re.split(r'\n{2,}', content)

# for i, chunk in enumerate(chunks, 1):
#     print('<------ chunk', i, '------>\n', chunk, '\n')

params = {}

llm = ChatOllama(
    base_url='http://localhost:11434',
    streaming=True,
    model="llama3.1",
    temperature=0.0,
    seed=2,
    top_k=10,
    top_p=0.3,
    num_ctx=3072,
    num_predict=-2
)

vectorstore = Neo4jVector(
    url='neo4j://localhost:7687',
    username='neo4j',
    password='pr0baPera',
    embedding=embeddings_model,
    pre_delete_collection=True
)
vectorstore.create_new_index()
vectorstore.add_texts(chunks)

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3, 'fetch_k': 30}
)

system_message = "Please give me precise information. Don't be verbose."

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
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
)

questions = [
    "If leveled plan, what levels are included and what is the name of each level?",
    "If leveled plan, what are the per diem rates for each level?",
    "If leveled plan, what are the revenue codes for each level?",
    "Please list all included services for each level and provide the level definition."
]
for i, question in enumerate(questions, 1):
    print(f'Q{i}: {question}')
    print(f'A{i}: {qa_chain.invoke(question)}')

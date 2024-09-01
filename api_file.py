# import os
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Neo4jVector
# from langchain.chains import RetrievalQA
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain_community.graphs import Neo4jGraph
# from chains import load_embedding_model, load_llm
# from dotenv import load_dotenv
# import logging
# import uuid
#
# load_dotenv(".env")
#
# url = os.getenv("NEO4J_URI")
# username = os.getenv("NEO4J_USERNAME")
# password = os.getenv("NEO4J_PASSWORD")
# ollama_base_url = os.getenv("OLLAMA_BASE_URL")
# embedding_model_name = os.getenv("EMBEDDING_MODEL")
# llm_name = os.getenv("LLM")
# # os.environ["NEO4J_URL"] = url
#
# # Setup logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# embeddings, dimension = load_embedding_model(
#     embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
# )
#
# llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})
#
# app = FastAPI()
# neo4j_graph = Neo4jGraph(
#     url=url,
#     username=username,
#     password=password
# )
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# vectorstore = None
#
# # Function to reset the graph
# def reset_graph():
#     try:
#         with neo4j_graph._driver.session() as session:
#             session.write_transaction(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container, initial_text=""):
#         self.container = container
#         self.text = initial_text
#
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.text += token
#         self.container.append(self.text)
#
# def process_pdfs(files) -> list:
#     all_chunks = []
#
#     for file in files:
#         loader = PyPDFLoader(file)
#         documents = loader.load()
#
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200, length_function=len
#         )
#
#         file_name = os.path.basename(file.filename)
#         for doc in documents:
#             chunks = text_splitter.split_text(text=doc.page_content)
#             for chunk in chunks:
#                 all_chunks.append({"text": chunk, "metadata": {"file_name": file_name}})
#
#     return all_chunks
#
# def insert_pdf_data(chunks: list) -> None:
#     import_query = """
#     UNWIND $data AS chunk
#     MERGE (document:Document {file_name: chunk.metadata.file_name})
#     CREATE (chunk_node:PdfChunk {text: chunk.text, file_name: chunk.metadata.file_name})
#     CREATE (document)-[:HAS_CHUNK]->(chunk_node)
#     """
#     neo4j_graph.query(import_query, {"data": chunks})
#
# def store_chunks_in_vectorstore(chunks: list):
#     texts = [chunk["text"] for chunk in chunks]
#     metadata = [chunk["metadata"] for chunk in chunks]
#     vectorstore = Neo4jVector.from_texts(
#         texts,
#         url=url,
#         username=username,
#         password=password,
#         embedding=embeddings,
#         index_name="pdf_bot",
#         node_label="PdfBotChunk",
#         pre_delete_collection=True
#     )
#     return vectorstore
#
# def ask_question(vectorstore, query: str, file_name: str = None, filter_by_filename: bool = True):
#     retriever = vectorstore.as_retriever()
#
#     if filter_by_filename and file_name:
#         # Retrieve filtered chunks by filename using a Cypher query
#         cypher_query = """
#         MATCH (d:Document {file_name: $file_name})-[:HAS_CHUNK]->(c:PdfBotChunk)
#         RETURN c.text AS text
#         """
#         with neo4j_graph._driver.session() as session:
#             chunks = [record["text"] for record in session.run(cypher_query, file_name=file_name)]
#         retriever = retriever.filter_by_texts(chunks)
#
#     qa = RetrievalQA.from_chain_type(
#         llm=llm, chain_type="stuff", retriever=retriever
#     )
#
#     stream_handler = StreamHandler([])
#     result = qa.run(query, callbacks=[stream_handler])
#
#     return {"answer": result, "streamed_output": stream_handler.text}
#
# # def ask_question(vectorstore, query: str, file_name: str = None, filter_by_filename: bool = True):
# #     retriever = vectorstore.as_retriever()
#
# #     if filter_by_filename and file_name:
# #         retriever = retriever.filter(lambda x: x.metadata.get("file_name") == file_name)
#
# #     qa = RetrievalQA.from_chain_type(
# #         llm=llm, chain_type="stuff", retriever=retriever
# #     )
#
# #     stream_handler = StreamHandler([])
# #     result = qa.run(query, callbacks=[stream_handler])
#
# #     return {"answer": result, "streamed_output": stream_handler.text}
#
# @app.post("/process_pdfs/")
# async def process_pdfs_endpoint(files: list[UploadFile]):
#     try:
#         chunks = process_pdfs(files)
#         insert_pdf_data(chunks)
#         global vectorstore
#         vectorstore = store_chunks_in_vectorstore(chunks)
#         return {"status": "PDFs processed successfully", "chunks": chunks}
#     except Exception as e:
#         logger.error(f"Error processing PDFs: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing PDFs: {e}")
#
# @app.post("/ask_question/")
# async def ask_question_endpoint(query: str = Form(...), file_name: str = Form(None), filter_by_filename: bool = Form(True)):
#     global vectorstore
#     if vectorstore is None:
#         raise HTTPException(status_code=400, detail="No PDF has been processed yet.")
#
#     try:
#         result = ask_question(vectorstore, query, file_name, filter_by_filename)
#         return {"answer": result['answer'], "streamed_output": result['streamed_output']}
#     except Exception as e:
#         logger.error(f"Error retrieving answer: {e}")
#         raise HTTPException(status_code=500, detail=f"Error retrieving answer: {e}")
#
# @app.get("/reset_graph/")
# def reset_graph_endpoint():
#     reset_graph()
#     return {"status": "Graph has been reset successfully"}
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)

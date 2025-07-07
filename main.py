from utils.api_setup import api_setup
from utils.document_storage import document_storage
from utils.retriver_setup import retriver_setup


api_setup_obj = api_setup()
api_setup_obj.pinecone_setup()

doc_storage_obj = document_storage()
doc_storage_obj.vector_storage(api_setup_obj)

retriever_setup_obj = retriver_setup(api_setup_obj)
retriever_setup_obj.reranker()
retriever_setup_obj.llm_call(api_setup_obj)

query = "when was the gemini launch ?"
result = retriever_setup_obj.qa_chain.invoke({"query": query})

print("Answer:", result["result"])

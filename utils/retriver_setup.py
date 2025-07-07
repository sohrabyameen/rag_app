from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import RetrievalQA


class retriver_setup:
    def llm_call(self, api_setup_obj):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=api_setup_obj.llm,
            retriever=self.compression_retriever,
        )
    def reranker(self):
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )

    
    def __init__(self,api_setup_obj):
        self.qa_chain = None
        self.retriever = api_setup_obj.vector_store.as_retriever(search_kwargs={"k": 8})

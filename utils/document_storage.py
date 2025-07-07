import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class document_storage:
    def vector_storage(self,api_setup_obj):
        _ = api_setup_obj.vector_store.add_documents(documents=self.all_splits)
    def __init__(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        if not files:
            raise FileNotFoundError("No PDF files found in the data folder.")
        
        pdf_path = os.path.join(data_dir, files[0])
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=80)
        self.all_splits = text_splitter.split_documents(docs)

        
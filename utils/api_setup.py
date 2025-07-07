import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings



class api_setup:
    def pinecone_setup(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pinecone'))

        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

        if not files:
            raise FileNotFoundError("No txt file found in the data folder.")

        file_path = os.path.join(data_dir, files[0])
        with open(file_path, 'r') as file:
            lines = file.readlines()
        name = lines[0].strip()
        api_key = lines[1].strip()

        pc = Pinecone(api_key=api_key)
        index = pc.Index(name)
        #index.delete(delete_all=True)
        self.vector_store = PineconeVectorStore(embedding=self.embeddings, index=index)

    def __init__(self):
        self.llm = None
        self.vector_store = None
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gemini'))

        files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

        if not files:
            raise FileNotFoundError("No json file found in the data folder.")

        file_path = os.path.join(data_dir, files[0])

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_path

        self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

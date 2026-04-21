# RAG Application with Gemini and Pinecone

A Retrieval-Augmented Generation (RAG) application that uses Google Gemini for embeddings and LLM, Pinecone for vector storage, and LangChain for orchestration.

## Features

- **Document Processing**: Loads and processes PDF documents
- **Vector Storage**: Stores document embeddings in Pinecone vector database
- **Semantic Search**: Retrieves relevant documents using vector similarity
- **Reranking**: Uses BAAI/bge-reranker-base cross-encoder for improved retrieval accuracy
- **LLM Integration**: Leverages Google Gemini 2.0 Flash for question answering

## Project Structure

```
app/
├── data/                          # PDF documents to process
│   └── Google.pdf
├── gemini/                        # Google API credentials
│   └── gen-lang-client-*.json
├── pinecone/                      # Pinecone API configuration
│   └── pinekey.txt
├── utils/                         # Utility modules
│   ├── api_setup.py              # API initialization (Pinecone & Gemini)
│   ├── document_storage.py      # PDF loading and text splitting
│   └── retriver_setup.py        # Retriever and reranker configuration
├── main.py                       # Main application entry point
└── requirements.txt              # Python dependencies
```

## Prerequisites

- Python 3.8+
- Google Cloud project with Gemini API enabled
- Pinecone account with an index created
- Google Service Account credentials (JSON key file)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Google Gemini Setup

Place your Google Service Account JSON credentials file in the `gemini/` directory:
```
gemini/gen-lang-client-*.json
```

### 2. Pinecone Setup

Create a `pinekey.txt` file in the `pinecone/` directory with the following format:
```
your-index-name
your-pinecone-api-key
```

The first line should be your Pinecone index name, and the second line should be your API key.

### 3. Document Setup

Place PDF documents in the `data/` directory. The application will automatically load the first PDF file it finds.

## Usage

Run the application:
```bash
python main.py
```

The application will:
1. Initialize the Pinecone and Gemini APIs
2. Load and split the PDF document
3. Store document embeddings in Pinecone
4. Set up the retriever with reranking
5. Execute a sample query: "when was the gemini launch?"
6. Print the answer

## Customization

### Modify the Query

Edit the query in `main.py` (line 16):
```python
query = "your question here"
```

### Adjust Retrieval Parameters

- **Chunk size/overlap**: Modify in `utils/document_storage.py` (line 19)
- **Top-k retrieval**: Modify in `utils/retriver_setup.py` (line 23)
- **Reranker top-n**: Modify in `utils/retriver_setup.py` (line 15)

### Change LLM Model

Modify the model in `utils/api_setup.py` (line 43):
```python
self.llm = init_chat_model("your-model-name", model_provider="google_genai")
```

## Dependencies

- langchain
- langchain-google-genai
- langchain-community
- langchain-pinecone
- pinecone-client
- transformers
- torch
- google-api-python-client
- PyPDF2
- tiktoken

## Architecture

1. **API Setup** (`api_setup.py`): Initializes Pinecone vector store and Google Gemini LLM/embeddings
2. **Document Storage** (`document_storage.py`): Loads PDFs, splits text into chunks, and stores embeddings
3. **Retriever Setup** (`retriver_setup.py`): Configures vector retriever with cross-encoder reranking and QA chain
4. **Main Pipeline** (`main.py`): Orchestrates the entire workflow and executes queries

## Notes

- The application uses the BAAI/bge-reranker-base model for reranking retrieved documents
- Documents are split into chunks of 550 characters with 80-character overlap
- The retriever fetches 8 documents initially, which are reranked to top 3

# ingest_documents.py
import os
from vectorstore import VectorStore  # Assuming your class is importable
from config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # You'll need to add langchain to requirements
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

def ingest():
    # Initialize vector store
    vector_store = VectorStore()
    
    # Load documents from a ./data directory
    loader = DirectoryLoader('./data', glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add to vector store (you'll need to adapt this to your VectorStore.add_documents method)
    texts = [chunk.page_content for chunk in chunks]
    vector_store.add_documents(texts)
    
    print(f"Ingested {len(chunks)} chunks.")

if __name__ == "__main__":
    ingest()

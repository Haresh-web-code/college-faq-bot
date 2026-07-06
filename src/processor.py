import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_documents(data_path="data/", db_path="vector_db/"):
    # 1. Load PDFs and TXTs
    documents = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    
    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings and Save to Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print(f"Knowledge base created with {len(chunks)} chunks.")
    return vector_db
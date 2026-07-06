import os
from dotenv import load_dotenv
from src.processor import ingest_documents

def main():
    # Load environment variables (e.g. OPENAI_API_KEY)
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment or .env file.")
        
    print("Starting document ingestion...")
    data_dir = "data/"
    db_dir = "vector_db/"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        
    # Check if there are any documents to ingest
    files = os.listdir(data_dir)
    if not files:
        print(f"No files found in '{data_dir}'. Please add some .txt or .pdf files first.")
        return
        
    try:
        ingest_documents(data_path=data_dir, db_path=db_dir)
        print("Ingestion completed successfully!")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    main()

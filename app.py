import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from src.processor import ingest_documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="College Admin Bot")
st.title("🎓 College Admin FAQ Bot")

# Initialize Vector DB (Singleton pattern for efficiency)
@st.cache_resource
def load_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma(persist_directory="vector_db/", embedding_function=embeddings)

db = load_db()

# Basic Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about fees, exams, or contacts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval logic
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    
    with st.chat_message("assistant"):
        response = qa_chain.invoke(prompt)["result"]
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
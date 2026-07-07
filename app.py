import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from src.processor import ingest_documents

st.set_page_config(page_title="College Admin Bot")
st.title("🎓 College Admin FAQ Bot")

# Initialize Vector DB (Singleton pattern for efficiency)
@st.cache_resource
def load_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma(persist_directory="vector_db/", embedding_function=embeddings)

db = load_db()

# Build the prompt template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful college administration assistant. "
     "Use the following retrieved context to answer the question. "
     "If you don't know the answer, say so.\n\n{context}"),
    ("human", "{question}"),
])

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

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

    # Retrieve relevant documents and build context
    retriever = db.as_retriever()
    docs = retriever.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    chain = qa_prompt | llm
    with st.chat_message("assistant"):
        response = chain.invoke({"context": context, "question": prompt})
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
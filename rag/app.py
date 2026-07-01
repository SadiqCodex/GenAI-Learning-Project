import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

PERSIST_DIR = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "phi3:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def safe_remove_directory(path: str, retries: int = 5, delay: float = 0.5) -> bool:
    if not os.path.exists(path):
        return True

    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            if attempt == retries - 1:
                return False
            time.sleep(delay)

    return False

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

            if os.path.exists(PERSIST_DIR):
                removed = safe_remove_directory(PERSIST_DIR)
                if not removed:
                    st.warning("Previous vector store is locked by Windows. Using a fresh directory instead.")
                    timestamp = int(time.time())
                    PERSIST_DIR = str(BASE_DIR / f"chroma_db_{timestamp}")

            os.makedirs(PERSIST_DIR, exist_ok=True)

            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorstore.save_local(PERSIST_DIR)

        st.success("Vector database created!")


if os.path.exists(PERSIST_DIR):

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    vectorstore_ready = False

    if os.path.exists(os.path.join(PERSIST_DIR, "index.faiss")):
        vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore_ready = True

    if not vectorstore_ready:
        st.info("No saved vector store found yet. Upload a PDF and create the database first.")
    else:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        llm = ChatOllama(model=CHAT_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
                ),
                (
                    "human",
                    """Context:
{context}

Question:
{question}
"""
                )
            ]
        )

        st.divider()
        st.subheader("Ask Questions From the Book")

        query = st.text_input("Enter your question")

        if query:

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

            st.write("### AI Answer")
            st.write(response.content)
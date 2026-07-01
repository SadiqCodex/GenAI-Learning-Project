import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

PERSIST_DIR = str(BASE_DIR / "chroma_db")
PDF_PATH = BASE_DIR / "document loaders" / "deeplearning.pdf"
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def safe_remove_directory(path: str, retries: int = 5, delay: float = 0.5) -> None:
    if not os.path.exists(path):
        return

    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


data = PyPDFLoader(str(PDF_PATH))
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

if os.path.exists(PERSIST_DIR):
    safe_remove_directory(PERSIST_DIR)
os.makedirs(PERSIST_DIR, exist_ok=True)

vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
vectorstore.save_local(PERSIST_DIR)

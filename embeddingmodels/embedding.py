# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
# from langchain_gemini import GeminiEmbeddings
# from langchain_groq import GroqEmbeddings
from dotenv import load_dotenv

load_dotenv()

query = "Hello world"
texts = ["This is a sample text for embedding generation.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "They are used in various natural language processing tasks, such as search and recommendation."]

# openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimension=64)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# gemini_embeddings = GeminiEmbeddings(model="gemini-embedding-1", dimension=64)
# groq_embeddings = GroqEmbeddings(model="groq-embedding-1", dimension=64)

# openai_vector = openai_embeddings.embed_query(query)
ollama_vector = ollama_embeddings.embed_documents(texts)
# gemini_vector = gemini_embeddings.embed_query(query)
# groq_vector = groq_embeddings.embed_query(query)

# print("OpenAI vector:", openai_vector)
print("Ollama vector:", ollama_vector)
# print("Gemini vector:", gemini_vector)
# print("Groq vector:", groq_vector)

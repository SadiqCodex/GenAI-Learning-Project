from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

query = "Hello world"
texts = ["This is a sample text for embedding generation.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "They are used in various natural language processing tasks, such as search and recommendation."]

# Local HuggingFace Embeddings
local_embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-xsmall-v1")
local_vector = local_embeddings.embed_documents(texts)
local_query_vector = local_embeddings.embed_query(query)

# HuggingFace API Embeddings
# api_embeddings = HuggingFaceEmbeddings(model_name="mxbai-embed-xsmall-v1")
# api_vector = api_embeddings.embed_documents(texts)
# api_query_vector = api_embeddings.embed_query(query)

print("Local HuggingFace vector:", local_vector)
print("Local Query vector:", local_query_vector)
# print("\nAPI HuggingFace vector:", api_vector)
# print("API Query vector:", api_query_vector)

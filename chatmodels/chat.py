from dotenv import load_dotenv
import os

load_dotenv()

# from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import init_chat_model
# from langchain_gemini import ChatGemini
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# model = ChatOpenAI(model="gpt-4.1")
# response = model.invoke("Hello, how are you?")
# print(response.content)

# model = ChatGemini(
#     model="gemini-1.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.7
# )

# model = ChatGroq(
#     model="llama-3.1-70b-versatile",  
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.7
# )

model = ChatOllama(
    model="phi3",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.9,
    max_tokens=512,
)

response = model.invoke("write a short poem about ai")
print(response.content)
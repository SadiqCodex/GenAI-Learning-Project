from dotenv import load_dotenv
import os

load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result = 5)

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_CHAT_MODEL", "phi3:latest")
llm = ChatOllama(model=ollama_model, base_url=ollama_base_url, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt | llm | StrOutputParser()

try:
    news_result = search_tool.run("Latest AI news of 2026")
except Exception as e:
    news_result = (
        "Unable to fetch live news right now. "
        f"The search service failed with: {e}"
    )
    print("Tavily search failed:", e)

result = chain.invoke({"news": news_result})

print(result)

print(search_tool.description)
print(search_tool.name)
print(search_tool.args)

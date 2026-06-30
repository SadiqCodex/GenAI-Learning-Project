from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOllama(
    model="phi3",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.7,  # ✅ Increased from 0.1
)

system_prompt = "You are a funny ai agent."

messages = [
    SystemMessage(content=system_prompt)
    ]

print("Welcome to the chat. Type '0' to exit.")
print("-" * 50)

while True:
    prompt = input("\nYou: ")
    # ✅ Add user message in proper format
    messages.append(HumanMessage(content=prompt))
    if prompt.strip().lower() == "0":
        print("Exiting chat.")
        break
    # ✅ Use the messages list directly (not concatenated string)
    response = model.invoke(messages)
    bot_reply = response.content

    messages.append(AIMessage(content=bot_reply))
    print(f"Bot : {bot_reply} ")
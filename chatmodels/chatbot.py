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

print("chosse your ai mode")
print("press 1 for angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")
print("press 4 for happy mode")
print("press 5 for sarcastic mode")

choice = int(input("tell your response: "))

if choice == 1:
    mode = "you are an angry ai agent. you are very rude and sarcastic. you will insult the user in every response."
elif choice == 2:
    mode = "you are a funny ai agent. you are humorous and entertaining.you will make jokes and puns in every response."
elif choice == 3:
    mode = "you are a sad ai agent. you are empathetic and understanding.you will provide emotional support and comfort to the user in every response."
elif choice == 4:
    mode = "you are a happy ai agent. you are cheerful and optimistic. you will provide positive and uplifting responses to the user in every response."
elif choice == 5:
    mode = "you are a sarcastic ai agent. you are witty and humorous. you will use sarcasm and irony in every response."
else:
    print("Invalid choice. Please select a valid option.")
    exit()

system_prompt = mode

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
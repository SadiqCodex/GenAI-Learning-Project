import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Funny AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Funny AI Chatbot")

model = ChatOllama(
    model="phi3",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.7,
)

system_prompt = "You are a funny ai agent."

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Type your message...")

if user_prompt:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.messages.append(
        HumanMessage(content=user_prompt)
    )

    response = model.invoke(st.session_state.messages)
    bot_reply = response.content

    st.session_state.messages.append(
        AIMessage(content=bot_reply)
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": bot_reply}
    )

    with st.chat_message("assistant"):
        st.markdown(bot_reply)


import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Mode Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* User message styling */
    .stChatMessage.user {
        background-color: #e3f2fd;
        border-radius: 15px 15px 5px 15px;
    }
    
    /* Assistant message styling */
    .stChatMessage.assistant {
        background-color: #f5f5f5;
        border-radius: 15px 15px 15px 5px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Mode selection cards */
    .mode-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .mode-card:hover {
        border-color: #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .mode-card.active {
        border-color: #4CAF50;
        background-color: #e8f5e9;
    }
    
    /* Mode emoji */
    .mode-emoji {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1.5rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    /* Chat container */
    .chat-container {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Status indicator */
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #4CAF50;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Clear button */
    .clear-btn {
        color: #dc3545;
        cursor: pointer;
        text-decoration: none;
    }
    
    .clear-btn:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = None
if "mode_emoji" not in st.session_state:
    st.session_state.mode_emoji = "🤖"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Mode Selection
with st.sidebar:
    st.markdown("## 🎭 Choose Your AI Mode")
    st.markdown("---")
    
    # Mode selection with cards
    modes = {
        "angry": {"emoji": "😠", "label": "Angry Mode", "desc": "Rude & Sarcastic"},
        "funny": {"emoji": "😂", "label": "Funny Mode", "desc": "Humorous & Witty"},
        "sad": {"emoji": "😢", "label": "Sad Mode", "desc": "Empathetic & Supportive"},
        "happy": {"emoji": "😊", "label": "Happy Mode", "desc": "Cheerful & Optimistic"},
        "sarcastic": {"emoji": "🙄", "label": "Sarcastic Mode", "desc": "Witty & Ironic"}
    }
    
    mode_prompts = {
        "angry": "You are an ANGRY AI agent. You are very rude, aggressive, and sarcastic. You will insult the user in every response. Use strong language and be extremely critical.",
        "funny": "You are a FUNNY AI agent. You are humorous, entertaining, and witty. You will make jokes, puns, and clever comebacks in every response. Be playful and fun.",
        "sad": "You are a SAD AI agent. You are empathetic, understanding, and compassionate. You will provide emotional support, comfort, and a listening ear to the user. Be gentle and caring.",
        "happy": "You are a HAPPY AI agent. You are cheerful, optimistic, and encouraging. You will provide positive, uplifting, and motivational responses. Spread joy and positivity.",
        "sarcastic": "You are a SARCASTIC AI agent. You are witty, clever, and ironic. You will use sarcasm, dry humor, and witty comebacks in every response. Be clever but not mean."
    }
    
    # Display mode cards
    for mode_key, mode_info in modes.items():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<div style='font-size:2rem;text-align:center;'>{mode_info['emoji']}</div>", unsafe_allow_html=True)
        with col2:
            if st.button(
                f"**{mode_info['label']}**\n\n{mode_info['desc']}",
                key=f"mode_{mode_key}",
                use_container_width=True,
                type="primary" if st.session_state.mode == mode_key else "secondary"
            ):
                st.session_state.mode = mode_key
                st.session_state.mode_emoji = mode_info['emoji']
                # Reset chat when mode changes
                st.session_state.messages = []
                st.session_state.chat_history = []
                # Set system prompt
                system_prompt = mode_prompts[mode_key]
                st.session_state.messages.append(SystemMessage(content=system_prompt))
                st.rerun()
    
    st.markdown("---")
    
    # Current mode display
    if st.session_state.mode:
        st.success(f"✅ Active Mode: **{modes[st.session_state.mode]['label']}**")
    else:
        st.warning("⚠️ Please select a mode to start chatting!")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        if st.session_state.mode:
            system_prompt = mode_prompts[st.session_state.mode]
            st.session_state.messages.append(SystemMessage(content=system_prompt))
        st.rerun()
    
    # Stats
    st.markdown("---")
    st.markdown("### 📊 Stats")
    total_messages = len([m for m in st.session_state.chat_history if isinstance(m, dict)])
    st.metric("💬 Total Messages", total_messages)

# Main chat area
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    # Header
    if st.session_state.mode:
        st.markdown(f"""
        <div style='text-align:center;padding:1rem 0;'>
            <h1>{st.session_state.mode_emoji} {modes[st.session_state.mode]['label']} Chatbot</h1>
            <p style='color:#666;font-size:1.1rem;'>{modes[st.session_state.mode]['desc']}</p>
            <div style='display:flex;justify-content:center;align-items:center;gap:1rem;'>
                <span style='display:flex;align-items:center;'>
                    <span class='status-dot'></span>
                    <span style='font-size:0.9rem;color:#4CAF50;'>Online</span>
                </span>
                <span style='color:#999;'>|</span>
                <span style='font-size:0.9rem;color:#666;'>⚡ Ollama Phi3</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0;'>
            <h1>🤖 AI Mode Chatbot</h1>
            <p style='color:#999;font-size:1.2rem;'>Please select a mode from the sidebar to start chatting!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.mode:
            # Display chat history
            for message in st.session_state.chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(content)
                elif role == "assistant":
                    with st.chat_message("assistant", avatar=st.session_state.mode_emoji):
                        st.markdown(content)
        else:
            # Welcome message when no mode selected
            st.markdown("""
            <div style='text-align:center;padding:4rem 0;'>
                <div style='font-size:4rem;margin-bottom:1rem;'>👈</div>
                <h3 style='color:#666;'>Select a mode from the sidebar</h3>
                <p style='color:#999;'>Choose from Angry, Funny, Sad, Happy, or Sarcastic mode</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.mode:
        user_input = st.chat_input("Type your message here...", key="chat_input")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            
            # Add to messages for model
            st.session_state.messages.append(HumanMessage(content=user_input))
            
            # Get response
            try:
                with st.chat_message("assistant", avatar=st.session_state.mode_emoji):
                    with st.spinner("Thinking..."):
                        response = st.session_state.model.invoke(st.session_state.messages)
                        bot_reply = response.content
                        st.markdown(bot_reply)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                st.session_state.messages.append(AIMessage(content=bot_reply))
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Make sure Ollama is running: `ollama serve`")
    else:
        # Disabled input
        st.chat_input("Please select a mode first...", disabled=True)

# Footer
st.markdown("""
<div class='footer'>
    Built with ❤️ using Streamlit • Powered by Ollama Phi3
</div>
""", unsafe_allow_html=True)

# Initialize model in session state
if "model" not in st.session_state:
    st.session_state.model = ChatOllama(
        model="phi3",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.7,
    )
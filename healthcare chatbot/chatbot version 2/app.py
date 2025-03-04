import streamlit as st
import requests
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# FastAPI backend URL (adjust if your backend is hosted elsewhere)
BACKEND_URL = "http://localhost:8000"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LangChain ChatOpenAI
language_model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Function to classify the query using LangChain
def classify_user_query(query):
    classification_prompt = ChatPromptTemplate.from_template(
        "Determine if the following query requires medical knowledge to answer. "
        "Query: {query}\n"
        "Reply with just 'Yes' or 'No'."
    )
    classification_chain = classification_prompt | language_model | StrOutputParser()
    result = classification_chain.invoke({"query": query})
    return result.strip().lower() == 'yes'

# Chain for non-medical queries
general_assistant_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
general_assistant_chain = general_assistant_prompt | language_model | StrOutputParser()

# Set page config to use a dark theme
st.set_page_config(page_title="MediChat", page_icon="🏥", layout="wide")

# Custom CSS for better visibility and chat-like interface
st.markdown("""
<style>
    body {
        color: #FFFFFF;
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
    }
    .main {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
    .warning {
        background-color: #8B8000;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Main content
main_column, sidebar_column = st.columns([3, 1])

with main_column:
    st.title("🏥 MediChat - Your Medical Assistant")

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []

    # Chat history container
    chat_history_container = st.container()

    # Input container
    user_input_container = st.container()

    # Warning message
    st.markdown("""
    <div class="warning">
        ⚠️ This AI assistant provides general information and should not replace professional medical advice. 
        Always consult with a qualified healthcare provider for personal medical concerns.
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages from history on app rerun
    with chat_history_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    with user_input_container:
        if user_input := st.chat_input("What is your medical question?"):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            # Display user message in chat message container
            with chat_history_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
            
            # Classify the query
            is_medical_query = classify_user_query(user_input)
            
            if is_medical_query:
                # Send the message to the backend and get a response
                try:
                    response = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input})
                    if response.status_code == 200:
                        response_data = response.json()
                        assistant_response = response_data["answer"]
                        sources = response_data.get("sources", [])
                        
                        # Display assistant response in chat message container
                        with chat_history_container:
                            with st.chat_message("assistant"):
                                st.markdown(assistant_response)
                        # Add assistant response to chat history
                        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
                        
                        # Update current sources
                        st.session_state.current_sources = sources
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.RequestException as e:
                    st.error(f"Error communicating with the backend: {str(e)}")
            else:
                # Use LangChain for non-medical queries
                assistant_response = general_assistant_chain.invoke({"question": user_input})
                with chat_history_container:
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
                st.session_state.current_sources = []  # Clear sources for non-medical queries

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.session_state.current_sources = []
        st.experimental_rerun()

# Sidebar for displaying sources
with sidebar_column:
    st.sidebar.title("Information Sources")
    if st.session_state.current_sources:
        st.sidebar.markdown("Sources for the most recent answer:")
        for idx, source in enumerate(st.session_state.current_sources, 1):
            with st.sidebar.expander(f"Source {idx}"):
                st.markdown(source["content"])
    else:
        st.sidebar.write("No sources available for the current answer.")
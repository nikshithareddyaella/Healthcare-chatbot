# Healthcare Chatbot

## Overview
This project is a *Healthcare Chatbot* designed to assist users with medical queries. It uses *FastAPI* for the backend, *Streamlit* for the frontend, and *Pinecone* for vector storage and retrieval. The chatbot leverages *OpenAI's GPT model* to provide accurate and context-aware medical answers.

---



## Features
- *Medical Query Handling*: Answers medical-related questions using a retrieval-augmented generation (RAG) pipeline.
- *Non-Medical Query Handling*: Answers general questions using OpenAI's GPT model.
- *Chat Interface*: A user-friendly Streamlit-based chat interface.
- *Source Attribution*: Displays sources for medical answers (if available).
- *Real-Time Communication*: Uses WebSocket for seamless interaction between the frontend and backend.

---

## Tech Stack
- *Frontend*: Streamlit, HTML, CSS
- *Backend*: FastAPI, Uvicorn
- *AI Models*: OpenAI GPT, LangChain
- *Vector Storage*: Pinecone
- *Authentication*: JWT (if applicable)
- *Hosting*: Docker, Render

---

## Setup Instructions

### 1. Clone the Repository
bash
git clone https://github.com/K-Tarunkumar/Healthcare-chatbot.git
cd Healthcare-chatbot


### 2. Set Up a Virtual Environment
bash
python -m venv venv


- *Activate the Virtual Environment*:
  - On Windows:
    bash
    venv\Scripts\activate
    
  - On macOS/Linux:
    bash
    source venv/bin/activate
    

### 3. Install Dependencies
bash
pip install -r requirements.txt


### 4. Set Up Environment Variables
Create a .env file in the root directory with the following content:
plaintext
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment


Replace your_openai_api_key, your_pinecone_api_key, and your_pinecone_environment with your actual API keys.

### 5. Run with Docker
bash
docker-compose up --build


### 6. Access the App
Open your browser and go to:
- *Frontend*: [http://localhost:8501](http://localhost:8501)
- *Backend API*: [http://localhost:8000](http://localhost:8000)

---

## API Endpoints
- *Chat*: POST /chat
  - Input: {"message": "Your medical question"}
  - Output: {"answer": "Bot's response", "sources": [...]}

- *WebSocket*: ws://localhost:8000/ws
  - Real-time communication for chat interactions.

---

## GitHub Repository
GitHub Link: [https://github.com/K-Tarunkumar/Healthcare-chatbot.git](https://github.com/K-Tarunkumar/Healthcare-chatbot.git)



---

## Acknowledgments
- *OpenAI* for the GPT model.
- *Pinecone* for vector storage.
- *FastAPI* and *Streamlit* for backend and frontend frameworks.

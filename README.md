# Banking Customer Support Chatbot

## Overview
The **Banking Customer Support Chatbot** is an intelligent, conversational AI assistant developed to provide immediate and accurate responses to customer queries in the banking domain. Powered by **Google's Gemini Flash model**, **LangChain**, **ChromaDB**, and **Streamlit**, this chatbot delivers responses through Retrieval-Augmented Generation (RAG), enabling it to give relevant and informed answers from a custom FAQ dataset.

This project features:
- Streamlit web UI for live chatbot interactions
- Jupyter notebook for step-by-step development and testing
- Document-based RAG pipeline using ChromaDB and Gemini embeddings
- Semantic search and retrieval from a local text corpus (Banking FAQs)

---

## Features
- ✅ Real-time chatbot interface using Streamlit
- ✅ Uses Gemini Flash LLM from Google
- ✅ Retrieval of relevant chunks using ChromaDB
- ✅ Preprocessed with RecursiveCharacterTextSplitter
- ✅ Context-aware, concise responses
- ✅ Session-based chat history for continuous conversation
- ✅ Friendly, professional assistant tone

---

## Tech Stack
- **Language Model:** Gemini 2.0 Flash via Google Generative AI
- **Embeddings:** `models/embedding-001`
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit
- **Backend:** LangChain RAG Pipeline
- **Data Source:** `Banking FAQS.txt`
- **Notebook Environment:** Jupyter Notebook (used for building and debugging components)

---

## How It Works
1. **Data Ingestion**: Loads and reads the `Banking FAQS.txt` file using `TextLoader`.
2. **Chunking**: Splits the text using `RecursiveCharacterTextSplitter` for better retrieval.
3. **Embeddings**: Converts text chunks into vectors with GoogleGenerativeAIEmbeddings.
4. **Vector Store**: Stores and retrieves relevant chunks from ChromaDB.
5. **Prompt Engineering**: Constructs context-aware prompts using LangChain's `ChatPromptTemplate`.
6. **LLM Response**: Uses `ChatGoogleGenerativeAI` to respond using contextually relevant chunks.
7. **Web UI**: Streamlit displays chat history, input box, and bot responses in real-time.

---

## Getting Started

### Prerequisites
- Python >= 3.10
- Google API key for Gemini models
- `.env` file containing:
  ```bash
  GOOGLE_API_KEY=your_google_api_key
  ```

### Installation
```bash
git clone https://github.com/Nonny-123/Banking-Customer-Chatbot.git
cd Banking-Customer-Chatbot
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run chatbot_app.py
```

---

## Usage
1. Start the chatbot interface.
2. Type any banking-related question (e.g., "How do I apply for a loan?").
3. Bash (the chatbot) retrieves relevant info and responds in real-time.
4. History is preserved across session.

---

## File Structure
- `chatbot_app.py` - Main Streamlit chatbot file
- `Banking FAQS.txt` - Source data for chatbot knowledge
- `notebook_build_pipeline.ipynb` - Development notebook to test each component
- `.env` - Environment variables (not pushed to GitHub)

---

## Future Improvements
- Add support for follow-up questions
- Improve memory handling for longer conversations
- Deploy as a web app via Hugging Face Spaces or AWS
- Fine-tune prompt formatting for more nuanced answers

---

## Author
**Okonji Chukwunonyelim Gabriel**  
Machine Learning Engineer | Data Scientist  
[GitHub](https://github.com/Nonny-123)

---

## License
This project is licensed under the MIT License.


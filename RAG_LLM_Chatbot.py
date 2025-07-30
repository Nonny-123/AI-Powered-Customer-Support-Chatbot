import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import os

#API Key Configuration
try:
    # For Streamlit Community Cloud
    api_key = st.secrets["GEMINI_API_KEY"]
except (ModuleNotFoundError, KeyError):
    # For local development
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

#Caching the Retriever
@st.cache_resource
def get_retriever(_api_key):
    url = "https://raw.githubusercontent.com/Nonny-123/AI-Powered-Customer-Support-Chatbot/main/Banking%20FAQS.txt"
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

#LLM and Prompt Configuration
def get_llm_chain(retriever, api_key):
    system_prompt = (
        """
        You are a customer support chatbot for answering customer-related queries.
        Use the following pieces of retrieved context to answer the question.
        If the answers are not in the context, you can answer to the best of your ability
        but do not lie. If you do not know the answer, say that you do not know.
        Use a concise and friendly tone, be professional and informative.
        When the conversation begins the first time introduce yourself as Bash, your banking
        customer support chatbot. As the conversation continues do not introduce yourself again unless
        asked to.
        \n\n
        {context}
        """
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key
    )

    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, Youtube_chain)
    return rag_chain

#Streamlit App UI
st.title("Banking Customer Support Chatbot 🤖")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add initial greeting
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "Hi there! I'm Bash, your banking customer support chatbot. How can I help you today?"
    })

# Display chat messages from history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main App Logic
try:
    with st.spinner("Preparing chatbot..."):
        retriever = get_retriever(api_key)
        rag_chain = get_llm_chain(retriever, api_key)
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.stop()

# Handle user input
if query := st.chat_input("Ask me anything about your banking..."):
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get and display assistant response
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": query})
        answer = response.get("answer", "I'm sorry, I couldn't find an answer.")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Rerun to display the new assistant message immediately
    st.rerun()
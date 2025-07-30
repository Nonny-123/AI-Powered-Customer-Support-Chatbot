import streamlit as st
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import os

try:
    api_key = st.secrets["GEMINI_API_KEY"]

except (ModuleNotFoundError, KeyError):
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

st.title("Banking Customer Support Chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "greeting_shown" not in st.session_state:
    st.session_state["greeting_shown"] = False

# Display initial greeting only once at the beginning
if not st.session_state["greeting_shown"]:
    greeting = "Hi there! I'm Bash, your banking customer support chatbot. How can I help you today?"
    st.session_state["chat_history"].append({"role": "assistant", "content": greeting})
    st.session_state["greeting_shown"] = True
    st.rerun() 


for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    with st.spinner("Loading..."):
        loader =  TextLoader("Banking FAQS.txt")
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    persist_path = os.path.abspath("chroma_db")
    os.makedirs(persist_path, exist_ok=True)

    vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory=persist_path)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

except Exception as e:
    st.error(f"There was an error: {e}")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                             temperature=0.3, 
                             max_tokens=None,
                             timeout=None)

except Exception as e:
    st.error(f"There was an error loading Gemini model: {e}")
    st.stop()

system_prompt = (
    """
    You are a customer support chatbot for answering customer-related queries.
    Use the following pieces of retrieved context to answer the question.
    If the answers are not in the context, you can answer to the best of your ability
    but do not lie. If you do not know the answer, say that you do not know.
    Use a concise and friendly tone, be professional and informative.
    When the conversation begins the first time introduce, as the 
    conversation continues do not introduce yourself again unless
    asked to.
    \n\n
    {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("system", "{context}"),  
        ("human", "{input}"),
    ]
)

query = st.chat_input("Say something")
#prompt = query

if query:
    st.session_state["chat_history"].append({"role": "user", "content": query})
    # with st.chat_message("user"):
    #     st.markdown(query)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    # with st.chat_message("assistant"):
    #     st.markdown(answer)
    #     st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    st.rerun()
    
    
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os

# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_history" not in st.session_state:
    st.session_state.message_history = ChatMessageHistory()
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Updated to use RecursiveCharacterTextSplitter (recommended over CharacterTextSplitter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
    
    # Updated prompt template using ChatPromptTemplate
    system_prompt = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create document chain and retrieval chain using modern LCEL approach
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def process_docs(pdf_docs):
    try:
        # Get PDF text
        raw_text = get_pdf_text(pdf_docs)
        
        if not raw_text.strip():
            st.error("No text could be extracted from the PDFs. Please check your files.")
            return False
        
        # Get text chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store using FAISS
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.session_state.processComplete = True
        
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            # Reset chat history when processing new documents
            st.session_state.chat_history = []
            st.session_state.message_history = ChatMessageHistory()
            success = process_docs(pdf_docs)
            if success:
                st.success("Processing complete!")
    
    # Add a reset button
    if st.button("Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.message_history = ChatMessageHistory()
        st.rerun()

# Main chat interface
if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        try:
            with st.spinner("Thinking..."):
                # Prepare chat history for the chain
                chat_history_messages = []
                for role, message in st.session_state.chat_history:
                    if role == "You":
                        chat_history_messages.append(HumanMessage(content=message))
                    else:
                        chat_history_messages.append(AIMessage(content=message))
                
                # Get response from conversation chain
                response = st.session_state.conversation.invoke({
                    "input": user_question,
                    "chat_history": chat_history_messages
                })
                
                answer = response["answer"]
                
                # Update chat history
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", answer))
                
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)

# Display initial instructions
else:
    st.write("👈 Upload your PDFs in the sidebar to get started!")
    st.info("""
    **How to use:**
    1. Upload one or more PDF files using the sidebar
    2. Click the 'Process' button to analyze your documents
    3. Ask questions about your PDFs in the chat interface
    4. Use 'Reset Chat' to start a new conversation
    """)

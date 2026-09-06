import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(
    page_title="PDF RAG Chatbot"
)

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

if not model_name:
    st.error("MODEL_NAME not found in .env file")
    st.stop()

def create_llm():

    return ChatGroq(
        model=model_name,
        temperature=0.3,
        api_key=groq_api_key
    )


llm = create_llm()

def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = create_embeddings()

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_pdf_path = temp_file.name


    try:
        loader = PyPDFLoader(
            temp_pdf_path
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(
            documents
        )


        if not chunks:

            raise ValueError(
                "No text found in the uploaded PDF."
            )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

        return retriever, len(documents), len(chunks)

    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for a PDF question-answering system.

Your job is to answer the user's question using ONLY the
information available in the provided PDF context.

Conversation history is provided only to understand references
such as:

- it
- this
- that
- these
- those
- the above
- the first one
- the second one

Use the conversation history to understand what the user means,
but do NOT use outside knowledge.

IMPORTANT:
- Answer only from the PDF context.
- Do not make up information.
- Do not use outside knowledge.
- If the PDF context does not contain the answer, say exactly:

"I don't know based on the provided document."

Conversation History:
{chat_history}

PDF Context:
{context}

Current User Question:
{question}

Answer:
""")

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "retriever" not in st.session_state:

    st.session_state.retriever = None


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None

def format_chat_history():

    if not st.session_state.chat_history:

        return "No previous conversation."


    history_text = ""

    for message in st.session_state.chat_history:

        if isinstance(message, HumanMessage):

            history_text += (
                f"User: {message.content}\n"
            )

        elif isinstance(message, AIMessage):

            history_text += (
                f"Assistant: {message.content}\n"
            )

    return history_text

def create_retrieval_query(question):

    """
    Create a retrieval query using the current question
    and recent conversation.

    This avoids an additional LLM call for question rewriting.
    """

    if not st.session_state.chat_history:

        return question


    recent_history = st.session_state.chat_history[-4:]

    history_text = ""

    for message in recent_history:

        if isinstance(message, HumanMessage):

            history_text += (
                f"User: {message.content}\n"
            )

        elif isinstance(message, AIMessage):

            history_text += (
                f"Assistant: {message.content}\n"
            )


    retrieval_query = f"""
Previous conversation:
{history_text}

Current question:
{question}
"""
    return retrieval_query

def ask_question(question):
    retrieval_query = create_retrieval_query(
        question
    )

    docs = st.session_state.retriever.invoke(
        retrieval_query
    )

    if not docs:

        context = "No relevant information was found in the PDF."

    else:

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    history_text = format_chat_history()

    messages = answer_prompt.invoke({

        "chat_history": history_text,

        "context": context,

        "question": question
    })

    response = llm.invoke(
        messages
    )

    answer = response.content

    st.session_state.chat_history.append(
        HumanMessage(
            content=question
        )
    )

    st.session_state.chat_history.append(
        AIMessage(
            content=answer
        )
    )
    return answer


st.title("PDF RAG Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)


with st.sidebar:

    st.header("Upload PDF")


    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )


    if uploaded_file is not None:

        if (
            st.session_state.pdf_name
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                try:

                    retriever, page_count, chunk_count = (
                        process_pdf(
                            uploaded_file
                        )
                    )


                    st.session_state.retriever = retriever

                    st.session_state.pdf_name = (
                        uploaded_file.name
                    )


                 
                    st.session_state.chat_history = []


                    st.success(
                        "PDF processed successfully!"
                    )


                    st.write(
                        f"Pages: {page_count}"
                    )

                    st.write(
                        f"Chunks: {chunk_count}"
                    )


                except Exception as e:

                    st.error(
                        f"Error processing PDF: {e}"
                    )




    if st.session_state.pdf_name:

        st.divider()

        st.write("### Current PDF")

        st.write(
            st.session_state.pdf_name
        )



if st.session_state.retriever is None:

    st.info(
        "Please upload a PDF from the sidebar to start chatting."
    )

    st.stop()



for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.write(
                message.content
            )

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.write(
                message.content
            )


question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:
    with st.chat_message("user"):

        st.write(
            question
        )

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the PDF..."
        ):

            try:
                answer = ask_question(
                    question
                )

                st.write(
                    answer
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


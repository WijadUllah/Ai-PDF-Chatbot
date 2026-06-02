import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")

st.title("📄 AI PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Vector Store
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        st.success("PDF processed successfully!")

    # Question Input
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        # Retrieve Relevant Chunks
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say "I don't know based on the provided PDF."

Context:
{context}

Question:
{question}
"""
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": question
        })

        # LLM
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.7
        )

        # Generate Answer
        response = llm.invoke(final_prompt)

        parser = StrOutputParser()

        answer = parser.invoke(response)

        st.subheader("Answer")
        st.write(answer)
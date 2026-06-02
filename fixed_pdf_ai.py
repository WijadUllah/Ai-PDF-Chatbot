import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Document Loading
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Lecture 10 and 11 .pdf")
docs = loader.load()

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Vector Store
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""
)

st.title("📄 PDF Chatbot")
st.write("Ask questions from your PDF")

question = st.text_input("Enter your question")

if st.button("Ask"):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)

    st.subheader("Answer")
    st.write(response.content)
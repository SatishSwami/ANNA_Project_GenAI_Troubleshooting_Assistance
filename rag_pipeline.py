
import os
import streamlit as st

from functools import lru_cache
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Configuration
FAISS_DIRECTORY = "faiss_index_v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.8-flash"


def get_api_key():
    """Read Gemini API key from local environment or Streamlit secrets."""

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Check your .env or Streamlit secrets."
        )

    return api_key


@lru_cache(maxsize=1)
def load_rag_pipeline():

    print("Loading RAG pipeline...")

    api_key = get_api_key()

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Load FAISS index
    if not os.path.exists(FAISS_DIRECTORY):
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_DIRECTORY}. Run ingest.py first."
        )

    vectorstore = FAISS.load_local(
        FAISS_DIRECTORY,
        embeddings,
        allow_dangerous_deserialization=True
    )

        # Configure retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=api_key
    )

    print("RAG pipeline loaded successfully!")

    return retriever, llm


def get_answer(question):

    if not question or not question.strip():
        raise ValueError("Please enter a valid question.")

    retriever, llm = load_rag_pipeline()

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "The information is not available in the knowledge base.",
            "sources": [],
            "source_details": [],
            "retrieved_chunks": []
        }

    context_parts = []
    sources = []
    source_details = []

    # Prepare context and source information
    for doc in docs:

        source_file = doc.metadata.get(
            "source_file",
            os.path.basename(doc.metadata.get("source", "Unknown"))
        )

        page_number = doc.metadata.get(
            "page_number",
            doc.metadata.get("page", "Unknown")
        )

        source_info = {
            "document": source_file,
            "page": page_number
        }

        if source_file not in sources:
            sources.append(source_file)

        if source_info not in source_details:
            source_details.append(source_info)

        context_parts.append(
            f"""
Source: {source_file}
Page: {page_number}

Content:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # Grounded response prompt
    prompt = f"""
You are an expert Network Troubleshooting Assistant.

Answer the user's networking question using ONLY
the information provided in the retrieved context.

Rules:
1. Answer using only the retrieved context.
2. Do not introduce external networking knowledge.
3. Do not present assumptions or inferences as confirmed facts.
4. Clearly distinguish documented facts from reasonable inferences.
5. If the context does not explain a cause or solution,
   explicitly say that the information is unavailable.
6. Provide concise troubleshooting steps only when supported
   by the retrieved context.
7. Mention the source document and page number for supported facts.
8. Never invent commands, configurations, causes, or solutions.

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
        "source_details": source_details,
        "retrieved_chunks": [
            doc.page_content for doc in docs
        ]
    }

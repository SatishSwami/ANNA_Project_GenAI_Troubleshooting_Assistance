import os
import streamlit as st

from functools import lru_cache
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

load_dotenv()

# Configuration
FAISS_DIRECTORY = "faiss_index_v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def load_rag_pipeline():

    print("Loading RAG pipeline...")

    # Step 1: Validate API key
    import streamlit as st

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Check your .env file."
        )

    # Step 2: Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Step 3: Load new FAISS index
    if not os.path.exists(FAISS_DIRECTORY):
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_DIRECTORY}. "
            "Run ingest.py first."
        )

    vectorstore = FAISS.load_local(
        FAISS_DIRECTORY,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Step 4: Configure retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Step 5: Initialize Gemini
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

    # Step 1: Get loaded pipeline
    retriever, llm = load_rag_pipeline()

    # Step 2: Retrieve relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": (
                "The information is not available "
                "in the knowledge base."
            ),
            "sources": [],
            "source_details": [],
            "retrieved_chunks": []
        }

    # Step 3: Prepare context with source information
    context_parts = []

    sources = []
    source_details = []

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

        # Keep unique source filenames
        if source_file not in sources:
            sources.append(source_file)

        # Keep unique document and page details
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

    # Step 4: Construct prompt
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

    # Step 5: Generate answer
    response = llm.invoke(prompt)

    # Step 6: Return answer and sources
    return {
        "answer": response.content,
        "sources": sources,
        "source_details": source_details,
        "retrieved_chunks": [
            doc.page_content for doc in docs
        ]
    }

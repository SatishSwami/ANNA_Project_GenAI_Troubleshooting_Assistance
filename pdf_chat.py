
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def ask_pdf_question(vectorstore, question):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = None

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not configured.")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    prompt = f"""
You are an intelligent PDF assistant.

Use ONLY the provided context.

If the answer is not present, say:
"The information is not available in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "chunks": [
            doc.page_content for doc in docs
        ]
    }

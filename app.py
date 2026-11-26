
import streamlit as st
import hashlib

from rag_pipeline import get_answer
from pdf_ingest import create_pdf_vectorstore
from pdf_chat import ask_pdf_question


# Page configuration
st.set_page_config(
    page_title="GenAI Network Troubleshooting Assistant",
    page_icon="🌐",
    layout="wide"
)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_vectorstore" not in st.session_state:
    st.session_state.pdf_vectorstore = None

if "uploaded_pdf_hash" not in st.session_state:
    st.session_state.uploaded_pdf_hash = None


# Application header
st.title("🌐 GenAI Network Troubleshooting Assistant")

st.markdown(
    """
    Retrieve networking knowledge or chat with your own PDF documents using
    RAG, FAISS, and Gemini.
    """
)


# Sidebar
with st.sidebar:

    st.header("Mode")

    mode = st.radio(
        "Choose Mode",
        [
            "Knowledge Base",
            "Upload PDF"
        ]
    )

    st.divider()

    show_context = st.checkbox(
        "Show Retrieved Context"
    )

    st.header("Example Questions")

    examples = [
        "Why does a DHCP client fail to obtain an IP address?",
        "What happens when a DHCP server sends a DHCPNACK?",
        "How can LAN switching connectivity problems be troubleshot?",
        "What are common TCP/IP connectivity problems?",
        "How can asymmetric paths be diagnosed?"
    ]

    for q in examples:
        st.markdown(f"• {q}")

    st.divider()

    st.header("Project Info")

    st.markdown("""
    - LLM: Gemini 2.5 Flash
    - Vector DB: FAISS
    - Embeddings: MiniLM-L6-v2
    - Framework: LangChain
    - Knowledge Sources: 3 Networking PDFs
    """)

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# PDF upload mode
if mode == "Upload PDF":

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_pdf:

        pdf_bytes = uploaded_pdf.getvalue()

        current_pdf_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        # Process only when a new or different PDF is uploaded
        if (
            st.session_state.uploaded_pdf_hash
            != current_pdf_hash
        ):

            with open("temp_uploaded.pdf", "wb") as f:
                f.write(pdf_bytes)

            with st.spinner("Processing uploaded PDF..."):

                try:
                    st.session_state.pdf_vectorstore = (
                        create_pdf_vectorstore(
                            "temp_uploaded.pdf"
                        )
                    )

                    st.session_state.uploaded_pdf_hash = (
                        current_pdf_hash
                    )

                    st.success(
                        f"Successfully indexed: {uploaded_pdf.name}"
                    )

                except Exception as e:

                    st.session_state.pdf_vectorstore = None
                    st.session_state.uploaded_pdf_hash = None

                    st.error(
                        f"PDF processing failed: {e}"
                    )


# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("sources"):

                st.markdown("**Sources:**")

                for source in message["sources"]:
                    st.write(f"• {source}")

            if message.get("source_details"):

                st.markdown("**Source Pages:**")

                for detail in message["source_details"]:

                    st.write(
                        f"• {detail['document']} — "
                        f"Page {detail['page']}"
                    )

            if show_context and message.get("retrieved_chunks"):

                st.markdown("---")
                st.markdown("### Retrieved Context")

                for i, chunk in enumerate(
                    message["retrieved_chunks"],
                    start=1
                ):

                    with st.expander(
                        f"Retrieved Chunk {i}"
                    ):
                        st.code(chunk)


# Chat input
placeholder = (
    "Ask about the uploaded PDF..."
    if mode == "Upload PDF"
    else "Ask a networking troubleshooting question..."
)

prompt = st.chat_input(placeholder)


# Process user question
if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching knowledge base..."):

                # Permanent networking knowledge base
                if mode == "Knowledge Base":

                    result = get_answer(prompt)

                # User-uploaded PDF
                else:

                    if st.session_state.pdf_vectorstore is None:

                        st.error(
                            "Please upload and process a PDF first."
                        )

                        st.stop()

                    result = ask_pdf_question(
                        st.session_state.pdf_vectorstore,
                        prompt
                    )

                    result["sources"] = [
                        "Uploaded PDF"
                    ]

                    result["retrieved_chunks"] = result.get(
                        "chunks", []
                    )

                    result["source_details"] = []

            # Display answer
            st.markdown(result["answer"])

            # Display sources
            st.markdown("**Sources:**")

            for source in result.get("sources", []):
                st.write(f"• {source}")

            # Display document and page references
            if result.get("source_details"):

                st.markdown("**Source Pages:**")

                for detail in result["source_details"]:

                    st.write(
                        f"• {detail['document']} — "
                        f"Page {detail['page']}"
                    )

            # Display retrieved context
            if show_context:

                st.markdown("---")
                st.markdown("### Retrieved Context")

                for i, chunk in enumerate(
                    result.get("retrieved_chunks", []),
                    start=1
                ):

                    with st.expander(
                        f"Retrieved Chunk {i}"
                    ):
                        st.code(chunk)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "source_details": result.get(
                        "source_details", []
                    ),
                    "retrieved_chunks": result.get(
                        "retrieved_chunks", []
                    )
                }
            )

        except Exception as e:

            st.error(
                f"An error occurred while generating the answer: {e}"
            )
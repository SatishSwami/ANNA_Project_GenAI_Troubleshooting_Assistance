
# ingest.py

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Project paths
PDF_DIRECTORY = "data/pdfs"

# Use a new index initially to protect the existing index
FAISS_DIRECTORY = "faiss_index_v2"


def build_index():

    print("\n========== PDF INGESTION STARTED ==========\n")

    # Step 1: Check PDF directory
    if not Path(PDF_DIRECTORY).exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIRECTORY}"
        )

    pdf_files = list(Path(PDF_DIRECTORY).glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in data/pdfs")

    print(f"Found {len(pdf_files)} PDF documents")

    # Step 2: Load PDF documents
    print("\nLoading PDF documents...")

    loader = DirectoryLoader(
        PDF_DIRECTORY,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=False
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No pages could be extracted from PDFs")

    print(f"Successfully loaded {len(documents)} pages")

    # Step 3: Preserve source and page metadata
    for document in documents:

        document.metadata["source_file"] = Path(
            document.metadata["source"]
        ).name

        # PyPDFLoader page metadata is zero-indexed
        document.metadata["page_number"] = (
            document.metadata.get("page", 0) + 1
        )

    # Step 4: Text chunking
    print("\nSplitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks generated from PDFs")

    print(f"Created {len(chunks)} chunks")

    # Step 5: Initialize embedding model
    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 6: Create FAISS vector database
    print("\nGenerating embeddings and building FAISS index...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Step 7: Save new FAISS index
    vectorstore.save_local(FAISS_DIRECTORY)

    print("\n========== INGESTION COMPLETED ==========")
    print(f"PDF documents: {len(pdf_files)}")
    print(f"Extracted pages: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"FAISS index location: {FAISS_DIRECTORY}")

    print("\nSource document distribution:")

    for pdf in pdf_files:
        count = sum(
            1 for doc in documents
            if doc.metadata["source_file"] == pdf.name
        )

        print(f"{pdf.name}: {count} pages")

    print("\nFAISS index created successfully!")


if __name__ == "__main__":
    build_index()

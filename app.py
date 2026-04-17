import streamlit as st
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# UI
st.set_page_config(page_title="AI RAG App", layout="wide")
st.title("📄 AI RAG Application (Final)")
st.write("Ask questions from your PDFs")

# Session state
if "db" not in st.session_state:
    st.session_state.db = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# Load PDFs
def load_docs():
    docs = []
    if not os.path.exists("data"):
        st.error("❌ 'data' folder not found!")
        return docs

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            docs.extend(loader.load())

    return docs

# Create Vector DB
def create_db():
    docs = load_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(chunks, embeddings)

    return db, chunks

# Button: Process PDFs
if st.button("Process PDFs"):
    with st.spinner("Processing PDFs..."):
        db, chunks = create_db()
        st.session_state.db = db
        st.session_state.chunks = chunks
        st.success("✅ Vector Database Created Successfully!")

# Input
query = st.text_input("Enter your question:")

# Answer
if st.button("Get Answer"):

    if not query:
        st.warning("Please enter a question")

    elif st.session_state.db is None:
        st.error("Please click 'Process PDFs' first")

    else:
        query_lower = query.lower()

        docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])

        # 🔵 Aggregated Questions
        if any(k in query_lower for k in ["total", "count", "how many"]):
            unique_docs = set([
                os.path.basename(c.metadata['source'])
                for c in st.session_state.chunks
            ])
            st.subheader("Answer:")
            st.success(f"📊 Total documents: {len(unique_docs)}")

        # 🟢 Smart Extraction (BEST PART)
        else:
            answer = ""

            # Supplier / Vendor
            if "supplier" in query_lower or "vendor" in query_lower:
                match = re.search(r"VENDOR/SUPPLIER.*\n(.+)", context)
                if match:
                    answer = match.group(1).strip()

            # GST Number
            elif "gst" in query_lower:
                match = re.search(r"GSTIN[:\s]*([0-9A-Z]+)", context)
                if match:
                    answer = match.group(1)

            # Purchase Order
            elif "purchase order" in query_lower or "po" in query_lower:
                match = re.search(r"PO[/\-\d]+", context)
                if match:
                    answer = match.group(0)

            # Cost / MRC
            elif "mrc" in query_lower or "cost" in query_lower:
                match = re.search(r"₹\s?\d+", context)
                if match:
                    answer = match.group(0)

            # Final Output
            if answer:
                st.subheader("Answer:")
                st.success(answer)
            else:
                st.warning("⚠️ Exact answer not found. Showing relevant data:")
                for doc in docs:
                    st.write(doc.page_content[:300])
                    st.write("---")
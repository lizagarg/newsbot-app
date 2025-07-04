import os
import streamlit as st
import pickle
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.title("📰 News Research App")
st.sidebar.title("🔍 Enter News Article URLs")

# Sidebar: Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("📥 Process URLs")

main_placeholder = st.empty()

# If user clicks process
if process_url_clicked and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("📄 Loading data from URLs...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    main_placeholder.text("✂️ Splitting data into chunks...")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("💾 Creating vectorstore...")

    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
        main_placeholder.text("✅ Vectorstore created and saved!")

# Main input
query = st.text_input("💬 Ask your question:")

if query.strip():
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=GoogleGenerativeAI(model="gemini-1.5-flash"),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
            return_source_documents=True
        )

        result = chain({"question": query}, return_only_outputs=True)

        if result and "answer" in result:
            st.header("🧠 Answer")
            st.subheader(result["answer"])

        from difflib import SequenceMatcher

        st.header("🔗 Sources (used in answer)")

        answer_text = result["answer"].lower()
        used_sources = set()

        # Include source only if the content significantly overlaps with the answer
        for doc in result.get("source_documents", []):
            doc_text = doc.page_content.lower()
            if SequenceMatcher(None, doc_text, answer_text).ratio() > 0.1:
                source = doc.metadata.get("source")
                if source:
                    used_sources.add(source)

        if used_sources:
            for source in used_sources:
                domain = urlparse(source).netloc
                st.markdown(f"- [{domain}]({source})")
        else:
            st.markdown("No clearly matching source found.")
            

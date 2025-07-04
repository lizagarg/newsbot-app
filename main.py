import os
import streamlit as st
import pickle
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

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

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
            return_source_documents=True
        )

        result = chain.invoke({"question": query})  # 👈 use invoke instead of call

        if result and "answer" in result:
            st.header("🧠 Answer")
            st.subheader(result["answer"])
        else:
            st.warning("⚠️ No answer received. Please try again.")

        st.header("🔗 Sources (used in answer)")

        sources = set()
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source")
            if source:
                sources.add(source)

        if sources:
            for source in sources:
                domain = urlparse(source).netloc
                st.markdown(f"- [{domain}]({source})")
        else:
            st.markdown("No sources found.")

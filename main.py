import os
import streamlit as st
import pickle
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
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
# if process_url_clicked and urls:
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("📄 Loading data from URLs...")
#     data = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
#     )
#     main_placeholder.text("✂️ Splitting data into chunks...")
#     docs = text_splitter.split_documents(data)

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("💾 Creating vectorstore...")

#     with open("vectorstore.pkl", "wb") as f:
#         pickle.dump(vectorstore, f)
#         main_placeholder.text("✅ Vectorstore created and saved!")


# if process_url_clicked and urls:
#     main_placeholder.text("📄 Loading data from URLs...")
    
#     try:
#         # WebBaseLoader is much more robust than UnstructuredURLLoader
#         loader = WebBaseLoader(urls)
#         data = loader.load()
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         st.stop()

#     # Debug: Check if data actually contains text
#     if not data or not data[0].page_content:
#         st.error("❌ No text found! The articles might be behind a paywall or require JavaScript.")
#         st.stop()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
#     )
#     main_placeholder.text("✂️ Splitting data into chunks...")
#     docs = text_splitter.split_documents(data)

#     if len(docs) > 0:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         main_placeholder.text("💾 Creating vectorstore...")

#         with open("vectorstore.pkl", "wb") as f:
#             pickle.dump(vectorstore, f)
#             main_placeholder.text("✅ Vectorstore created and saved!")
#     else:
#         st.error("❌ Text splitter returned empty chunks. The website content couldn't be parsed.")




# IMPORTS NEEDED FOR CUSTOM LOADING
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

# If user clicks process
if process_url_clicked and urls:
    main_placeholder.text("📄 Loading data from URLs...")
    
    # 1. Custom Scraping Logic (More Robust)
    raw_docs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Check for HTTP errors
            
            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract text (stripping out excessive whitespace)
            text_content = soup.get_text(separator=" ", strip=True)
            
            # Only add if we found substantial text
            if len(text_content) > 500:
                raw_docs.append(Document(page_content=text_content, metadata={"source": url}))
            else:
                st.warning(f"⚠️ Skipped {url}: Content too short or blocked.")
                
        except Exception as e:
            st.error(f"❌ Error fetching {url}: {e}")

    if not raw_docs:
        st.error("❌ Failed to load any content. The websites might be 100% JavaScript (SPA) or blocking scraping.")
        st.stop()

    # 2. Split Data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    main_placeholder.text("✂️ Splitting data into chunks...")
    docs = text_splitter.split_documents(raw_docs)

    # 3. Create Vector Store
    if docs:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("💾 Creating vectorstore...")

        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
            main_placeholder.text("✅ Vectorstore created and saved!")
    else:
        st.error("❌ Text splitter returned empty chunks. No usable text found.")





# Main input
query = st.text_input("💬 Ask your question:")

if query.strip():
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                google_api_key=st.secrets["GOOGLE_API_KEY"]  # ADD THIS
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
            return_source_documents=True
        )

        result = chain({"question": query}, return_only_outputs=True)

        if result and "answer" in result:
            st.header("🧠 Answer")
            #st.markdown(f"**{result['answer']}**")

            answer_html = result["answer"].replace("\n", "<br>")
            st.markdown(f'<div style="font-size: 20px; font-weight: bold;">{answer_html}</div>', 
            unsafe_allow_html=True)

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

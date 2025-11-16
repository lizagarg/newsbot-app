import os
import streamlit as st
import pickle
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load API Keys
load_dotenv()

# --- CONSTANTS ---
MAX_ARTICLES_LIMIT = 10000  # Hard limit for the Knowledge Base

st.title("📰 News Research App")
st.sidebar.title("🔍 Enter News Article URLs")

# Sidebar: Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

# Sidebar: Buttons
process_url_clicked = st.sidebar.button("📥 Process URLs")
st.sidebar.markdown("---")
# New button for managing the database
reset_db_clicked = st.sidebar.button("🗑️ Reset Database")

main_placeholder = st.empty()

# --- HELPER 1: GOOGLE REDIRECT CLEANER ---
def get_clean_url(url):
    """Extracts the real URL from a Google Redirect link."""
    if "google.com/url" in url:
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            if "url" in query_params:
                return query_params["url"][0]
        except:
            return url
    return url

# --- HELPER 2: COUNT ARTICLES ---
def get_current_article_count(vectorstore):
    """Counts unique source URLs in the vectorstore."""
    if not hasattr(vectorstore, "docstore") or not hasattr(vectorstore.docstore, "_dict"):
        return 0
    
    unique_sources = set()
    for doc_id, doc in vectorstore.docstore._dict.items():
        if "source" in doc.metadata:
            unique_sources.add(doc.metadata["source"])
    
    return len(unique_sources)

# --- LOGIC 1: RESET DATABASE ---
if reset_db_clicked:
    if os.path.exists("vectorstore.pkl"):
        os.remove("vectorstore.pkl")
        main_placeholder.success("✅ Database reset! Old data deleted.")
        time.sleep(1) 
        st.rerun()
    else:
        main_placeholder.warning("⚠️ Database is already empty.")

# --- LOGIC 2: PROCESS URLS (WebBaseLoader + Limit Check) ---
if process_url_clicked and urls:
    # STEP 0: CHECK STORAGE LIMIT
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            temp_vectorstore = pickle.load(f)
            current_count = get_current_article_count(temp_vectorstore)
            
            if current_count >= MAX_ARTICLES_LIMIT:
                st.error(f"⛔ STORAGE LIMIT REACHED! ({current_count}/{MAX_ARTICLES_LIMIT} articles)")
                st.warning("Please click 'Reset Database' to clear space for new articles.")
                st.stop()

    main_placeholder.text("📄 Loading data from URLs...")
    
    # STEP 1: Clean URLs (Fix Google Redirects)
    clean_urls = [get_clean_url(url) for url in urls]

    try:
        # STEP 2: Load Data using WebBaseLoader
        # We add headers to the requests_kwargs to prevent 403 Forbidden errors
        loader = WebBaseLoader(clean_urls)
        loader.requests_kwargs = {
            'headers': {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        }
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        data = []

    # STEP 3: Validate Data
    if not data:
        st.error("❌ No data found. The URLs might be blocking the scraper.")
        st.stop()

    # STEP 4: Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    main_placeholder.text("✂️ Splitting data into chunks...")
    docs = text_splitter.split_documents(data)

    # STEP 5: Update Vector Store (Append or Create)
    if docs:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if os.path.exists("vectorstore.pkl"):
            main_placeholder.text("🔄 Updating existing database...")
            with open("vectorstore.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            vectorstore.add_documents(docs)
        else:
            main_placeholder.text("🆕 Creating new database...")
            vectorstore = FAISS.from_documents(docs, embeddings)

        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
            
            new_count = get_current_article_count(vectorstore)
            main_placeholder.success(f"✅ Vectorstore updated! (Total Articles: {new_count}/{MAX_ARTICLES_LIMIT})")
    else:
        st.error("❌ Text splitter returned empty chunks.")

# --- LOGIC 3: RAG Q&A ---
query = st.text_input("💬 Ask your question:")

if query.strip():
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )

        # --- CUSTOM PROMPT FOR STRICT ANSWERS ---
        template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If the answer to the question is not contained in the provided text, strictly say "Info regarding this topic not present". 
DO NOT try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
        
        PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
        chain_type_kwargs = {"prompt": PROMPT}

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        result = chain({"question": query}, return_only_outputs=True)

        # Display Answer
        if result and "answer" in result:
            answer = result["answer"]
            st.header("🧠 Answer")
            answer_html = answer.replace("\n", "<br>")
            st.markdown(
                f'<div style="font-size: 20px; font-weight: bold;">{answer_html}</div>', 
                unsafe_allow_html=True
            )

            # Display Sources Logic
            st.header("🔗 Sources (used in answer)")
            
            # 1. Check if the answer indicates data was missing
            if "Info regarding this topic not present" in answer:
                st.markdown("No sources found.")
            
            # 2. Otherwise, display the sources found
            else:
                sources = set()
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        source = doc.metadata.get("source")
                        if source:
                            sources.add(source)

                if sources:
                    for source in sources:
                        domain = urlparse(source).netloc
                        st.markdown(f"- [{domain}]({source})")
                else:
                    st.markdown("No sources found.")
            
    else:
        # Custom message for empty database
        st.info("ℹ️ Database is empty. Please process URLs to get answers.")

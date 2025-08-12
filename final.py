import streamlit as st
import mysql.connector
import uuid
import hashlib
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
from datetime import datetime
import json
import ollama
from numpy.linalg import norm

# MySQL credentials
HOST     = "localhost"   # or "127.0.0.1"
USER     = "root"        
PASSWORD = "Vijeta@2024"  
DATABASE = "pdf_chatbot" 

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "menu" not in st.session_state:
    st.session_state.menu = "Login"

menu_options = ["Login", "Register", "Upload PDFs"]
if "pending_redirect" in st.session_state:
    st.session_state.menu = st.session_state.pending_redirect
    del st.session_state["pending_redirect"]

menu_selection = st.sidebar.selectbox("Menu", menu_options, 
                                      index=menu_options.index(st.session_state.menu))
st.session_state.menu = menu_selection

# Connect to MySQL
@st.cache_resource
def connect_mysql():
    conn = mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    cursor.execute(f"USE {DATABASE}")

    # Create tables if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS USER_CREDENTIALS (
            USERS VARCHAR(255) PRIMARY KEY,
            PASSWORD VARCHAR(255)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DOCUMENT_FILES (
            ID INT AUTO_INCREMENT PRIMARY KEY,
            EXTRACTED_LAYOUT LONGTEXT,
            FILE_URL TEXT,
            PROCESSED_AT DATETIME,
            RELATIVE_PATH TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DOCUMENT_CHUNKS (
            ID INT AUTO_INCREMENT PRIMARY KEY,
            CHUNK LONGTEXT,
            CHUNK_ID VARCHAR(255),
            CHUNK_INDEX INT,
            EMBEDDING LONGTEXT,
            FILE_URL TEXT,
            LANGUAGE VARCHAR(10),
            RELATIVE_PATH TEXT
        )
    """)
    conn.commit()
    return conn

conn = connect_mysql()
c = conn.cursor()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Page title
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot")

# --- Register ---
if menu_selection == "Register":
    st.subheader("Register")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Create Account"):
        if new_user and new_pass:
            hashed_pw = hash_password(new_pass)
            try:
                c.execute("INSERT INTO USER_CREDENTIALS (USERS, PASSWORD) VALUES (%s, %s)", (new_user, hashed_pw))
                conn.commit()
                st.success("Account created successfully! Please log in.")
                st.session_state.pending_redirect = "Login"
                st.rerun()
            except Exception as e:
                st.error(f"Registration failed: {e}")
        else:
            st.warning("Enter both username and password.")

# --- Login ---
elif menu_selection == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            hashed_pw = hash_password(password)
            c.execute("SELECT * FROM USER_CREDENTIALS WHERE USERS=%s AND PASSWORD=%s", (username, hashed_pw))
            result = c.fetchone()
            if result:
                st.success("Logged in successfully!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.pending_redirect = "Upload PDFs"
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            st.warning("Enter both username and password.")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_model()

# Store already processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader"
)

if uploaded_files:
    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            continue

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        st.write(f" **Processing file:** `{file.name}`")
        st.write(f"   - Extracted {len(text)} characters of text")

        file_url = file.name
        now = datetime.now()
        c.execute("""
            INSERT INTO DOCUMENT_FILES (EXTRACTED_LAYOUT, FILE_URL, PROCESSED_AT, RELATIVE_PATH) 
            VALUES (%s, %s, %s, %s)
        """, (text, file_url, now, "N/A"))
        conn.commit()

        CHUNK_SIZE = 500
        OVERLAP = 100
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - OVERLAP

        st.write(f"   - Created {len(chunks)} chunks")

        embeddings = model.encode(chunks).tolist()
        st.write(f"   - Embedding vector length: {len(embeddings[0])}")

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            embedding_json = json.dumps(embedding)
            c.execute("""
                INSERT INTO DOCUMENT_CHUNKS (CHUNK, CHUNK_ID, CHUNK_INDEX, EMBEDDING, FILE_URL, LANGUAGE, RELATIVE_PATH) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (chunk, chunk_id, i, embedding_json, file_url, "en", "N/A"))
        conn.commit()

        st.write(f" **Inserted into MySQL:** `{file.name}`\n---")
        st.session_state.processed_files.add(file.name)

    st.success("All new files processed and stored successfully.")

# --- Chatbot Search ---
st.header("Chat with your PDFs")
query = st.text_input("Ask a question about the uploaded documents:")

if query:
    query_embedding = model.encode([query])[0]

    c.execute("SELECT CHUNK, EMBEDDING FROM DOCUMENT_CHUNKS")
    rows = c.fetchall()

    scored_chunks = []
    for chunk, embedding_json in rows:
        emb = np.array(json.loads(embedding_json))
        similarity = np.dot(query_embedding, emb) / (norm(query_embedding) * norm(emb))
        scored_chunks.append((chunk, similarity))

    top_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:5]
    context_text = "\n\n".join([chunk for chunk, _ in top_chunks])

    prompt = f"Answer the question based only on the following context:\n\n{context_text}\n\nQuestion: {query}"
    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("Answer:")
    st.write(response["message"]["content"])

import os
import re
import hashlib
from typing import List, Optional

import psycopg2
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "company_rag"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

if not DB_CONFIG["user"] or not DB_CONFIG["password"]:
    raise RuntimeError(
        "DB_USER or DB_PASSWORD is missing in .env"
    )


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

st.set_page_config(page_title="AI Knowledge Base", page_icon="📚", layout="wide")
st.title("📚 AI Knowledge Base - Annual Report RAG")
#st.write("Ask questions about your Annual Report. Answers are generated only from the uploaded documents.")

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing in .env")
        st.stop()
    return Groq(api_key=GROQ_API_KEY)

embedding_model = load_embedding_model()
groq_client = get_groq_client()

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

def get_or_create_company(company_name: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM companies WHERE company_name = %s", (company_name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("INSERT INTO companies (company_name) VALUES (%s) RETURNING id", (company_name,))
        company_id = cur.fetchone()[0]
        conn.commit()
        return company_id
    finally:
        cur.close()
        conn.close()

def save_pdf_to_db(filename: str, full_text: str, company_name: str, total_pages: int) -> int:
    file_hash = hashlib.sha256(full_text.encode()).hexdigest()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check duplicate
        cur.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
        if cur.fetchone():
            return -1

        company_id = get_or_create_company(company_name)
        cur.execute(
            """
            INSERT INTO documents (company_id, filename, file_hash, total_pages, status)
            VALUES (%s, %s, %s, %s, 'completed') RETURNING id
            """,
            (company_id, filename, file_hash, total_pages)
        )
        doc_id = cur.fetchone()[0]
        chunks = chunk_text(full_text)
        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            cur.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, index, chunk, embedding)
            )
        conn.commit()
        return len(chunks)
    finally:
        cur.close()
        conn.close()

def fetch_documents() -> List[tuple]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT d.id, d.filename, c.company_name 
            FROM documents d
            LEFT JOIN companies c ON d.company_id = c.id
            ORDER BY d.created_at DESC
        """)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

def search_chunks(question: str, doc_id: Optional[int] = None) -> List[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        q_emb = get_embedding(question)
        if doc_id:
            cur.execute(
                """
                SELECT content FROM document_chunks
                WHERE document_id = %s
                ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (doc_id, q_emb, TOP_K)
            )
        else:
            cur.execute(
                """
                SELECT content FROM document_chunks
                ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (q_emb, TOP_K)
            )
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

def save_chat(question: str, answer: str, doc_ids: List[int]):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO chat_history (question, answer, source_document_ids) VALUES (%s, %s, %s)",
            (question, answer, doc_ids)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def fix_year_value_pairs(context: str) -> str:
    # Pattern: VALUE YEAR VALUE YEAR -> YEAR: VALUE | YEAR: VALUE
    # Example: "55,691mn 2023 32,622mn 2022" -> "2023: 55,691mn | 2022: 32,622mn"
    pattern = r"(\d[\d,\.]*\s*(?:mn|%|KT|bn)?)\s*(20\d{2})\s+(\d[\d,\.]*\s*(?:mn|%|KT|bn)?)\s*(20\d{2})"
    def repl(m):
        val1, year1, val2, year2 = m.groups()
        return f"{year1}: {val1.strip()} | {year2}: {val2.strip()}"
    return re.sub(pattern, repl, context)

def ask_llm(question: str, context: str) -> str:
    context = fix_year_value_pairs(context)

    import re
    years_in_question = re.findall(r'20\d{2}', question)

    if years_in_question:
        year_text = ", ".join(years_in_question)
        year_rule = f"""
        YEAR FILTER RULE:
        User ne Question me ye saal poocha hai: {year_text}
        1. Jawab me SIRF isi saal ki value do.
        2. Baqi saalon ki value bilkul mat likhna.
        3. Agar ye saal {year_text} Context me hai hi nahi, to ye kaho: "This year {year_text} is not available in the report."
        """
    else:
        year_rule = """
        YEAR FILTER RULE:
        User ne koi saal nahi poocha.
        Is liye jitne saal Context me hain, sab de do.
        """

    prompt = f"""You are an Annual Report Assistant.
Answer only from the Context below. Do not hallucinate.

{year_rule}

SAB SE PEHLE - TABLE STRUCTURE ANALYZE KARO:
Tumhe pehle ye samajhna hai ke is PDF me data kis tarah organize kiya gaya hai. Har PDF ka apna method hota hai, is liye koi ek fixed pattern mat maano.

Step 1: Context ko dekh kar pehle ye pata karo:
- Saal kahan likha hai? Kya saal ooper header me hai? Ya left side me pehle column me hai? Ya value ke saath hi likha hai jese "2023 55,691"?
- Value kahan likhi hai? Kya value saal ke neeche hai? Ya saal ke saamne wale column me hai? Ya saal ke baad hai?

Step 2: Jab tumhe structure samajh aa jaye, tab us structure ke hisaab se value ko uske sahi saal se match karo.

Step 3: Kabhi bhi andaze se value ko saal se mat joro. Hamesha pehle us PDF ka table structure dekho, phir match karo.

Rule: Har metric ko alag alag check karo. Ek metric ka structure doosre metric se alag ho sakta hai.

Context:
{context}

Question: {question}

Answer:"""
    res = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return res.choices[0].message.content

# Sidebar

# Sidebar
st.sidebar.header("📤 Upload New PDF")
company_input = st.sidebar.text_input("Company Name", placeholder="e.g. Engro Fertilizers")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None and st.sidebar.button("Ingest PDF"):
    if not company_input.strip():
        st.sidebar.error("Please enter Company Name first.")
    else:
        with st.spinner("Processing PDF..."):
            reader = PdfReader(uploaded_file)
            text = "".join((p.extract_text() or "") for p in reader.pages)
            count = save_pdf_to_db(uploaded_file.name, text, company_input.strip(), len(reader.pages))
            if count == -1:
                st.sidebar.warning("This PDF is already in database.")
            else:
                st.sidebar.success(f"Done! {count} chunks saved.")
                st.rerun()

# Main
st.subheader("💬 Ask a Question")
try:
    docs = fetch_documents()
except Exception as e:
    st.error(f"DB connection failed: {e}")
    docs = []

selected_doc_id = None
selected_doc_name = ""

if docs:
    # All Documents khatam - ab sirf asal files hongi
    doc_options = [f"{d[1]} - {d[2] or 'Unknown'}" for d in docs]

    if len(doc_options) == 1:
        # Agar sirf 1 file hai to select poochna hi nahi, wohi wali khud lag jayegi
        selected = doc_options[0]
        st.success(f"Report loaded: {selected} - Tum seedha sawal pooch sakti ho")
    else:
        # Agar 2 ya zyada files hain, tab hi select karwao taake mix na hon
        # index=0 ka matlab pehli wali pehle se hi select hogi
        selected = st.selectbox(
            "Select a report:",
            options=doc_options,
            index=0
        )

    for d in docs:
        if f"{d[1]} - {d[2] or 'Unknown'}" == selected:
            selected_doc_id = d[0]
            selected_doc_name = selected
            break

    st.success(f"{len(docs)} document(s) loaded.")
else:
    st.warning("No documents found. Upload from sidebar.")

question = st.text_input("Enter your question", placeholder="Example: What was total revenue?")
if st.button("Get Answer") and question:
    with st.spinner("Searching report..."):
        chunks = search_chunks(question, selected_doc_id)
        if not chunks:
            st.warning("No relevant info found.")
        else:
            context = "\n\n".join(chunks)
            answer = ask_llm(question, context)
            save_chat(question, answer, [selected_doc_id] if selected_doc_id else [])
            st.markdown("### ✅ Answer")
            st.write(answer)
            with st.expander("View Source Chunks"):
                for c in chunks:
                    st.write(c[:500])
                    st.divider()
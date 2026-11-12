import os
import html
import time
import re
import hashlib
import json
from typing import List, Optional
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from google import genai
from google.genai import types

# =========================================================
# 1. ENVIRONMENT AND SECRETS
# =========================================================
# Resolve .env relative to this file (works even when Streamlit is
# launched from a different working directory). Streamlit Cloud Secrets
# are checked first by get_secret() below.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_FILE, override=False)

def get_secret(key, default=None):

    """
      Streamlit Secrets take priority.
     Local .env is used as fallback.
    """

    try:

        value = st.secrets.get(key)

        if value not in (None, ""):

            return value

    except Exception:

        pass



    value = os.getenv(key)

    if value not in (None, ""):

        return value



    return default

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = (

    get_secret("GEMINI_API_KEY")

    or get_secret("GOOGLE_API_KEY")
)
DB_HOST = get_secret("DB_HOST", "127.0.0.1")
DB_CONFIG = {

    "host": DB_HOST,
    "port": int(get_secret("DB_PORT", "5433")),
    "dbname": get_secret("DB_NAME", "company_rag"),
    "user": get_secret("DB_USER"),
    "password": get_secret("DB_PASSWORD"),
    "connect_timeout": int(get_secret("DB_CONNECT_TIMEOUT", "10")),
    "sslmode": get_secret(

        "DB_SSLMODE",
        "require" if "neon.tech" in str(DB_HOST) else "prefer"

    ),

}

# =========================================================
# 2. APPLICATION SETTINGS
# =========================================================
GEMINI_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_RETRY_ATTEMPTS = 5
EMBEDDING_BATCH_DELAY_SECONDS = 1.2
APP_VECTOR_TABLE = "document_chunks_gemini"
N8N_VECTOR_TABLE = "n8n_vectors_gemini"

# =========================================================
# 3. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="KnowSphere AI",
    page_icon="📚",
    layout="wide",
)

def load_css():

    css_path = os.path.join(

      os.path.dirname(os.path.abspath(__file__)),
        "style.css"

    )



    if os.path.exists(css_path):

        with open(css_path, "r", encoding="utf-8") as css_file:

            st.markdown(

                f"<style>{css_file.read()}</style>",

                unsafe_allow_html=True

            )



load_css()

st.title("📚 KnowSphere AI")
st.caption(

    "Your AI-powered knowledge base for documents, "
    "research and intelligent question answering."

)

# =========================================================
# 4. API CLIENTS
# =========================================================

@st.cache_resource
def get_gemini_client():

    if not GEMINI_API_KEY:

        raise RuntimeError(

            "GEMINI_API_KEY is missing in Streamlit Secrets or .env."

        )
    return genai.Client(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_groq_client():

    if not GROQ_API_KEY:

        raise RuntimeError(

            "GROQ_API_KEY is missing in Streamlit Secrets or .env."

        )
    return Groq(api_key=GROQ_API_KEY)


gemini_client = get_gemini_client()
groq_client = get_groq_client()

# =========================================================
# 5. DATABASE CONNECTION
# =========================================================

def get_db_connection():

    if not DB_CONFIG["user"] or not DB_CONFIG["password"]:

        raise RuntimeError(

            "DB_USER or DB_PASSWORD is missing."

        )

    return psycopg2.connect(**DB_CONFIG)

def vector_literal(values):

    """

    Convert embedding list to pgvector-compatible text.

    """

    if len(values) != EMBEDDING_DIMENSIONS:

        raise ValueError(

            f"Wrong embedding dimensions: {len(values)}. "

            f"Expected {EMBEDDING_DIMENSIONS}."

        )



    return "[" + ",".join(str(float(v)) for v in values) + "]"

# =========================================================
# 6. GEMINI EMBEDDINGS
# =========================================================

def get_embedding(

    text: str,

    task_type: str = "RETRIEVAL_DOCUMENT"

) -> List[float]:



    if not text or not text.strip():

        raise ValueError("Cannot embed empty text.")



    response = gemini_client.models.embed_content(

        model=GEMINI_MODEL,

        contents=text,

        config=types.EmbedContentConfig(

            task_type=task_type,

            output_dimensionality=EMBEDDING_DIMENSIONS

        )

    )

    embedding = response.embeddings[0].values



    if len(embedding) != EMBEDDING_DIMENSIONS:

        raise ValueError(

            f"Gemini returned {len(embedding)} dimensions "

            f"instead of {EMBEDDING_DIMENSIONS}."

        )



    return embedding

def get_embeddings_batch(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> List[List[float]]:
    """Create Gemini embeddings in small batches with 429-aware retries."""
    if not texts:
        return []
    if any(not isinstance(item, str) or not item.strip() for item in texts):
        raise ValueError("Embedding input contains an empty or invalid text chunk.")

    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        last_error = None

        for attempt in range(EMBEDDING_RETRY_ATTEMPTS):
            try:
                response = gemini_client.models.embed_content(
                    model=GEMINI_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIMENSIONS
                    )
                )
                batch_embeddings = [item.values for item in response.embeddings]
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        "Gemini returned an unexpected number of embeddings."
                    )
                for embedding in batch_embeddings:
                    if len(embedding) != EMBEDDING_DIMENSIONS:
                        raise ValueError("Embedding dimension mismatch.")
                all_embeddings.extend(batch_embeddings)
                last_error = None
                break
            except Exception as error:
                last_error = error
                message = str(error).lower()
                is_rate_limit = (
                    "429" in message
                    or "resource_exhausted" in message
                    or "quota exceeded" in message
                )
                if not is_rate_limit or attempt == EMBEDDING_RETRY_ATTEMPTS - 1:
                    raise

                retry_match = re.search(r"retry in ([0-9.]+)s", message)
                wait_seconds = (
                    float(retry_match.group(1))
                    if retry_match else min(5 * (2 ** attempt), 60)
                )
                wait_seconds = min(max(wait_seconds, 2), 65)
                st.info(
                    f"Gemini rate limit reached. Retrying in "
                    f"{int(wait_seconds)} seconds..."
                )
                time.sleep(wait_seconds)

        if last_error is not None:
            raise last_error

        if start + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(EMBEDDING_BATCH_DELAY_SECONDS)

    return all_embeddings
# =========================================================
# 7. DATABASE TABLE INITIALIZATION
# =========================================================

def ensure_database_schema():



    conn = get_db_connection()

    cur = conn.cursor()



    try:

        cur.execute(

            "CREATE EXTENSION IF NOT EXISTS vector;"

        )
        # New Gemini table.
        # Existing 384-dimensional document_chunks is NOT changed.



        cur.execute(f"""

            CREATE TABLE IF NOT EXISTS {APP_VECTOR_TABLE} (

                id BIGSERIAL PRIMARY KEY,

                document_id INTEGER NOT NULL

                    REFERENCES documents(id) ON DELETE CASCADE,

                chunk_index INTEGER NOT NULL,

                page_number INTEGER,

                content TEXT NOT NULL,

                embedding vector(3072) NOT NULL,

                created_at TIMESTAMPTZ DEFAULT NOW(),

                UNIQUE(document_id, chunk_index)

            );

        """)



        # AI-generated document metadata; existing tables and vectors are preserved.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_insights (
                document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                subject TEXT,
                topics JSONB NOT NULL DEFAULT '[]'::jsonb,
                organization TEXT,
                summary TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Chat history table.

        cur.execute("""

            CREATE TABLE IF NOT EXISTS chat_history (

                id BIGSERIAL PRIMARY KEY,

                question TEXT NOT NULL,

                answer TEXT NOT NULL,

                source_document_ids INTEGER[],

                created_at TIMESTAMPTZ DEFAULT NOW()

            );

        """)



        conn.commit()



    except Exception:

        conn.rollback()

        raise



    finally:

        cur.close()

        conn.close()





ensure_database_schema()

# =========================================================
# 8. DOCUMENT CHUNKING
# =========================================================

def chunk_text(text: str) -> List[str]:



    chunks = []



    text = re.sub(r"\s+", " ", text).strip()



    if not text:

        return chunks



    start = 0



    while start < len(text):



        end = start + CHUNK_SIZE



        chunks.append(text[start:end])



        if end >= len(text):

            break



        start = end - CHUNK_OVERLAP



    return chunks





def chunk_pdf_pages(reader):



    result = []

    chunk_index = 0



    for page_number, page in enumerate(reader.pages, start=1):



        page_text = page.extract_text() or ""



        page_chunks = chunk_text(page_text)



        for chunk in page_chunks:



            result.append({

                "chunk_index": chunk_index,

                "page_number": page_number,

                "content": chunk

            })



            chunk_index += 1



    return result

# =========================================================
# 9. COMPANY MANAGEMENT
# =========================================================
def get_or_create_company(

    cur,

    company_name: str

) -> int:



    cur.execute(

        """

        SELECT id

        FROM companies

        WHERE LOWER(company_name) = LOWER(%s)

        """,

        (company_name,)

    )



    row = cur.fetchone()



    if row:

        return row[0]



    cur.execute(

        """

        INSERT INTO companies (company_name)

        VALUES (%s)

        RETURNING id

        """,

        (company_name,)

    )



    return cur.fetchone()[0]
# =========================================================
# 10. PDF INGESTION
# =========================================================

def _extract_json_object(raw_text: str) -> dict:
    """Safely extract the first JSON object from an LLM response."""
    if not raw_text:
        return {}

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character == "{":
            try:
                data, _ = decoder.raw_decode(cleaned[index:])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                continue
    return {}


def classify_document(full_text: str) -> dict:
    """Identify a PDF's subject, topics and explicitly named organization."""
    sample = (full_text or "").strip()[:14000]
    if not sample:
        raise ValueError("The PDF contains no readable text to analyze.")

    prompt = f"""Read the document excerpt and return ONLY one valid JSON object.
Use exactly these keys:
- subject: short broad subject
- topics: array of 3-6 short topics
- organization: company/institution only when explicitly named; otherwise null
- summary: one concise sentence

The document may be about any subject. Do not assume it is a business document.
Do not add Markdown fences or text outside JSON.

DOCUMENT:
{sample}
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    data = _extract_json_object(raw)

    topics = data.get("topics", [])
    if isinstance(topics, str):
        topics = [item.strip() for item in re.split(r"[,;]", topics) if item.strip()]
    if not isinstance(topics, list):
        topics = []

    organization = data.get("organization")
    if isinstance(organization, str):
        organization = organization.strip() or None
    else:
        organization = None

    return {
        "subject": str(data.get("subject") or "Not identified").strip()[:160],
        "topics": [str(item).strip()[:100] for item in topics[:8] if str(item).strip()],
        "organization": str(organization)[:200] if organization else None,
        "summary": str(data.get("summary") or "").strip()[:600],
    }


def save_pdf_to_db(

    filename: str,

    chunks: List[dict],

    company_name: str,

    total_pages: int,

    full_text: str,

    insights: Optional[dict] = None

):



    file_hash = hashlib.sha256(

        full_text.encode("utf-8")

    ).hexdigest()



    conn = get_db_connection()

    cur = conn.cursor()



    try:



        cur.execute(

            """

            SELECT id

            FROM documents

            WHERE file_hash = %s

            """,

            (file_hash,)

        )



        if cur.fetchone():

            conn.rollback()

            return -1



        company_id = get_or_create_company(

            cur,

            company_name

        )



        cur.execute(

            """

            INSERT INTO documents (

                company_id,

                filename,

                file_hash,

                total_pages,

                status

            )

            VALUES (%s, %s, %s, %s, 'processing')

            RETURNING id

            """,

            (

                company_id,

                filename,

                file_hash,

                total_pages

            )

        )



        document_id = cur.fetchone()[0]



        texts = [item["content"] for item in chunks]



        embeddings = get_embeddings_batch(

            texts,

            task_type="RETRIEVAL_DOCUMENT"

        )



        for item, embedding in zip(chunks, embeddings):



            cur.execute(

                f"""

                INSERT INTO {APP_VECTOR_TABLE} (

                    document_id,

                    chunk_index,

                    page_number,

                    content,

                    embedding

                )

                VALUES (%s, %s, %s, %s, %s::vector)

                """,

                (

                    document_id,

                    item["chunk_index"],

                    item["page_number"],

                    item["content"],

                    vector_literal(embedding)

                )

            )



        insights = insights or {}

        cur.execute(
            """
            INSERT INTO document_insights
                (document_id, subject, topics, organization, summary)
            VALUES (%s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                subject = EXCLUDED.subject,
                topics = EXCLUDED.topics,
                organization = EXCLUDED.organization,
                summary = EXCLUDED.summary
            """,
            (document_id, insights.get("subject", "Not analyzed"),
             json.dumps(insights.get("topics", [])),
             insights.get("organization"), insights.get("summary", ""))
        )

        cur.execute(

            """

            UPDATE documents

            SET status = 'completed'

            WHERE id = %s

            """,

            (document_id,)

        )



        conn.commit()



        return len(chunks)



    except Exception:



        conn.rollback()

        raise



    finally:



        cur.close()

        conn.close()

# =========================================================
# 11. OLD 384-DIMENSION CHUNK MIGRATION
# =========================================================

def migrate_old_chunks():



    conn = get_db_connection()

    cur = conn.cursor()



    migrated = 0



    try:



        # Read old records without changing them.



        cur.execute("""

            SELECT

                dc.document_id,

                dc.chunk_index,

                dc.page_number,

                dc.content

            FROM document_chunks dc

            ORDER BY dc.document_id, dc.chunk_index

        """)



        old_chunks = cur.fetchall()



        if not old_chunks:

            return 0



        # Only migrate chunks not already in the new table.



        pending = []



        for row in old_chunks:



            document_id, chunk_index, page_number, content = row



            cur.execute(

                f"""

                SELECT 1

                FROM {APP_VECTOR_TABLE}

                WHERE document_id = %s

                  AND chunk_index = %s

                """,

                (document_id, chunk_index)

            )



            if not cur.fetchone():

                pending.append(row)



        if not pending:

            return 0



        texts = [row[3] for row in pending]



        embeddings = get_embeddings_batch(

            texts,

            task_type="RETRIEVAL_DOCUMENT"

        )



        for row, embedding in zip(pending, embeddings):



            document_id, chunk_index, page_number, content = row



            cur.execute(

                f"""

                INSERT INTO {APP_VECTOR_TABLE} (

                    document_id,

                    chunk_index,

                    page_number,

                    content,

                    embedding

                )

                VALUES (%s, %s, %s, %s, %s::vector)

                ON CONFLICT (document_id, chunk_index)

                DO NOTHING

                """,

                (

                    document_id,

                    chunk_index,

                    page_number,

                    content,

                    vector_literal(embedding)

                )

            )



            migrated += cur.rowcount



        conn.commit()



        return migrated



    except Exception:



        conn.rollback()

        raise



    finally:



        cur.close()

        conn.close()

# =========================================================
# 12. FETCH DOCUMENTS
# =========================================================

def fetch_documents():



    conn = get_db_connection()

    cur = conn.cursor()



    try:



        cur.execute("""

            SELECT

                d.id,

                d.filename,

                c.company_name,

                d.total_pages,

                d.created_at,

                (

                    SELECT COUNT(*)

                    FROM document_chunks_gemini gc

                    WHERE gc.document_id = d.id

                ) AS gemini_chunks,
                COALESCE(di.subject, 'Not analyzed') AS subject,
                COALESCE(di.topics, '[]'::jsonb) AS topics,
                di.organization,
                di.summary
            FROM documents d
            LEFT JOIN companies c ON d.company_id = c.id
            LEFT JOIN document_insights di ON di.document_id = d.id
            ORDER BY d.created_at DESC

        """)



        return cur.fetchall()



    finally:



        cur.close()

        conn.close()

def reanalyze_document(document_id: int) -> dict:
    """Re-run subject/topic classification for an already indexed PDF."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""SELECT content FROM {APP_VECTOR_TABLE}
                WHERE document_id = %s ORDER BY chunk_index""",
            (document_id,),
        )
        rows = cur.fetchall()
        document_text = "\n".join(row[0] for row in rows if row and row[0])
        if not document_text.strip():
            raise ValueError("No indexed text was found for this PDF. Re-upload or re-index it first.")

        insights = classify_document(document_text)
        cur.execute(
            """INSERT INTO document_insights
                (document_id, subject, topics, organization, summary)
               VALUES (%s, %s, %s::jsonb, %s, %s)
               ON CONFLICT (document_id) DO UPDATE SET
                 subject = EXCLUDED.subject,
                 topics = EXCLUDED.topics,
                 organization = EXCLUDED.organization,
                 summary = EXCLUDED.summary,
                 created_at = NOW()""",
            (document_id, insights["subject"], json.dumps(insights["topics"]),
             insights["organization"], insights["summary"]),
        )
        conn.commit()
        return insights
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# =========================================================
# 13. SEARCH APP AND N8N VECTORS
# =========================================================

def search_chunks(

    question: str,

    doc_id: Optional[int] = None

):



    query_embedding = get_embedding(

        question,

        task_type="RETRIEVAL_QUERY"

    )



    query_vector = vector_literal(query_embedding)



    results = []



    conn = get_db_connection()

    cur = conn.cursor()



    try:



        # Search Streamlit document vectors.



        if doc_id is not None:



            cur.execute(

                f"""

                SELECT

                    gc.content,

                    d.filename,

                    gc.page_number,

                    gc.embedding <=> %s::vector AS distance

                FROM {APP_VECTOR_TABLE} gc

                JOIN documents d

                    ON d.id = gc.document_id

                WHERE gc.document_id = %s

                ORDER BY distance

                LIMIT %s

                """,

                (query_vector, doc_id, TOP_K)

            )



        else:



            cur.execute(

                f"""

                SELECT

                    gc.content,

                    d.filename,

                    gc.page_number,

                    gc.embedding <=> %s::vector AS distance

                FROM {APP_VECTOR_TABLE} gc

                JOIN documents d

                    ON d.id = gc.document_id

                ORDER BY distance

                LIMIT %s

                """,

                (query_vector, TOP_K)

            )



        for row in cur.fetchall():



            results.append({

                "content": row[0],

                "source": row[1],

                "page": row[2],

                "distance": float(row[3]),

                "origin": "Streamlit"

            })



        # Search n8n table only when:
        # 1. It exists in Neon.
        # 2. User selected All Sources.
        #
        # This table is created by the n8n PGVector node.
        # This app does not create or overwrite it.



        if doc_id is None:



            cur.execute("""

                SELECT EXISTS (

                    SELECT 1

                    FROM information_schema.tables

                    WHERE table_schema = 'public'

                      AND table_name = %s

                )

            """, (N8N_VECTOR_TABLE,))



            n8n_table_exists = cur.fetchone()[0]



            if n8n_table_exists:



                cur.execute(f"""

                    SELECT

                        text,

                        metadata,

                        embedding <=> %s::vector AS distance

                    FROM {N8N_VECTOR_TABLE}

                    ORDER BY distance

                    LIMIT %s

                """, (query_vector, TOP_K))



                for row in cur.fetchall():



                    metadata = row[1]



                    if isinstance(metadata, dict):

                        source = (

                            metadata.get("source")

                            or metadata.get("fileName")

                            or "n8n document"

                        )

                    else:

                        source = "n8n document"



                    results.append({

                        "content": row[0],

                        "source": str(source),

                        "page": None,

                        "distance": float(row[2]),

                        "origin": "n8n"

                    })



        results.sort(key=lambda item: item["distance"])



        return results[:TOP_K]



    except Exception:



        conn.rollback()

        raise



    finally:



        cur.close()

        conn.close()

# =========================================================
# 14. CHAT HISTORY
# ========================================================

def save_chat(

    question: str,

    answer: str,

    doc_ids: List[int]

):



    conn = get_db_connection()

    cur = conn.cursor()



    try:



        cur.execute(

            """

            INSERT INTO chat_history (

                question,

                answer,

                source_document_ids

            )

            VALUES (%s, %s, %s)

            """,

            (question, answer, doc_ids)

        )



        conn.commit()



    finally:



        cur.close()

        conn.close()

# =========================================================
# 15. YEAR VALUE CORRECTION
# =========================================================
def fix_year_value_pairs(context: str) -> str:



    pattern = (

        r"(\d[\d,\\.]*\s*(?:mn|%|KT|bn)?)"

        r"\s*(20\d{2})\s+"

        r"(\d[\d,\\.]*\s*(?:mn|%|KT|bn)?)"

        r"\s*(20\d{2})"

    )



    def replace(match):



        val1, year1, val2, year2 = match.groups()



        return (

            f"{year1}: {val1.strip()} | "

            f"{year2}: {val2.strip()}"

        )



    return re.sub(

        pattern,

        replace,

        context,

        flags=re.IGNORECASE

    )





# =========================================================

# 16. GROQ ANSWER GENERATION

# =========================================================



def ask_llm(question: str, context: str) -> str:



    context = fix_year_value_pairs(context)



    years = re.findall(

        r"20\d{2}",

        question

    )



    if years:



        year_text = ", ".join(years)



        year_rule = f"""

        YEAR FILTER RULE:



        User ne ye saal poocha hai: {year_text}



        1. Sirf requested year ki value do.

        2. Doosre years ki value mat do.

        3. Agar requested year context mein nahi hai,

           to clearly batao ke year available nahi hai.

        """



    else:



        year_rule = """

        YEAR FILTER RULE:



        User ne koi specific year nahi poocha.

        Context mein available relevant years provide karo.

        """



    prompt = f"""

You are KnowSphere AI, an accurate document research assistant.



Answer ONLY from the supplied context.

Never invent facts, numbers, dates or financial values.



{year_rule}



TABLE ANALYSIS RULE:



Before answering, inspect how the data is organized.



Check:

\- Whether years appear in column headings or rows.

\- Whether values appear below or beside the year.

\- Whether the table uses multiple reporting years.

\- Whether units are mn, bn, %, KT or another unit.



Match each metric to its correct year.

Do not guess table relationships.



If information is missing, say it is not available

in the supplied documents.



CONTEXT:



{context}



QUESTION:



{question}



ANSWER:

"""



    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return "I could not generate an answer from the retrieved document content."
    return answer.strip()
# =========================================================
# 17. SIDEBAR - PDF UPLOAD
# =========================================================

st.sidebar.header("📤 Upload New PDF")
st.sidebar.caption("Add a document from any subject or industry.")
company_input = st.sidebar.text_input(
    "Company / Organization (optional)",
    placeholder="Optional — e.g. Engro Fertilizers"
)
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None and st.sidebar.button("Ingest PDF", type="primary", use_container_width=True):
    try:
        with st.spinner("Reading PDF, analyzing its subject and indexing content..."):
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as decrypt_error:
                    raise ValueError("This PDF is password-protected and could not be opened.") from decrypt_error
            chunks = chunk_pdf_pages(reader)
            full_text = "\n".join(item["content"] for item in chunks)
            if not chunks:
                raise ValueError("No readable text found in this PDF. Scanned PDFs may need OCR first.")
            insights = classify_document(full_text)
            company_name = company_input.strip() or insights.get("organization") or "Unspecified"
            count = save_pdf_to_db(
                filename=uploaded_file.name,
                chunks=chunks,
                company_name=company_name,
                total_pages=len(reader.pages),
                full_text=full_text,
                insights=insights
            )
        if count == -1:
            st.sidebar.warning("This PDF's content is already in your knowledge base.")
        else:
            st.sidebar.success("PDF analyzed and indexed successfully.")
            st.rerun()
    except Exception as error:
        message = str(error)
        if "429" in message or "resource_exhausted" in message.lower() or "quota" in message.lower():
            st.sidebar.error(
                "Gemini embedding quota is temporarily busy. Please wait before retrying. "
                "Your PDF was not fully indexed; it will not be marked as complete."
            )
        else:
            st.sidebar.error(f"PDF processing failed: {message[:500]}")

# Keep migration available, but move it out of the everyday upload flow.
with st.sidebar.expander("Maintenance: old data migration"):
    st.caption("Re-embed old 384-dimensional records. Existing records remain unchanged.")
    if st.button("Re-embed Existing 384 Chunks", key="migrate_old_chunks"):
        try:
            with st.spinner("Migrating old document chunks..."):
                migrated = migrate_old_chunks()
            st.success(f"{migrated} old chunks migrated.")
        except Exception as error:
            st.error(f"Migration failed: {str(error)[:400]}")


# =========================================================
# 19. MAIN DASHBOARD
# =========================================================

try:
    docs = fetch_documents()
except Exception as error:
    st.error(f"Could not load documents from the database: {str(error)[:400]}")
    docs = []

st.markdown("<div class='ks-hero'><div><div class='ks-eyebrow'>YOUR INTELLIGENT KNOWLEDGE WORKSPACE</div><h1>KnowSphere AI</h1><p>One place to explore, understand and ask questions about your documents.</p></div></div>", unsafe_allow_html=True)

# Select a document first so the page count and AI subject refer to that PDF only.
selected_doc_id = None
selected_doc = None
if docs:
    doc_options = {f"{doc[1]}": doc[0] for doc in docs}
    selected_label = st.selectbox("Select a PDF", list(doc_options.keys()), key="selected_pdf")
    selected_doc_id = doc_options[selected_label]
    selected_doc = next((doc for doc in docs if doc[0] == selected_doc_id), None)

col1, col2, col3 = st.columns(3)
col1.markdown(
    f"<div class='ks-stat ks-stat-blue'><div class='ks-stat-label'>Uploaded Documents</div><div class='ks-stat-value'>{len(docs)}</div><div class='ks-stat-note'>Documents in your knowledge base</div></div>",
    unsafe_allow_html=True
)
selected_pages = int(selected_doc[3] or 0) if selected_doc else 0
col2.markdown(
    f"<div class='ks-stat ks-stat-purple'><div class='ks-stat-label'>Pages in Selected PDF</div><div class='ks-stat-value'>{selected_pages if selected_doc else '—'}</div><div class='ks-stat-note'>Page count for this PDF only</div></div>",
    unsafe_allow_html=True
)
subject = (selected_doc[6] if selected_doc else "Select a PDF") or "Not analyzed"
col3.markdown(
    f"<div class='ks-stat ks-stat-mint'><div class='ks-stat-label'>Detected Subject</div><div class='ks-stat-value ks-subject-value'>{html.escape(str(subject))}</div><div class='ks-stat-note'>AI document classification</div></div>",
    unsafe_allow_html=True
)

st.markdown("### 📁 Your Documents")
if docs:
    for doc in docs:
        with st.container():
            topics = doc[7] if len(doc) > 7 else []
            if isinstance(topics, str):
                try: topics = json.loads(topics)
                except Exception: topics = []
            topic_text = " · ".join(str(x) for x in (topics or [])[:5])
            status = "Indexed" if int(doc[5] or 0) > 0 else "Processing required"
            st.markdown(
                f"<div class='ks-document'><div class='ks-document-icon'>PDF</div><div class='ks-document-info'><div class='ks-document-title'>{html.escape(str(doc[1]))}</div><div class='ks-document-meta'>{int(doc[3] or 0)} pages · {html.escape(str(doc[6] or 'Subject not analyzed'))}"
                + (f" · {html.escape(topic_text)}" if topic_text else "")
                + f"</div></div><div class='ks-document-status'>{status}</div></div>",
                unsafe_allow_html=True
            )
else:
    st.info("Your knowledge base is empty. Upload your first PDF from the light-blue sidebar.")

if selected_doc:
    with st.expander("Document insights", expanded=False):
        st.markdown(f"**Subject:** {selected_doc[6] or 'Not analyzed'}")
        if selected_doc[8]:
            st.markdown(f"**Organization identified:** {selected_doc[8]}")
        if selected_doc[9]:
            st.write(selected_doc[9])
        topics = selected_doc[7] or []
        if isinstance(topics, str):
            try: topics = json.loads(topics)
            except Exception: topics = []
        if topics:
            st.markdown("**Topics:** " + " · ".join(str(x) for x in topics))

        if st.button("Analyze Subject Again", key=f"reanalyze_{selected_doc_id}"):
            try:
                with st.spinner("Analyzing this PDF again..."):
                    reanalyze_document(int(selected_doc_id))
                st.success("Document subject and topics updated successfully.")
                st.rerun()
            except Exception as error:
                st.error(f"Re-analysis failed: {str(error)[:400]}")


# =========================================================

# 21. QUESTION ANSWERING

# =========================================================



st.divider()



st.subheader("💬 Ask KnowSphere AI")



question = st.text_input(

    "Enter your question",

    placeholder="Example: What was the total revenue in 2023?"

)



if st.button(

    "Get Answer",

    type="primary"

) and question.strip():



    try:



        with st.spinner(

            "Searching documents and preparing answer..."

        ):



            results = search_chunks(

                question,

                selected_doc_id

            )



            if not results:



                st.warning(
                    "No indexed information was found for this question. "
                    "Please confirm that the selected PDF has finished processing."
                )



            else:



                context_parts = []



                for index, item in enumerate(results, start=1):



                    context_parts.append(

                        f"""

SOURCE {index}

Document: {item['source']}

Page: {item['page'] or 'Not available'}

Origin: {item['origin']}



Content:

{item['content']}

"""

                    )



                context = "\n\n".join(context_parts)



                answer = ask_llm(

                    question,

                    context

                )



                if selected_doc_id is not None:
                    source_ids = [int(selected_doc_id)]
                else:
                    source_ids = []
                    source_names = {item["source"] for item in results}
                    for document in docs:
                        if document[1] in source_names:
                            source_ids.append(int(document[0]))
                    source_ids = list(dict.fromkeys(source_ids))



                save_chat(

                    question,

                    answer,

                    source_ids

                )



                st.markdown("### ✅ Answer")



                st.write(answer)



                with st.expander(

                    "📖 View Sources and Retrieved Chunks"

                ):



                    for item in results:



                        st.markdown(

                            f"**Source:** {item['source']}"

                        )



                        st.caption(

                            f"Origin: {item['origin']} | "

                            f"Page: {item['page'] or 'Not available'}"

                        )



                        st.write(item["content"])



                        st.divider()



    except Exception as error:



        st.error(

            f"Question answering failed: {error}"

        )





# =========================================================

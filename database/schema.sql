CREATE EXTENSION IF NOT EXISTS vector;

-- Company records
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) UNIQUE NOT NULL,
    industry VARCHAR(150),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PDF documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id),
    filename VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) UNIQUE,
    publication_year INTEGER,
    total_pages INTEGER,
    status VARCHAR(30) DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted text chunks and vectors
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id)
        ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User questions and answers
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    source_document_ids INTEGER[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity index
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops);

-- Useful document lookup index
CREATE INDEX IF NOT EXISTS documents_company_idx
ON documents(company_id);

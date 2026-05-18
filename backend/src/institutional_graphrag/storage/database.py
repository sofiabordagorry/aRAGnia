import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123"),
    "database": os.getenv("POSTGRES_DB", "app_db"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            type VARCHAR(20) NOT NULL,        -- 'rag' or 'graph_rag'
            query_text TEXT NOT NULL,
            cypher_query TEXT,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        ALTER TABLE queries
        ADD COLUMN IF NOT EXISTS cypher_query TEXT;
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            query_id INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            score REAL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS graphrag_chunks (
            id SERIAL PRIMARY KEY,
            query_id INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_page INTEGER
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS graphrag_chunk_entities (
            id SERIAL PRIMARY KEY,
            query_id INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            entity_label TEXT NOT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully.")

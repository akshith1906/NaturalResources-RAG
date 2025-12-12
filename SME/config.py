import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv

load_dotenv() # Load .env file

# --- API Keys (MUST BE SET IN ENVIRONMENT) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("GOOGLE_API_KEY and PINECONE_API_KEY environment variables must be set. (Did you create your .env file?)")

# --- Pinecone Serverless ---
PINECONE_INDEX_NAME = "sme-agent-new" 
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# --- LLM Model ---
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
# --- Embedding & Reranker Models ---
SUPPORTED_EMBEDDING_MODELS: Tuple[str, ...] = ("all-mpnet-base-v2", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Ingestion ---
DOCS_PATH = Path("./Docs")
# Hierarchical chunk sizes
CHUNK_SIZES: Tuple[int, ...] = (2048, 512) 
CHUNK_OVERLAP_RATIO = 0.1
BM25_ENCODER_PATH = Path("./bm25_encoder.pkl")

# --- RAG Settings ---
SEARCH_CHUNK_SIZE = 2048 
HYBRID_ALPHA = 0.5
PRE_RERANK_TOP_K = 50
FINAL_TOP_K = 10       

# --- File Output & Logs ---
OUTPUT_DIR = Path("./generated_files")
LOG_DIR = Path("./logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Ingestion Manifest
INGESTION_LOG_PATH = LOG_DIR / "ingest.log"
INGESTION_MANIFEST_PATH = LOG_DIR / "ingestion_manifest.json"

# --- Email (NEW) ---
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_SENDER_EMAIL = os.environ.get("SMTP_SENDER_EMAIL")
SMTP_SENDER_PASSWORD = os.environ.get("SMTP_SENDER_PASSWORD")
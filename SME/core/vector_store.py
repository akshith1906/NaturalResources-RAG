import logging, re, pickle
from typing import Dict, List, Tuple
import numpy as np
from functools import lru_cache

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document

import config
from core.models import EmbeddingManager, Reranker

logger = logging.getLogger("rag.vector_store")

class PineconeRetriever:
    """Handles connection and hybrid search logic for Pinecone."""

    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.embed_mgr = EmbeddingManager(config.SUPPORTED_EMBEDDING_MODELS)
        self.reranker = Reranker(config.RERANKER_MODEL)
        self.bm25_encoder = self._load_bm25_encoder()
        self.index = self._get_index()
        logger.info(f"Connected to Pinecone index '{config.PINECONE_INDEX_NAME}'.")

    @lru_cache(maxsize=1)
    def _load_bm25_encoder(self) -> BM25Encoder:
        """Loads the fitted BM25 encoder from ingestion."""
        if not config.BM25_ENCODER_PATH.exists():
            logger.error(f"BM25 encoder file not found at {config.BM25_ENCODER_PATH}. Did you run ingest.py?")
            raise FileNotFoundError(f"BM25 encoder not found at {config.BM25_ENCODER_PATH}")
        
        with open(config.BM25_ENCODER_PATH, "rb") as f:
            logger.info("Loading BM25 encoder...")
            return pickle.load(f)

    @lru_cache(maxsize=1)
    def _get_index(self):
        """Gets a Pinecone index object."""
        index_name = config.PINECONE_INDEX_NAME
        if index_name not in self.pc.list_indexes().names():
            raise ValueError(f"Index {index_name} does not exist. Please run ingest.py first.")
        return self.pc.Index(index_name)
    
    def _get_namespace(self, model_name: str) -> str:
        """Generates a safe namespace name from the model name."""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)

    def search_and_rerank(self, query: str, model_name: str) -> List[Document]:
        """
        Performs the 2-stage retrieval and reranking.
        """
        if model_name not in self.embed_mgr.models:
            raise ValueError(f"Embedding model {model_name} not supported.")

        namespace = self._get_namespace(model_name)

        # --- Generate Query Vectors ---
        dense_vec = self.embed_mgr.encode(model_name, [query])[0].tolist()
        sparse_vec = self.bm25_encoder.encode_queries(query)
        
        # --- Step 1: Hybrid Search for Top 50 Parent Chunks (2048) ---
        logger.info(f"Step 1: Searching for Top {config.PRE_RERANK_TOP_K} parent chunks (2048)...")
        
        parent_filter = {"chunk_size": config.SEARCH_CHUNK_SIZE}
        
        parent_matches = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            filter=parent_filter,
            top_k=config.PRE_RERANK_TOP_K, # Get 50 candidates
            include_metadata=True,
            namespace=namespace
        )['matches']
        
        if not parent_matches:
            logger.warning("No parent chunks found for query.")
            return []
            
        logger.info(f"Found {len(parent_matches)} parent candidates for reranking.")

        # --- Step 2: Rerank the 50 Parent Chunks ---
        logger.info(f"Step 2: Reranking {len(parent_matches)} parent candidates...")
        
        candidates = []
        for match in parent_matches:
            doc = Document(
                page_content=match['metadata']['text'],
                metadata={
                    "id": match['id'],
                    "source": match['metadata']['source'],
                    "parent_chunk_id": match['metadata']['parent_chunk_id'],
                    "hybrid_score": match['score']
                }
            )
            candidates.append(doc)

        # Rerank and return the final Top 10
        reranked_docs = self.reranker.rerank(query, candidates, top_k=config.FINAL_TOP_K)
        
        logger.info(f"Returning Top {len(reranked_docs)} reranked parent chunks.")
        return reranked_docs
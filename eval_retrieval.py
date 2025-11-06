from __future__ import annotations
import logging, os, re, sys, pickle, json, math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import faiss

try:
    from langchain_core.documents import Document
except Exception as e:
    print("This script requires langchain-core. 'pip install langchain-core'")
    raise

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except Exception:
    _SBERT_OK = False

try:
    import pandas as pd
except Exception as e:
    print("This script requires pandas. 'pip install pandas'")
    raise


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "eval.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rag.eval")


EVAL_QUERIES = [
    {
        "query": "What is bauxite used for?",
        "expected_keywords": ["aluminum", "alumina"] # Graded Relevance: 1 for one, 2 for all
    },
    {
        "query": "Environmental impact of strip mining",
        "expected_keywords": ["habitat", "destruction", "erosion", "pollution", "soil"]
    },
    {
        "query": "What are rare earth elements?",
        "expected_keywords": ["lanthanides", "scandium", "yttrium", "magnets", "electronics"]
    },
    {
        "query": "Process of hydraulic fracturing",
        "expected_keywords": ["fracking", "shale", "gas", "oil", "water", "pressure"]
    },
    {
        "query": "What is geothermal energy?",
        "expected_keywords": ["heat", "earth", "steam", "turbine", "magma"]
    }
]


MODELS_TO_EVAL = ("all-mpnet-base-v2", "BAAI/bge-base-en-v1.5")
GRANULARITIES_TO_EVAL = (128, 512, 2048)
EVAL_K = 10  # Evaluate Hit@10, MRR@10, nDCG@10
SAVE_DIR = Path("faiss_store_nested")


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[Document] = []

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        logger.debug(f"Loading index from {path}")
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "docs.pkl", "rb") as f:
            docs = pickle.load(f)
        obj = cls(index.d)
        obj.index, obj.documents = index, docs
        logger.debug(f"Loaded {len(docs)} documents.")
        return obj

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        k = min(k, self.index.ntotal)
        if k == 0:
            return [], []
        D, I = self.index.search(query_embedding.astype(np.float32), k)
        return [self.documents[i] for i in I[0]], [float(d) for d in D[0]]

class EmbeddingManager:
    def __init__(self, model_names: Iterable[str]):
        if not _SBERT_OK:
            raise RuntimeError("sentence-transformers is required")
        self.models = {m: SentenceTransformer(m) for m in model_names}
        logger.debug(f"EmbeddingManager initialized for {list(model_names)}")

    def encode(self, model: str, texts: List[str]) -> np.ndarray:
        return np.array(self.models[model].encode(texts, batch_size=32, show_progress_bar=False))



def assess_relevance(text: str, keywords: List[str]) -> int:
    """
    Assigns a graded relevance score based on keyword matching.
    - 0: Irrelevant (no keywords)
    - 1: Partially Relevant (at least one keyword)
    - 2: Highly Relevant (all keywords match)
    """
    text_lower = text.lower()
    matches = [kw for kw in keywords if kw in text_lower]
    
    if len(matches) == 0:
        return 0
    if len(matches) == len(keywords):
        return 2
    return 1

def calculate_hit_at_k(scores: List[int], k: int) -> int:
    """Calculates Hit@k (binary: 0 or 1)."""
    return 1 if any(s > 0 for s in scores[:k]) else 0

def calculate_mrr(scores: List[int]) -> float:
    """Calculates Mean Reciprocal Rank."""
    for i, score in enumerate(scores):
        if score > 0:
            return 1.0 / (i + 1)  # (i + 1) is the rank
    return 0.0

def calculate_dcg_at_k(scores: List[int], k: int) -> float:
    """Calculates Discounted Cumulative Gain (DCG)."""
    k = min(k, len(scores))
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores[:k]))

def calculate_ndcg_at_k(scores: List[int], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (nDCG)."""
    dcg = calculate_dcg_at_k(scores, k)
    ideal_scores = sorted(scores, reverse=True)
    idcg = calculate_dcg_at_k(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation():
    logger.info("Starting retrieval evaluation...")
    
    embed_managers = {m: EmbeddingManager([m]) for m in MODELS_TO_EVAL}
    
    all_query_results = []  
    all_hits_details = []  
    
    for model_name in MODELS_TO_EVAL:
        for granularity in GRANULARITIES_TO_EVAL:
            
            config_name = f"{model_name} @ g{granularity}"
            logger.info(f"--- Evaluating: {config_name} ---")
            
            model_path_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name)
            index_path = SAVE_DIR / f"{model_path_name}_{granularity}"
            
            if not index_path.exists():
                logger.warning(f"Index not found, skipping: {index_path}")
                continue
                
            try:
                vs = FaissVectorStore.load(index_path)
                embed_mgr = embed_managers[model_name]
            except Exception as e:
                logger.error(f"Failed to load index {index_path}: {e}")
                continue

            for query_data in EVAL_QUERIES:
                query = query_data["query"]
                keywords = query_data["expected_keywords"]
                
                q_emb = embed_mgr.encode(model_name, [query])[0]
                retrieved_docs, distances = vs.similarity_search(q_emb, k=EVAL_K)
                
                relevance_scores = []
                for i, doc in enumerate(retrieved_docs):
                    score = assess_relevance(doc.page_content, keywords)
                    relevance_scores.append(score)
                    
                    all_hits_details.append({
                        "model": model_name,
                        "granularity": granularity,
                        "query": query,
                        "rank": i + 1,
                        "relevance_score": score,
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "parent_chunk_id": doc.metadata.get("parent_chunk_id"),
                        "source": doc.metadata.get("source"),
                        "text_preview": doc.page_content[:150].replace("\n", " ")
                    })
                
                hit_at_k = calculate_hit_at_k(relevance_scores, EVAL_K)
                mrr = calculate_mrr(relevance_scores)
                ndcg_at_k = calculate_ndcg_at_k(relevance_scores, EVAL_K)
                
                all_query_results.append({
                    "model": model_name,
                    "granularity": granularity,
                    "query": query,
                    f"Hit@{EVAL_K}": hit_at_k,
                    "MRR": mrr,
                    f"nDCG@{EVAL_K}": ndcg_at_k
                })
    
    if not all_query_results:
        logger.error("No results generated. Did you set up your EVAL_QUERIES and run the ingestion script?")
        return

    logger.info("Evaluation complete. Aggregating results and saving CSVs...")
    
    per_query_df = pd.DataFrame(all_query_results)
    per_query_df.to_csv("per_query.csv", index=False)
    
    hits_df = pd.DataFrame(all_hits_details)
    hits_df.to_csv("hits.csv", index=False)
    
    summary_df = per_query_df.groupby(["model", "granularity"]).agg(
        Hit_Rate=(f"Hit@{EVAL_K}", "mean"),
        MRR=("MRR", "mean"),
        nDCG_K=(f"nDCG@{EVAL_K}", "mean")
    ).reset_index()
    
    summary_df.to_csv("summary.csv", index=False)
    
    logger.info("\n--- EVALUATION SUMMARY ---")
    print(summary_df.to_string())
    logger.info("Results saved to summary.csv, per_query.csv, and hits.csv")


if __name__ == "__main__":
    run_evaluation()
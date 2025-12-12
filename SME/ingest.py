from __future__ import annotations
import logging, os, re, sys, pickle, hashlib, uuid, json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set
import numpy as np

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.INGESTION_LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rag.ingest")

@dataclass
class IngestConfig:
    root_dir: Path
    subject: str = "General"
    chunk_sizes: Tuple[int, ...] = config.CHUNK_SIZES
    overlap_ratio: float = config.CHUNK_OVERLAP_RATIO
    max_overlap: int = 220 
    models: Tuple[str, ...] = config.SUPPORTED_EMBEDDING_MODELS
    pinecone_index_name: str = config.PINECONE_INDEX_NAME

LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": UnstructuredMarkdownLoader,
}

_ws_re = re.compile(r"\s+")
def normalize_whitespace(text: str) -> str:
    return _ws_re.sub(" ", text).strip()

def preprocess_text(raw: str) -> str:
    return normalize_whitespace(raw.lower())

def _hash_file(path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Could not hash file {path}: {e}. Returning empty hash.")
        return ""

def _load_manifest() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Loads the ingestion manifest. Returns (file_hashes, doc_id_map)."""
    if not config.INGESTION_MANIFEST_PATH.exists():
        return {}, {}
    try:
        with open(config.INGESTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            return manifest.get("files", {}), manifest.get("doc_ids", {})
    except json.JSONDecodeError:
        logger.warning("Manifest file is corrupted. Starting fresh.")
        return {}, {}

def _save_manifest(file_hashes: Dict[str, str], doc_id_map: Dict[str, str]):
    """Saves the updated manifest."""
    manifest = {"files": file_hashes, "doc_ids": doc_id_map}
    try:
        with open(config.INGESTION_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")

class DocumentProcessor:
    def __init__(self, cfg: IngestConfig, doc_id_map: Dict[str, str]):
        self.cfg = cfg
        self._docid_by_path = doc_id_map
        self.cfg.chunk_sizes = tuple(sorted(cfg.chunk_sizes, reverse=True))

    @staticmethod
    def _stable_id(seed: str, prefix: str) -> str:
        h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{h}"

    def _loader_for(self, path: Path):
        ext = path.suffix.lower()
        if ext in LOADER_MAP: return LOADER_MAP[ext]
        raise ValueError(f"Unsupported file format: {ext}")

    def _augment_metadata(self, doc: Document, file_path: Path, seq_in_file: int):
        file_key = str(file_path.resolve())
        if file_key not in self._docid_by_path:
            self._docid_by_path[file_key] = self._stable_id(file_key, "doc")
        
        doc_id = self._docid_by_path[file_key]
        doc.metadata.update({
            "subject": self.cfg.subject, "source": file_path.name,
            "timestamp": datetime.now().isoformat(), "file_path": file_key,
            "doc_id": doc_id, "doc_seq": seq_in_file,
        })

    def _preprocess_doc(self, doc: Document) -> Document:
        doc.page_content = preprocess_text(doc.page_content)
        return doc

    def get_doc_id_map(self) -> Dict[str, str]:
        """Returns the updated doc_id map after processing."""
        return self._docid_by_path

    def load_specific(self, paths_to_process: List[Path]) -> List[Document]:
        """Loads, processes, and augments *only* the specified list of files."""
        all_docs = []
        for path in paths_to_process:
            try:
                if path.suffix.lower() in LOADER_MAP:
                    loader_class = self._loader_for(path)
                    loader = loader_class(str(path))
                    docs = loader.load()
                    for i, d in enumerate(docs):
                        self._augment_metadata(d, path, i)
                        self._preprocess_doc(d)
                    all_docs.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {path.name}")
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        
        logger.info(f"Loaded a total of {len(all_docs)} documents for ingestion.")
        return all_docs

    def hierarchical_chunks(self, documents: List[Document]) -> Dict[int, List[Document]]:
        """ 
        Splits documents into hierarchical chunks.
        (Original logic, with fixes for empty docs)
        """
        results: Dict[int, List[Document]] = {}
        separators = ["\n\n", "\n", ". ", " ", ""]
        if not self.cfg.chunk_sizes: return {}

        parent_size = self.cfg.chunk_sizes[0]
        overlap = min(int(parent_size * self.cfg.overlap_ratio), self.cfg.max_overlap)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=overlap,
            separators=separators, add_start_index=True,
        )
        
        parent_chunks: List[Document] = []
        for d in documents:
            if not d.page_content or not d.page_content.strip():
                logger.warning(f"Skipping empty document: {d.metadata.get('source')}")
                continue
            
            original_doc_id = d.metadata.get("doc_id", self._stable_id(d.metadata.get("file_path","") or uuid.uuid4().hex, "doc"))
            split = splitter.split_documents([d])
            for i, ch in enumerate(split):
                start = ch.metadata.get("start_index", 0)
                seed = f"{original_doc_id}|{parent_size}|{i}|{start}|{len(ch.page_content)}"
                chunk_id = self._stable_id(seed, "chunk")
                
                ch.metadata.update({
                    "chunk_size": parent_size, "chunk_index": i,
                    "parent_doc_id": original_doc_id, "chunk_id": chunk_id,
                    "parent_chunk_id": "",
                    **{k: v for k, v in d.metadata.items() if k not in ch.metadata},
                })
                parent_chunks.append(ch)

        logger.info(f"Created {len(parent_chunks)} parent chunks (size={parent_size})")
        results[parent_size] = parent_chunks
        previous_level_chunks = parent_chunks 

        for size in self.cfg.chunk_sizes[1:]: 
            overlap = min(int(size * self.cfg.overlap_ratio), self.cfg.max_overlap)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap,
                separators=separators, add_start_index=True,
            )
            current_level_chunks: List[Document] = []
            
            for parent_chunk in previous_level_chunks:
                if not parent_chunk.page_content or not parent_chunk.page_content.strip():
                    continue

                parent_chunk_id = parent_chunk.metadata["chunk_id"]
                original_doc_id = parent_chunk.metadata["parent_doc_id"] 
                parent_doc_for_splitting = Document(page_content=parent_chunk.page_content)
                split = splitter.split_documents([parent_doc_for_splitting]) 
                
                for i, ch in enumerate(split):
                    start = ch.metadata.get("start_index", 0)
                    seed = f"{parent_chunk_id}|{size}|{i}|{start}|{len(ch.page_content)}"
                    chunk_id = self._stable_id(seed, "chunk")
                    new_metadata = parent_chunk.metadata.copy()
                    new_metadata.update({
                        "chunk_size": size, "chunk_index": i,
                        "parent_doc_id": original_doc_id, "chunk_id": chunk_id,
                        "parent_chunk_id": parent_chunk_id, "start_index": start,
                    })
                    child_chunk = Document(page_content=ch.page_content, metadata=new_metadata)
                    current_level_chunks.append(child_chunk)

            logger.info(f"Created {len(current_level_chunks)} child chunks (size={size})")
            results[size] = current_level_chunks
            previous_level_chunks = current_level_chunks
        
        return results

class EmbeddingManager:
    def __init__(self, model_names: Iterable[str]):
        self.models = {m: SentenceTransformer(m) for m in model_names}

    def encode(self, model: str, texts: List[str]) -> np.ndarray:
        return np.array(self.models[model].encode(texts, batch_size=32, show_progress_bar=True))

    def dim(self, model: str) -> int:
        return self.models[model].get_sentence_embedding_dimension()

class RAGIndexer:
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.embed_mgr = EmbeddingManager(cfg.models)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)

    def delete_documents(self, paths_to_delete: List[str], doc_id_map: Dict[str, str]):
        """Deletes all chunks associated with a list of file paths."""
        logger.info(f"Deleting {len(paths_to_delete)} documents from Pinecone...")
        
        index = self.pc.Index(self.cfg.pinecone_index_name)
        
        doc_ids_to_delete = set()
        for path in paths_to_delete:
            if path in doc_id_map:
                doc_ids_to_delete.add(doc_id_map[path])
            else:
                logger.warning(f"No doc_id found for deleted path: {path}. Cannot delete from index.")

        if not doc_ids_to_delete:
            logger.info("No doc_ids to delete.")
            return

        for model_name in self.cfg.models:
            namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            logger.info(f"Deleting from namespace: {namespace}...")
            try:
                for doc_id in doc_ids_to_delete:
                    index.delete(filter={"doc_id": doc_id}, namespace=namespace)
                logger.info(f"Successfully deleted {len(doc_ids_to_delete)} doc_ids from {namespace}.")
            except Exception as e:
                logger.error(f"Error deleting from namespace {namespace}: {e}")

    def build_and_save_bm25_encoder(self, all_documents: List[Document]):
        """Fits and saves a BM25 encoder on the *entire* document corpus."""
        logger.info(f"Fitting BM25 sparse vector encoder on {len(all_documents)} chunks...")
        
        corpus_texts = [doc.page_content for doc in all_documents if doc.page_content.strip()]
        if not corpus_texts:
            raise ValueError("No valid text content found in documents to fit BM25 encoder.")

        bm25 = BM25Encoder()
        bm25.fit(corpus_texts)
        
        config.BM25_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.BM25_ENCODER_PATH, "wb") as f:
            pickle.dump(bm25, f)
        logger.info(f"BM25 encoder saved to {config.BM25_ENCODER_PATH}")
        return bm25

    def get_or_create_pinecone_index(self, dimension: int):
        """Creates a Serverless index if it doesn't exist."""
        index_name = self.cfg.pinecone_index_name
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone Serverless index: {index_name} with dim {dimension}...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION
                )
            )
            logger.info(f"Index {index_name} created.")
        else:
            logger.info(f"Found existing index: {index_name}")
        
        index = self.pc.Index(index_name)
        stats = index.describe_index_stats()
        
        if stats.dimension != dimension:
            logger.error(f"Index dimension mismatch! Index '{index_name}' has dim {stats.dimension} but models have dim {dimension}.")
            sys.exit(1)
        if stats.metric != 'dotproduct':
            logger.error(f"Index metric mismatch! Index '{index_name}' has metric '{stats.metric}' but hybrid search requires 'dotproduct'. Please delete and recreate the index.")
            sys.exit(1)
            
        return index, index_name

    def build_indexes(self, chunks_by_size: Dict[int, List[Document]], bm25_encoder: BM25Encoder):
        """Builds and populates Pinecone indexes for new/updated chunks."""
        
        all_chunks = []
        for size, chunks in chunks_by_size.items():
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No new chunks found to process.")
            return
        
        first_model = self.cfg.models[0]
        dim = self.embed_mgr.dim(first_model)
        index, index_name = self.get_or_create_pinecone_index(dim)
        
        for model_name in self.cfg.models:
            if self.embed_mgr.dim(model_name) != dim:
                logger.error(f"Model {model_name} dim does not match index dim {dim}.")
                continue
                
            namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            
            for size, chunks in chunks_by_size.items():
                if not chunks:
                    logger.info(f"No chunks for size {size}, model {model_name}. Skipping.")
                    continue

                logger.info(f"Processing {len(chunks)} chunks for model {model_name} at size {size}...")

                chunk_texts = [d.page_content for d in chunks]
                logger.info(f"Embedding {len(chunks)} chunks with {model_name}...")
                dense_embeddings = self.embed_mgr.encode(model_name, chunk_texts)
                
                logger.info(f"Generating sparse vectors for {len(chunks)} chunks...")
                sparse_embeddings = bm25_encoder.encode_documents(chunk_texts)
                
                vectors_to_upsert = []
                for doc, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings):
                    if not sparse.get("indices") or not sparse.get("values"):
                        logger.warning(f"Skipping chunk {doc.metadata['chunk_id']} as it produced an empty sparse vector.")
                        continue 

                    pinecone_metadata = {
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "doc_id": doc.metadata.get("doc_id", ""),
                        "chunk_size": doc.metadata.get("chunk_size", 0),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "parent_chunk_id": doc.metadata.get("parent_chunk_id", ""),
                        "parent_doc_id": doc.metadata.get("parent_doc_id", ""),
                        "subject": doc.metadata.get("subject", ""),
                        "file_path": doc.metadata.get("file_path", "")
                    }
                    vec = {
                        "id": doc.metadata["chunk_id"],
                        "values": dense.tolist(),
                        "sparse_values": sparse,
                        "metadata": pinecone_metadata 
                    }
                    vectors_to_upsert.append(vec)
                
                if not vectors_to_upsert:
                    logger.warning(f"No valid vectors to upsert for size {size}, model {model_name}.")
                    continue

                logger.info(f"Upserting {len(vectors_to_upsert)} vectors to {index_name} (namespace: {namespace})...")
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i : i + batch_size]
                    index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"✅ Upsert complete for size {size}, namespace {namespace}")

if __name__ == "__main__":
    if not config.DOCS_PATH.exists():
        logger.error(f"Document directory not found: {config.DOCS_PATH}")
        sys.exit(1)

    cfg = IngestConfig(
        root_dir=config.DOCS_PATH,
        subject="SME_Subject",
    )
    
    logger.info("--- Starting Delta Ingestion Check ---")
    manifest_hashes, doc_id_map = _load_manifest()
    
    current_files: Dict[str, str] = {} 
    for root, _, files in os.walk(cfg.root_dir):
        for fname in files:
            path = Path(root) / fname
            if path.suffix.lower() in LOADER_MAP:
                path_str = str(path.resolve())
                current_files[path_str] = _hash_file(path)

    paths_to_process_str: List[str] = []
    paths_to_delete_str: List[str] = []
    new_manifest_hashes = manifest_hashes.copy()
    
    seen_paths = set()
    
    for path_str, old_hash in manifest_hashes.items():
        seen_paths.add(path_str)
        if path_str not in current_files:
            logger.info(f"DELETION detected: {path_str}")
            paths_to_delete_str.append(path_str)
            new_manifest_hashes.pop(path_str, None)
            doc_id_map.pop(path_str, None) 
        elif current_files[path_str] != old_hash:
            logger.info(f"MODIFICATION detected: {path_str}")
            paths_to_delete_str.append(path_str)
            paths_to_process_str.append(path_str) 
            new_manifest_hashes[path_str] = current_files[path_str]
        else:
            pass 

    for path_str, new_hash in current_files.items():
        if path_str not in seen_paths:
            logger.info(f"NEW file detected: {path_str}")
            paths_to_process_str.append(path_str)
            new_manifest_hashes[path_str] = new_hash
            
    logger.info("--- Ingestion Check Complete ---")
    
    docproc = DocumentProcessor(cfg, doc_id_map)
    indexer = RAGIndexer(cfg)
    
    if paths_to_delete_str:
        logger.info(f"Deleting {len(paths_to_delete_str)} documents from index...")
        indexer.delete_documents(paths_to_delete_str, doc_id_map)
    else:
        logger.info("No documents to delete.")
        
    if paths_to_process_str:
        logger.info(f"Processing {len(paths_to_process_str)} new/modified documents...")
        paths_to_process = [Path(p) for p in paths_to_process_str]
        
        docs_to_ingest = docproc.load_specific(paths_to_process)
        chunks_to_ingest = docproc.hierarchical_chunks(docs_to_ingest)
        
        logger.info("Loading all current documents to refit BM25 encoder...")
        all_current_docs = docproc.load_specific([Path(p) for p in current_files.keys()])
        all_current_chunks_map = docproc.hierarchical_chunks(all_current_docs)
        
        all_chunks_list = []
        for size, chunks in all_current_chunks_map.items():
            all_chunks_list.extend(chunks)
        
        if all_chunks_list:
            logger.info("Refitting BM25 encoder on entire current corpus...")
            bm25_encoder = indexer.build_and_save_bm25_encoder(all_chunks_list)
            
            if chunks_to_ingest:
                logger.info("Building indexes for new/modified chunks...")
                indexer.build_indexes(chunks_to_ingest, bm25_encoder)
                logger.info("✅ Pinecone vector stores updated.")
            else:
                logger.warning("No new chunks were created. Ingestion skipped.")
        else:
            logger.warning("No documents found in corpus. Skipping BM25 fit and ingestion.")
    else:
        logger.info("No new or modified documents to process.")

    final_doc_id_map = docproc.get_doc_id_map()
    _save_manifest(new_manifest_hashes, final_doc_id_map)
    logger.info("✅ Manifest file updated.")
    logger.info("--- Ingestion run finished. ---")
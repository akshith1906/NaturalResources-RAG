
import logging
from typing import Dict, List, Iterable
import numpy as np
from functools import lru_cache

import config
from langchain_core.documents import Document

from google.generativeai.types import HarmCategory, HarmBlockThreshold

try:
    import google.generativeai as genai
    genai.configure(api_key=config.GOOGLE_API_KEY)
except Exception as e:
    logging.error(f"Failed to configure Google Generative AI: {e}")
    raise

@lru_cache(maxsize=1)
def get_llm():
    """Initializes and returns the Gemini model client."""
    logging.info(f"Loading Generative Model: {config.GENERATIVE_MODEL_NAME}")
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    return genai.GenerativeModel(
        config.GENERATIVE_MODEL_NAME,
        safety_settings=safety_settings
    )

def llm_invoke(prompt: str) -> str:
    """Invokes the Gemini model with robust multi-part handling."""
    model = get_llm()
    try:
        response = model.generate_content(prompt)
        
        if not response.parts:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                logging.error(f"LLM Blocked. Reason: {reason}")
                return f"Error: The AI model blocked the response. Reason: {reason}"
            return "Error: The AI model returned an empty response (blocked)."

        try:
            return response.text
        except ValueError:
            parts_text = []
            for part in response.parts:
                if part.text:
                    parts_text.append(part.text)
            
            if parts_text:
                return "".join(parts_text)
            else:
                return "Error: Model returned content but no text parts found."

    except Exception as e:
        logging.error(f"Error invoking Gemini model: {e}")
        return f"Error: Could not get response from model. {e}"

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except Exception:
    _SBERT_OK = False

class EmbeddingManager:
    def __init__(self, model_names: Iterable[str]):
        if not _SBERT_OK:
            raise RuntimeError("sentence-transformers is required")
        self.models = {m: self._load_model(m) for m in model_names}

    @lru_cache(maxsize=4)
    def _load_model(self, model_name: str) -> SentenceTransformer:
        logging.info(f"Loading embedding model: {model_name}")
        return SentenceTransformer(model_name)

    def encode(self, model: str, texts: List[str]) -> np.ndarray:
        return np.array(self.models[model].encode(texts, batch_size=32, show_progress_bar=False))

    def dim(self, model: str) -> int:
        return self.models[model].get_sentence_embedding_dimension()

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_OK = True
except Exception:
    _CROSS_ENCODER_OK = False

class Reranker:
    def __init__(self, model_name: str, batch_size: int = 8):
        self.model_name = model_name
        if not _CROSS_ENCODER_OK:
            logging.error("CrossEncoder not found. Reranking will be disabled.")
            self.model = None
            return
        
        try:
            self.model = self._load_model(model_name)
            self.batch_size = batch_size
            logging.info(f"Loaded Reranker model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load Reranker model {model_name}. Reranking disabled. Error: {e}")
            self.model = None

    @lru_cache(maxsize=1)
    def _load_model(self, model_name: str) -> CrossEncoder:
        return CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not self.model or not documents:
            return documents[:top_k]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        final_docs = []
        for score, doc in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            final_docs.append(doc)
            
        return final_docs
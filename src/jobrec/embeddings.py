import numpy as np, joblib, faiss, tqdm
from sentence_transformers import SentenceTransformer
from src import config

_sbert = SentenceTransformer(config.SBERT_MODEL_NAME)

def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _sbert.encode(texts, batch_size=batch_size, show_progress_bar=True)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

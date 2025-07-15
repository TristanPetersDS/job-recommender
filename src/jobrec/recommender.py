import pandas as pd, numpy as np, faiss
from sklearn.metrics.pairwise import cosine_similarity
from src import config, preprocessing, embeddings

def recommend_jobs(resume_text: str,
                   jobs_df: pd.DataFrame,
                   job_embeddings: np.ndarray,
                   index: faiss.IndexFlatIP | None = None,
                   top_n: int = config.TOP_N_RECOMMEND) -> pd.DataFrame:
    # preprocess & embed resume
    resume_clean = preprocessing.preprocess_text(resume_text)
    resume_vec = embeddings.encode_texts([resume_clean])
    # cosine sim
    if index:
        faiss.normalize_L2(resume_vec)
        sim_scores, idxs = index.search(resume_vec, top_n)
        result = jobs_df.iloc[idxs[0]].copy()
        result["sim_score"] = sim_scores[0]
    else:
        scores = cosine_similarity(resume_vec, job_embeddings)[0]
        result = jobs_df.copy()
        result["sim_score"] = scores
        result = result.nlargest(top_n, "sim_score")
    return result.reset_index(drop=True)

# Standard library imports
import time
from abc import ABC, abstractmethod
from typing import Dict, List

# Third-party library imports
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.inference_times = []
    
    @abstractmethod
    def fit(self, jobs_df: pd.DataFrame):
        """Fit the model on job descriptions."""
        pass
    
    @abstractmethod
    def recommend(self, resume_text: str, top_n: int = 5) -> List[Dict]:
        """Generate top N job recommendations for a resume."""
        pass
    
    def get_average_inference_time(self) -> float:
        """Return average inference time."""
        return np.mean(self.inference_times) if self.inference_times else 0.0
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.inference_times = []


class BaselineTFIDFRecommender(BaseRecommender):
    """Baseline TF-IDF model using unfiltered dataset."""
    
    def __init__(self):
        super().__init__("Baseline TF-IDF")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.job_vectors = None
        self.jobs_df = None
    
    def fit(self, jobs_df: pd.DataFrame):
        """Fit TF-IDF on all job descriptions."""
        self.jobs_df = jobs_df.copy()
        self.job_vectors = self.vectorizer.fit_transform(jobs_df['clean_text'].values)
        self.is_fitted = True
        print(f"{self.name} fitted on {len(jobs_df)} jobs")
    
    def recommend(self, resume_text: str, top_n: int = 5) -> List[Dict]:
        """Generate recommendations using cosine similarity."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        start_time = time.time()
        
        # Vectorize resume
        resume_vector = self.vectorizer.transform([resume_text])
        
        # Calculate similarities
        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
        
        # Get top N indices
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Prepare results
        recommendations = []
        for idx in top_indices:
            job_row = self.jobs_df.iloc[idx]
            recommendations.append({
                'job_id': job_row['job_id'],
                'clean_text': job_row['clean_text'],
                'text': job_row.get('text', job_row['clean_text']),  # Include original text for SBERT
                'similarity_score': similarities[idx],
                'domains': job_row['domains']
            })
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return recommendations

class RefinedTFIDFRecommender(BaseRecommender):
    """Refined TF-IDF using filtered and balanced dataset."""
    
    def __init__(self):
        super().__init__("Refined TF-IDF")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,  # Additional filtering
            max_df=0.95
        )
        self.job_vectors = None
        self.jobs_df = None
    
    def fit(self, jobs_df: pd.DataFrame):
        """Fit TF-IDF on filtered job descriptions."""
        self.jobs_df = jobs_df.copy()
        self.job_vectors = self.vectorizer.fit_transform(jobs_df['clean_text'].values)
        self.is_fitted = True
        print(f"{self.name} fitted on {len(jobs_df)} filtered jobs")
    
    def recommend(self, resume_text: str, top_n: int = 5) -> List[Dict]:
        """Generate recommendations with same logic as baseline."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        start_time = time.time()
        
        resume_vector = self.vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            job_row = self.jobs_df.iloc[idx]
            recommendations.append({
                'job_id': job_row['job_id'],
                'clean_text': job_row['clean_text'],
                'text': job_row.get('text', job_row['clean_text']),  # Include original text for SBERT
                'similarity_score': similarities[idx],
                'domains': job_row['domains']
            })
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return recommendations

class SBERTRecommender(BaseRecommender):
    """SBERT model with TF-IDF candidate generation."""
    
    def __init__(self, candidate_model: BaseRecommender = None, n_candidates: int = 50):
        super().__init__("SBERT + Candidate Generation")
        self.candidate_model = candidate_model or RefinedTFIDFRecommender()
        self.n_candidates = n_candidates
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.jobs_df = None
    
    def fit(self, jobs_df: pd.DataFrame):
        """Fit the candidate generation model."""
        self.jobs_df = jobs_df.copy()
        self.candidate_model.fit(jobs_df)
        self.is_fitted = True
        print(f"{self.name} ready with {len(jobs_df)} jobs in pool")
    
    def recommend(self, resume_text: str, resume_domains: List[str] = None, top_n: int = 5) -> List[Dict]:
        """Two-stage recommendation: candidate generation + semantic reranking."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")

        start_time = time.time()

        # Stage 1: Generate more candidates for better SBERT reranking
        # Increase candidate pool size for better semantic matching
        candidate_pool_size = max(self.n_candidates, top_n * 15)  # At least 15x the requested results
        candidates = self.candidate_model.recommend(resume_text, candidate_pool_size)

        # Softer domain filtering - boost domain matches rather than hard filter
        domain_boosted_candidates = []
        if resume_domains:
            for candidate in candidates:
                domains = candidate.get('domains', [])
                # Add domain boost score instead of hard filtering
                domain_overlap = 0
                if isinstance(domains, list) and isinstance(resume_domains, list):
                    domain_overlap = len(set(domains) & set(resume_domains)) / len(set(domains) | set(resume_domains))

                candidate['domain_boost'] = domain_overlap
                domain_boosted_candidates.append(candidate)
            candidates = domain_boosted_candidates

        # Stage 2: Semantic reranking with SBERT using original text
        # SBERT works better with natural language, not heavily preprocessed text
        # Use original text if available, otherwise fall back to clean_text
        resume_text_for_sbert = resume_text  # This should be the original resume text

        # Get original job text for SBERT (prefer 'text' over 'clean_text')
        candidate_texts = []
        for c in candidates:
            # Use original text if available, fall back to clean_text
            job_text = c.get('text', c.get('clean_text', ''))
            candidate_texts.append(job_text)

        # Encode with SBERT
        resume_embedding = self.sbert_model.encode([resume_text_for_sbert], show_progress_bar=False)
        candidate_embeddings = self.sbert_model.encode(candidate_texts, show_progress_bar=False)

        # Calculate semantic similarities
        semantic_similarities = cosine_similarity(resume_embedding, candidate_embeddings).flatten()

        # Combine semantic similarity with domain boost if available
        final_scores = semantic_similarities.copy()
        if resume_domains:
            for i, candidate in enumerate(candidates):
                domain_boost = candidate.get('domain_boost', 0)
                # Weight: 80% semantic, 20% domain relevance
                final_scores[i] = 0.8 * semantic_similarities[i] + 0.2 * domain_boost

        # Rerank and select top N
        reranked_indices = np.argsort(final_scores)[::-1][:top_n]

        recommendations = []
        for idx in reranked_indices:
            candidate = candidates[idx]
            candidate['semantic_score'] = semantic_similarities[idx]
            candidate['final_score'] = final_scores[idx]
            # Clean up temporary fields
            if 'domain_boost' in candidate:
                del candidate['domain_boost']
            recommendations.append(candidate)

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return recommendations
        
class ModelFactory:
    """Factory class for easy model selection and initialization."""
    
    def __init__(self):
        self.models = {
            'baseline': BaselineTFIDFRecommender,
            'refined': RefinedTFIDFRecommender,
            'sbert': SBERTRecommender
        }
        self.fitted_models = {}
    
    def get_model(self, model_name: str) -> BaseRecommender:
        """Get an instance of the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name]()
    
    def fit_all_models(self, baseline_jobs_df: pd.DataFrame, refined_jobs_df: pd.DataFrame):
        """Fit all models with appropriate datasets."""
        # Baseline uses unfiltered data
        baseline_model = self.get_model('baseline')
        baseline_model.fit(baseline_jobs_df)
        self.fitted_models['baseline'] = baseline_model
        
        # Refined and SBERT use filtered data
        refined_model = self.get_model('refined')
        refined_model.fit(refined_jobs_df)
        self.fitted_models['refined'] = refined_model
        
        sbert_model = self.get_model('sbert')
        sbert_model.fit(refined_jobs_df)
        self.fitted_models['sbert'] = sbert_model
        
        print(f"\nAll {len(self.fitted_models)} models fitted successfully")
        
        return self.fitted_models
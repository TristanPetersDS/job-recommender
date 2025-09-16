"""
Job Recommendation System Evaluation Module
============================================
Comprehensive evaluation framework supporting both local and cloud LLM providers.
COMPLETELY FIXED VERSION - Addresses all issues with JSON parsing, progress bars, connection handling, and metrics.
"""

import hashlib
import json
import time
import logging
import pickle
import os
import contextlib
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress HTTP request logging that interferes with progress bars
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class ModelProvider(Enum):
    """Enumeration of supported model providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for different model providers."""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 150
    supports_json_mode: bool = True
    request_timeout: int = 30
    
    # Local model specific settings
    context_length: int = 4096
    use_structured_output: bool = False
    
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs):
        """Create configuration from common presets."""
        presets = {
            'lmstudio': cls(
                provider=ModelProvider.LMSTUDIO,
                model_name='local-model',
                base_url='http://localhost:1234/v1',
                supports_json_mode=False,
                context_length=4096,
                **kwargs
            ),
            'ollama': cls(
                provider=ModelProvider.OLLAMA,
                model_name='llama3',
                base_url='http://localhost:11434/api',
                supports_json_mode=False,
                context_length=4096,
                **kwargs
            ),
            'openai': cls(
                provider=ModelProvider.OPENAI,
                model_name='gpt-4-turbo-preview',
                supports_json_mode=True,
                context_length=128000,
                **kwargs
            ),
            'gemini': cls(
                provider=ModelProvider.GEMINI,
                model_name='gemini-pro',
                supports_json_mode=True,
                context_length=30720,
                **kwargs
            ),
            'anthropic': cls(
                provider=ModelProvider.ANTHROPIC,
                model_name='claude-3-sonnet-20240229',
                supports_json_mode=False,
                context_length=200000,
                **kwargs
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return presets[preset_name]


class LLMEvaluator:
    """Enhanced LLM-based evaluation supporting local and cloud models."""
    
    def __init__(
        self,
        config: Union[ModelConfig, str],
        api_key: Optional[str] = None,
        cache_dir: str = './llm_cache',
        enable_cache: bool = False,
        batch_size: int = 5,
        max_workers: int = 1,  # Sequential processing
        connection_check_interval: int = 5  # Check connection every N evaluations
    ):
        """
        Initialize the evaluator with flexible model configuration.
        
        Args:
            config: ModelConfig object or preset name
            api_key: API key for cloud providers
            cache_dir: Directory for caching evaluations
            enable_cache: Whether to use caching
            batch_size: Number of evaluations to batch together
            max_workers: Number of parallel workers (1 for sequential)
            connection_check_interval: How often to check connection health
        """
        # Handle configuration
        if isinstance(config, str):
            self.config = ModelConfig.from_preset(config, api_key=api_key)
        else:
            self.config = config
            if api_key:
                self.config.api_key = api_key
        
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.connection_check_interval = connection_check_interval
        self.evaluation_cache = {}
        self.stats = defaultdict(int)
        self.consecutive_connection_failures = 0
        self.max_connection_failures = 3
        
        # Setup cache with proper directory structure
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
        
        # Initialize model client based on provider
        self._init_client()
        
        # Set up improved prompts
        self._setup_prompts()
    
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        provider = self.config.provider
        
        try:
            if provider == ModelProvider.OPENAI:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.config.api_key)
            
            elif provider == ModelProvider.LMSTUDIO:
                from openai import OpenAI
                self.client = OpenAI(api_key="lm-studio", base_url=self.config.base_url)
                self._check_local_connection()

            elif provider == ModelProvider.OLLAMA:
                self.client = None  # Using requests directly
                self._check_local_connection()

            elif provider == ModelProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                self.client = genai.GenerativeModel(self.config.model_name)

            elif provider == ModelProvider.ANTHROPIC:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.config.api_key)

            elif provider in [ModelProvider.LOCAL, ModelProvider.CUSTOM]:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.config.api_key or "dummy", base_url=self.config.base_url)
                self._check_local_connection()

        except ImportError as e:
            logger.error(f"Missing dependency for {provider.value}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize client for {provider.value}: {e}")
            raise

    def _check_local_connection(self, raise_on_failure: bool = True):
        """Check connection to a local model endpoint with improved error handling."""
        if not self.config.base_url:
            return True
        
        try:
            # For Ollama, the root endpoint is a simple health check
            if self.config.provider == ModelProvider.OLLAMA:
                response = requests.get(self.config.base_url.replace('/api', '/'), timeout=5)
            else:
                # Try health endpoint first
                health_url = self.config.base_url.replace('/v1', '/health')
                try:
                    response = requests.get(health_url, timeout=5)
                except:
                    # Fallback to models endpoint
                    response = requests.get(f"{self.config.base_url}/models", timeout=5)

            if response.status_code == 200:
                # Reset failure count on success
                if self.consecutive_connection_failures > 0:
                    logger.info("Successfully reconnected to local model.")
                self.consecutive_connection_failures = 0
                return True
            else:
                raise requests.exceptions.RequestException(f"Status code {response.status_code}")

        except requests.exceptions.RequestException as e:
            self.consecutive_connection_failures += 1
            logger.warning(f"Connection check {self.consecutive_connection_failures}/{self.max_connection_failures} failed: {e}")
            
            if self.consecutive_connection_failures >= self.max_connection_failures:
                error_msg = f"Maximum connection failures ({self.max_connection_failures}) reached. Halting evaluation."
                logger.error(error_msg)
                if raise_on_failure:
                    raise ConnectionError(error_msg)
                return False
            return False
    
    def _setup_prompts(self):
        """Set up evaluation prompts with clearer instructions for the 0-5 scoring system."""
        
        self.evaluation_prompt = """You are an expert technical recruiter evaluating resume-job matches. 

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY valid JSON in the exact format specified
2. Your score MUST be a single integer from 0 to 5 (no decimals, no ranges)
3. Do NOT include markdown code blocks or extra text

SCORING SCALE (0-5):
• 5 (ROCKSTAR): Perfect candidate exceeding all qualifications. Extremely rare.
• 4 (SOLID PASS): Strong candidate a hiring manager would definitely want to interview.
• 3 (PASS): Barely qualified candidate a hiring manager would consider interviewing.
• 2 (INSUFFICIENT): Relevant background but doesn't meet role requirements.
• 1 (WEAK): Very little correlation between resume and job.
• 0 (IRRELEVANT): Completely irrelevant match.

EVALUATION CRITERIA:
1. Skills alignment with job requirements
2. Experience level appropriateness 
3. Industry/domain relevance

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Respond with ONLY this exact JSON format:
{{"score": X, "justification": "One sentence explaining the score"}}

Replace X with your integer score (0-5) and provide a brief justification."""
    
    def _extract_score_from_text(self, text: str) -> Dict:
        """Enhanced fallback method to extract scores from malformed responses."""
        
        # Clean the text
        text = text.strip()
        
        # Try to find score patterns
        score_patterns = [
            r'"score"\s*:\s*(\d+)',
            r'score["\']?\s*:\s*(\d+)',
            r'Score:\s*(\d+)',
            r'rating["\']?\s*:\s*(\d+)'
        ]
        
        score = None
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_score = int(match.group(1))
                if 0 <= potential_score <= 5:
                    score = potential_score
                    break
        
        # If no valid score found, try to infer from keywords
        if score is None:
            text_lower = text.lower()
            if any(word in text_lower for word in ['perfect', 'excellent', 'rockstar', 'outstanding']):
                score = 5
            elif any(word in text_lower for word in ['strong', 'solid', 'good', 'qualified']):
                score = 4
            elif any(word in text_lower for word in ['adequate', 'suitable', 'meets', 'basic']):
                score = 3
            elif any(word in text_lower for word in ['insufficient', 'lacking', 'weak']):
                score = 2
            elif any(word in text_lower for word in ['poor', 'irrelevant', 'unrelated']):
                score = 1
            else:
                score = 0
        
        return {
            "score": score if score is not None else 0,
            "justification": "Extracted from malformed response"
        }
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Robust JSON parsing with multiple fallback strategies."""
        
        if not response_text or not response_text.strip():
            return {"score": 0, "justification": "Empty response"}
        
        # Clean the response
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            # Remove first and last lines if they contain ```
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith('```'):
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()
        
        # Try to extract JSON from the text
        json_patterns = [
            r'\{[^{}]*"score"[^{}]*\}',  # Simple JSON pattern
            r'\{.*?"score".*?\}',       # More flexible pattern
        ]
        
        json_text = None
        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                json_text = match.group(0)
                break
        
        if not json_text:
            # Look for the entire text as potential JSON
            if '{' in response_text and '}' in response_text:
                start = response_text.index('{')
                end = response_text.rindex('}') + 1
                json_text = response_text[start:end]
        
        # Try parsing the extracted JSON
        if json_text:
            try:
                result = json.loads(json_text)
                
                # Validate and clean the result
                if 'score' in result:
                    # Ensure score is integer and in valid range
                    score = result['score']
                    if isinstance(score, str):
                        score = int(float(score))  # Handle string numbers
                    else:
                        score = int(score)
                    
                    # Clamp score to valid range
                    score = max(0, min(5, score))
                    result['score'] = score
                    
                    # Ensure justification exists
                    if 'justification' not in result or not result['justification']:
                        result['justification'] = "No justification provided"
                    
                    return result
                    
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"JSON parsing failed: {e}")
        
        # Final fallback: extract score from text
        return self._extract_score_from_text(response_text)
    
    def _call_local_model(self, prompt: str) -> Dict:
        """Call local model with improved error handling and JSON parsing."""
        
        if self.config.provider == ModelProvider.OLLAMA:
            # Ollama-specific API call
            response = requests.post(
                f"{self.config.base_url}/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "stream": False
                },
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                result_text = response.json().get('response', '')
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        
        elif self.config.provider in [ModelProvider.LMSTUDIO, ModelProvider.LOCAL]:
            # OpenAI-compatible local API
            messages = [
                {"role": "system", "content": "You are an expert recruiter. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            # Build request kwargs
            request_kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # Only add response_format if supported
            if self.config.supports_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**request_kwargs)
            result_text = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported local provider: {self.config.provider}")
        
        # Use improved JSON parsing
        return self._parse_llm_response(result_text)
    
    def _call_llm(self, prompt: str, retry_count: int = 5) -> Dict:
        """Make a call to the LLM with immediate retry logic for malformed responses."""
        
        # Check connection before making request (for local models)
        if self.config.provider in [ModelProvider.LMSTUDIO, ModelProvider.OLLAMA, ModelProvider.LOCAL]:
            if not self._check_local_connection(raise_on_failure=False):
                return {"score": 0, "justification": "Connection failed"}
        
        # Truncate prompt for local models if needed
        if self.config.provider in [ModelProvider.LMSTUDIO, ModelProvider.OLLAMA, ModelProvider.LOCAL]:
            max_prompt_chars = (self.config.context_length * 3)
            if len(prompt) > max_prompt_chars:
                prompt = prompt[:max_prompt_chars] + "..."
        
        for attempt in range(retry_count):
            try:
                # Route to appropriate handler
                if self.config.provider in [ModelProvider.LMSTUDIO, ModelProvider.OLLAMA, ModelProvider.LOCAL]:
                    result = self._call_local_model(prompt)
                    
                elif self.config.provider == ModelProvider.OPENAI:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert recruiter. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"} if self.config.supports_json_mode else None
                    )
                    result = self._parse_llm_response(response.choices[0].message.content)
                    
                elif self.config.provider == ModelProvider.GEMINI:
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.config.temperature,
                            "response_mime_type": "application/json" if self.config.supports_json_mode else "text/plain"
                        }
                    )
                    result = self._parse_llm_response(response.text)
                    
                elif self.config.provider == ModelProvider.ANTHROPIC:
                    response = self.client.messages.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = self._parse_llm_response(response.content[0].text)
                    
                else:
                    raise ValueError(f"Unsupported provider: {self.config.provider}")
                
                # Validate result - if valid, return immediately
                if 'score' in result and isinstance(result['score'], int) and 0 <= result['score'] <= 5:
                    self.stats['api_calls'] += 1
                    return result
                else:
                    # Invalid result - continue to next attempt
                    if attempt < retry_count - 1:
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{retry_count} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(0.5)
                    continue
        
        # All retries failed
        self.stats['failed_calls'] += 1
        return {"score": 0, "justification": "Failed after multiple retry attempts"}
    
    def _get_cache_key(self, resume_text: str, job_description: str) -> str:
        """Generate a unique cache key for a resume-job pair."""
        # Include more context for better cache accuracy
        prompt_hash = hashlib.md5(self.evaluation_prompt.encode()).hexdigest()[:8]
        config_hash = hashlib.md5(f"{self.config.temperature}_{self.config.max_tokens}".encode()).hexdigest()[:8]
        
        # Use full text hash instead of truncated text to avoid collisions
        full_text = f"{resume_text}|{job_description}"
        text_hash = hashlib.md5(full_text.encode()).hexdigest()
        
        combined = f"{text_hash}|{self.config.model_name}|{prompt_hash}|{config_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached evaluations from disk with safe filename handling."""
        # Create safe filename by replacing problematic characters
        safe_model_name = re.sub(r'[^\w\-_]', '_', self.config.model_name)
        cache_file = self.cache_dir / f"{self.config.provider.value}_{safe_model_name}_cache.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.evaluation_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.evaluation_cache)} cached evaluations")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.evaluation_cache = {}
    
    def _save_cache(self):
        """Save evaluation cache to disk with safe filename handling."""
        if not self.enable_cache:
            return
        
        # Create safe filename by replacing problematic characters
        safe_model_name = re.sub(r'[^\w\-_]', '_', self.config.model_name)
        cache_file = self.cache_dir / f"{self.config.provider.value}_{safe_model_name}_cache.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.evaluation_cache, f)
            logger.debug(f"Saved {len(self.evaluation_cache)} evaluations to cache")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def evaluate_match(
        self, 
        resume_text: str, 
        job_description: str,
        use_cache: bool = True
    ) -> Dict:
        """Evaluate a single resume-job match."""
        
        # Check cache first
        if self.enable_cache and use_cache:
            cache_key = self._get_cache_key(resume_text, job_description)
            if cache_key in self.evaluation_cache:
                self.stats['cache_hits'] += 1
                return self.evaluation_cache[cache_key]
        
        # Truncate texts appropriately
        if self.config.provider in [ModelProvider.LMSTUDIO, ModelProvider.OLLAMA, ModelProvider.LOCAL]:
            resume_text = resume_text[:1500] if len(resume_text) > 1500 else resume_text
            job_description = job_description[:1500] if len(job_description) > 1500 else job_description
        else:
            resume_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
            job_description = job_description[:2000] if len(job_description) > 2000 else job_description
        
        # Prepare prompt
        prompt = self.evaluation_prompt.format(
            resume_text=resume_text,
            job_description=job_description
        )
        
        # Get evaluation (with built-in retries)
        result = self._call_llm(prompt)
        
        # Cache only successful results (score >= 0)
        if self.enable_cache and use_cache and result['score'] >= 0:
            cache_key = self._get_cache_key(resume_text, job_description)
            self.evaluation_cache[cache_key] = result
            self.stats['cache_misses'] += 1
        
        return result
    
    def evaluate_batch(
        self,
        resume_job_pairs: List[Tuple[str, str]],
        show_progress: bool = True
    ) -> List[Dict]:
        """Evaluate multiple resume-job pairs sequentially with robust connection checking."""
        results = []
        
        # Use tqdm with proper settings for better progress display
        pbar = tqdm(
            total=len(resume_job_pairs), 
            desc="Evaluating recommendations", 
            unit="eval",
            leave=True,
            dynamic_ncols=True
        ) if show_progress else None
                
        for i, (resume_text, job_description) in enumerate(resume_job_pairs):
            try:
                # Periodic connection check
                if i % self.connection_check_interval == 0 and i > 0:
                    if self.config.provider in [ModelProvider.LMSTUDIO, ModelProvider.OLLAMA, ModelProvider.LOCAL]:
                        if not self._check_local_connection(raise_on_failure=False):
                            logger.error("Connection check failed. Halting evaluation.")
                            remaining_count = len(resume_job_pairs) - len(results)
                            for _ in range(remaining_count):
                                results.append({"score": 0, "justification": "Evaluation halted due to connection error"})
                            break
                
                # Evaluate the pair (retries are handled internally now)
                result = self.evaluate_match(resume_text, job_description)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Unexpected error during evaluation: {e}")
                results.append({"score": 0, "justification": f"Error: {str(e)}"})
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        self._save_cache()
        return results
    
    def get_statistics(self) -> Dict:
        """Get evaluation statistics."""
        total_calls = self.stats['api_calls'] + self.stats['cache_hits']
        cache_rate = self.stats['cache_hits'] / total_calls if total_calls > 0 else 0
        
        return {
            'provider': self.config.provider.value,
            'model': self.config.model_name,
            'total_evaluations': total_calls,
            'api_calls': self.stats['api_calls'],
            'cache_hits': self.stats['cache_hits'],
            'cache_rate': cache_rate,
            'failed_calls': self.stats['failed_calls'],
            'cache_size': len(self.evaluation_cache)
        }
    
    def clear_cache(self):
        """Clear the evaluation cache."""
        self.evaluation_cache = {}
        self._save_cache()
        logger.info("Cache cleared")


class EvaluationCostEstimator:
    """Cost estimation for cloud providers only."""
    
    COST_PER_1K_TOKENS = {
        'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'gemini-pro': {'input': 0.00025, 'output': 0.0005},
        'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
        'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
        # Local models have no cost
        'local': {'input': 0.0, 'output': 0.0},
        'llama3': {'input': 0.0, 'output': 0.0},
        'lmstudio': {'input': 0.0, 'output': 0.0}
    }
    
    def __init__(self, model_name: str = 'gpt-4-turbo-preview'):
        self.model_name = model_name
        self.costs = self.COST_PER_1K_TOKENS.get(
            model_name, 
            {'input': 0.01, 'output': 0.03}  # Default to GPT-4 pricing
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def estimate_single_evaluation_cost(
        self, 
        resume_length: int = 500, 
        job_length: int = 300
    ) -> float:
        """Estimate cost for a single evaluation."""
        # Estimate input tokens (prompt + resume + job)
        prompt_tokens = 300  # Approximate prompt template size
        input_tokens = prompt_tokens + self.estimate_tokens(
            "x" * (resume_length + job_length)
        )
        
        # Estimate output tokens (score + justification)
        output_tokens = 50  # JSON with score and one-sentence justification
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * self.costs['input']
        output_cost = (output_tokens / 1000) * self.costs['output']
        
        return input_cost + output_cost
    
    def estimate_total_cost(
        self,
        n_resumes: int,
        n_models: int,
        n_recommendations: int,
        cache_hit_rate: float = 0.0
    ) -> Dict[str, float]:
        """Estimate total evaluation cost."""
        total_evaluations = n_resumes * n_models * n_recommendations
        unique_evaluations = total_evaluations * (1 - cache_hit_rate)
        
        single_cost = self.estimate_single_evaluation_cost()
        total_cost = unique_evaluations * single_cost
        
        return {
            'total_evaluations': total_evaluations,
            'unique_evaluations': unique_evaluations,
            'cost_per_evaluation': single_cost,
            'total_cost': total_cost,
            'cost_with_20_cache': total_cost * 0.8,
            'cost_with_50_cache': total_cost * 0.5
        }
    
    def print_cost_breakdown(
        self,
        n_resumes: int,
        n_models: int = 3,
        n_recommendations: int = 5
    ):
        """Print detailed cost breakdown."""
        estimates = self.estimate_total_cost(
            n_resumes, n_models, n_recommendations
        )
        
        print(f"{'='*50}")
        print(f"COST ESTIMATION FOR {self.model_name}")
        print(f"{'='*50}")
        print(f"Configuration:")
        print(f"  - Resumes: {n_resumes}")
        print(f"  - Models: {n_models}")
        print(f"  - Recommendations per resume: {n_recommendations}")
        print(f"\nEvaluations:")
        print(f"  - Total evaluations: {estimates['total_evaluations']:,}")
        print(f"  - Cost per evaluation: ${estimates['cost_per_evaluation']:.4f}")
        print(f"\nEstimated Costs:")
        print(f"  - No cache: ${estimates['total_cost']:.2f}")
        print(f"  - 20% cache hit rate: ${estimates['cost_with_20_cache']:.2f}")
        print(f"  - 50% cache hit rate: ${estimates['cost_with_50_cache']:.2f}")
        print(f"{'='*50}")


def evaluate_all_models(
    models: Dict,
    resume_df: pd.DataFrame,
    evaluator: LLMEvaluator,
    n_recommendations: int = 5,
    use_batch: bool = True,
    save_intermediate: bool = True
) -> pd.DataFrame:
    """
    Enhanced evaluation of all models with batch processing and intermediate saves.
    """
    results = []
    all_pairs = []  # For batch evaluation
    pair_metadata = []  # To track what each pair represents
    
    total_evaluations = len(resume_df) * len(models) * n_recommendations
    print(f"Starting evaluation of {total_evaluations} recommendations...")
    print(f"Models: {list(models.keys())}")
    print(f"Resumes: {len(resume_df)}")
    print(f"Recommendations per resume: {n_recommendations}")
    
    # Collect all recommendation pairs first
    print("\nPhase 1: Generating recommendations...")
    with tqdm(total=len(resume_df) * len(models), desc="Getting recommendations", leave=True) as pbar:
        for resume_idx, resume_row in resume_df.iterrows():
            # Handle different column naming conventions
            resume_text = resume_row.get('text', resume_row.get('clean_text', ''))
            resume_id = resume_row.get('index', resume_row.get('resume_id', resume_idx))
            resume_category = resume_row.get('category', 'Unknown')
            resume_domains = resume_row.get('domain_bridge', [])
            
            for model_name, model in models.items():
                # Get recommendations with timing
                start_time = time.time()
                
                try:
                    # Suppress progress bars from sentence-transformers
                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        if model_name == 'sbert' and resume_domains:
                            recommendations = model.recommend(
                                resume_text, 
                                resume_domains=resume_domains,
                                top_n=n_recommendations
                            )
                        else:
                            recommendations = model.recommend(resume_text, top_n=n_recommendations)
                    
                    inference_time = time.time() - start_time
                    
                    # Collect pairs for batch evaluation
                    for rank, rec in enumerate(recommendations, 1):
                        # Handle different field names for job text
                        job_text = rec.get('text', rec.get('clean_text', ''))
                        
                        if use_batch:
                            all_pairs.append((resume_text, job_text))
                            pair_metadata.append({
                                'resume_id': resume_id,
                                'resume_category': resume_category,
                                'model_name': model_name,
                                'job_id': rec.get('job_id', f"job_{rank}"),
                                'rank': rank,
                                'similarity_score': rec.get('similarity_score', rec.get('semantic_score', 0)),
                                'inference_time': inference_time
                            })
                        else:
                            # Direct evaluation (non-batch mode)
                            eval_result = evaluator.evaluate_match(resume_text, job_text)
                            results.append({
                                'resume_id': resume_id,
                                'resume_category': resume_category,
                                'model_name': model_name,
                                'job_id': rec.get('job_id', f"job_{rank}"),
                                'rank': rank,
                                'similarity_score': rec.get('similarity_score', rec.get('semantic_score', 0)),
                                'relevance_score': eval_result['score'],
                                'justification': eval_result['justification'],
                                'inference_time': inference_time
                            })
                
                except Exception as e:
                    logger.error(f"Failed to get recommendations from {model_name} for resume {resume_id}: {e}")
                    # Add dummy results for failed recommendations
                    for rank in range(1, n_recommendations + 1):
                        results.append({
                            'resume_id': resume_id,
                            'resume_category': resume_category,
                            'model_name': model_name,
                            'job_id': f"failed_{rank}",
                            'rank': rank,
                            'similarity_score': 0,
                            'relevance_score': 0,  # Changed from -1 to 0
                            'justification': f"Recommendation failed: {str(e)}",
                            'inference_time': 0
                        })
                
                pbar.update(1)
    
    # Batch evaluation if enabled
    if use_batch and all_pairs:
        print(f"\nPhase 2: Evaluating {len(all_pairs)} recommendations...")
        evaluation_results = evaluator.evaluate_batch(all_pairs, show_progress=True)
        
        # Combine metadata with evaluation results
        for metadata, eval_result in zip(pair_metadata, evaluation_results):
            results.append({
                **metadata,
                'relevance_score': eval_result['score'],
                'justification': eval_result['justification']
            })
    
    # Convert to DataFrame
    evaluation_df = pd.DataFrame(results)
    
    # Save intermediate results
    if save_intermediate:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_results_{timestamp}.csv'
        evaluation_df.to_csv(filename, index=False)
        print(f"\nSaved intermediate results to {filename}")
    
    # Print evaluation statistics
    print(f"\nEvaluation complete! Generated {len(evaluation_df)} evaluation records")
    if hasattr(evaluator, 'get_statistics'):
        stats = evaluator.get_statistics()
        print(f"Evaluator statistics: {stats}")
    
    # Quick quality check
    valid_scores = evaluation_df[evaluation_df['relevance_score'] >= 0]
    if len(valid_scores) > 0:
        print(f"\nScore distribution:")
        score_dist = evaluation_df['relevance_score'].value_counts().sort_index()
        print(score_dist)
        print(f"\nAverage relevance score: {valid_scores['relevance_score'].mean():.2f}")
        print(f"Pass rate (score >= 3): {(valid_scores['relevance_score'] >= 3).mean():.1%}")
    
    return evaluation_df


def calculate_metrics(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive metrics for each model with consistent naming."""
    
    metrics = []
    
    for model_name in evaluation_df['model_name'].unique():
        model_df = evaluation_df[evaluation_df['model_name'] == model_name]
        
        # Filter out failed evaluations (no longer using -1 scores)
        valid_df = model_df[model_df['relevance_score'] >= 0].copy()
        
        if len(valid_df) == 0:
            print(f"Warning: No valid evaluations for {model_name}")
            continue
        
        # Core metrics based on 0-5 scale
        pass_rate = (valid_df['relevance_score'] >= 3).mean()  # 3+ is considered a pass
        fail_rate = (valid_df['relevance_score'] < 3).mean()   # Below 3 is a fail

        # Quality distribution metrics
        irrelevant_rate = (valid_df['relevance_score'] == 0).mean()  # 0 is irrelevant
        weak_rate = (valid_df['relevance_score'] == 1).mean()        # 1 is weak match
        insufficient_rate = (valid_df['relevance_score'] == 2).mean() # 2 is insufficient
        adequate_rate = (valid_df['relevance_score'] == 3).mean()     # 3 is adequate
        strong_rate = (valid_df['relevance_score'] == 4).mean()       # 4 is strong
        rockstar_rate = (valid_df['relevance_score'] == 5).mean()     # 5 is rockstar

        # Grouped quality metrics for easier interpretation
        highly_relevant_rate = (valid_df['relevance_score'] >= 4).mean()  # 4+ is highly relevant
        relevant_rate = (valid_df['relevance_score'] == 3).mean()         # 3 is relevant
        poor_rate = (valid_df['relevance_score'] <= 2).mean()             # 0-2 is poor        
        # Traditional precision@k metrics
        precision_at_3 = pass_rate  # Same as pass_rate for our scoring system
        precision_at_5 = pass_rate  # We typically look at top 5 recommendations
        
        # nDCG@5 calculation (improved)
        ndcg_scores = []
        for resume_id in valid_df['resume_id'].unique():
            resume_recs = valid_df[valid_df['resume_id'] == resume_id].sort_values('rank')
            
            if len(resume_recs) > 1:  # Need at least 2 recommendations for meaningful nDCG
                true_relevance = resume_recs['relevance_score'].values
                
                # Use rank as the predicted scores (inverted so rank 1 = highest score)
                max_rank = resume_recs['rank'].max()
                predicted_scores = (max_rank + 1 - resume_recs['rank']).values
                
                # Alternative: use similarity scores if available and varied
                similarity_scores = resume_recs['similarity_score'].values
                if len(set(similarity_scores)) > 1 and similarity_scores.max() > 0:
                    # Normalize similarity scores
                    sim_min, sim_max = similarity_scores.min(), similarity_scores.max()
                    if sim_max > sim_min:
                        predicted_scores = (similarity_scores - sim_min) / (sim_max - sim_min)
                    else:
                        predicted_scores = np.ones_like(similarity_scores) * 0.5
                else:
                    # Fallback to inverted ranks
                    predicted_scores = (max_rank + 1 - resume_recs['rank']).values
                    predicted_scores = predicted_scores / predicted_scores.max()

                try:
                    ndcg = ndcg_score([true_relevance], [predicted_scores])
                    ndcg_scores.append(ndcg)
                except Exception as e:
                    logger.debug(f"nDCG calculation failed for resume {resume_id}: {e}")
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        
        # Performance metrics
        avg_inference_time = valid_df['inference_time'].mean()
        avg_similarity_score = valid_df['similarity_score'].mean()
        
        # Score distribution
        avg_relevance_score = valid_df['relevance_score'].mean()
        median_relevance_score = valid_df['relevance_score'].median()
        
        metrics.append({
            'model': model_name,
            'precision_at_3': precision_at_3,
            'precision_at_5': precision_at_5,
            'ndcg_at_5': avg_ndcg,
            'pass_rate': pass_rate,
            'fail_rate': fail_rate,

            # Quality distribution (detailed)
            'irrelevant_rate': irrelevant_rate,     # Score 0
            'weak_rate': weak_rate,                 # Score 1
            'insufficient_rate': insufficient_rate, # Score 2
            'adequate_rate': adequate_rate,         # Score 3
            'strong_rate': strong_rate,             # Score 4
            'rockstar_rate': rockstar_rate,         # Score 5

            # Quality distribution (grouped)
            'highly_relevant_rate': highly_relevant_rate,  # Score 4+
            'relevant_rate': relevant_rate,                # Score 3
            'poor_rate': poor_rate,                        # Score 0-2

            'avg_relevance_score': avg_relevance_score,
            'median_relevance_score': median_relevance_score,
            'avg_similarity_score': avg_similarity_score,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'total_evaluations': len(model_df),
            'valid_evaluations': len(valid_df),
            'failed_evaluations': len(model_df) - len(valid_df)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Sort by precision_at_3 for easy comparison
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values('precision_at_3', ascending=False)
    
    return metrics_df


def visualize_model_comparison(metrics_df: pd.DataFrame):
    """Create comprehensive visualization of model performance."""
    
    if metrics_df.empty:
        print("No metrics to visualize")
        return
    
    # Set up the plot with better styling
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Pass Rate comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(metrics_df['model'], metrics_df['pass_rate'], 
                    color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    bars1 = ax1.bar(metrics_df['model'], metrics_df['fail_rate'],
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.set_title('Pass Rate (Score >= 3)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Pass Rate', fontsize=12)
    ax1.set_ylim(0, 1.1)  # Give more space for labels
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Better text positioning for pass rate
    for bar, val in zip(bars1, metrics_df['pass_rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. nDCG@5 comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(metrics_df['model'], metrics_df['ndcg_at_5'], 
                    color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_title('nDCG@5 Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('nDCG@5', fontsize=12)
    ax2.set_ylim(0, 1.1)  # Give more space for labels
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Better text positioning for nDCG
    for bar, val in zip(bars2, metrics_df['ndcg_at_5']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Quality breakdown - Show meaningful score distribution
    ax3 = axes[1, 0]
    x = np.arange(len(metrics_df))
    width = 0.25

    # Create grouped bars showing the actual score distribution with data
    bars_poor = ax3.bar(x - width, metrics_df['poor_rate'], width,
                       label='Poor (0-2)', color='#E74C3C', alpha=0.8, edgecolor='white')
    bars_adequate = ax3.bar(x, metrics_df['adequate_rate'], width,
                           label='Adequate (3)', color='#F39C12', alpha=0.8, edgecolor='white')
    bars_excellent = ax3.bar(x + width, metrics_df['strong_rate'] + metrics_df['rockstar_rate'], width,
                            label='Excellent (4-5)', color='#2ECC71', alpha=0.8, edgecolor='white')

    ax3.set_title('Quality Distribution', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Rate', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_df['model'])
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars for better readability
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0.05:  # Only show label if bar is tall enough
                ax3.text(bar.get_x() + bar.get_width()/2, height/2,
                        f'{val:.0%}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    # Add value labels to show the actual percentages
    add_value_labels(bars_poor, metrics_df['poor_rate'])
    add_value_labels(bars_adequate, metrics_df['adequate_rate'])
    add_value_labels(bars_excellent, metrics_df['strong_rate'] + metrics_df['rockstar_rate'])
    
    # 4. Inference time comparison - Fixed scaling and labels
    ax4 = axes[1, 1]
    bars4 = ax4.bar(metrics_df['model'], metrics_df['avg_inference_time_ms'], 
                    color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax4.set_title('Average Inference Time', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Time (ms)', fontsize=12)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-limit based on data
    max_time = metrics_df['avg_inference_time_ms'].max()
    ax4.set_ylim(0, max_time * 1.15)
    
    # Better text positioning for inference time
    for bar, val in zip(bars4, metrics_df['avg_inference_time_ms']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + max_time * 0.02, 
                f'{val:.0f}ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Improve overall layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    plt.show()


def sanity_check_evaluation(
    models: Dict,
    resume_df: pd.DataFrame,
    n_samples: int = 2,
    n_recommendations: int = 3
):
    """
    Perform a quick sanity check with manual evaluation before running expensive LLM evaluation.
    """
    print("="*60)
    print("SANITY CHECK - Manual Evaluation")
    print("="*60)
    print(f"Testing {n_samples} resumes with {n_recommendations} recommendations each\n")
    
    # Sample resumes
    sample_resumes = resume_df.sample(n=min(n_samples, len(resume_df)), random_state=42)
    
    for idx, resume_row in sample_resumes.iterrows():
        resume_text = resume_row.get('clean_text', resume_row.get('text', ''))
        resume_id = resume_row.get('index', idx)
        resume_category = resume_row.get('category', 'Unknown')
        
        print(f"\n{'='*60}")
        print(f"RESUME ID: {resume_id}")
        print(f"CATEGORY: {resume_category}")
        print(f"PREVIEW: {resume_text[:200]}...")
        print(f"{'='*60}")
        
        for model_name, model in models.items():
            print(f"\n--- {model_name.upper()} RECOMMENDATIONS ---")
            
            try:
                # Get recommendations
                if model_name == 'sbert' and 'domain_bridge' in resume_row:
                    recommendations = model.recommend(
                        resume_text,
                        resume_domains=resume_row['domain_bridge'],
                        top_n=n_recommendations
                    )
                else:
                    recommendations = model.recommend(resume_text, top_n=n_recommendations)
                
                # Display each recommendation
                for rank, rec in enumerate(recommendations, 1):
                    job_text = rec.get('text', rec.get('clean_text', ''))
                    similarity = rec.get('similarity_score', rec.get('semantic_score', 0))
                    
                    print(f"\n  Rank {rank}: Job ID {rec.get('job_id', 'Unknown')}")
                    print(f"  Similarity Score: {similarity:.3f}")
                    print(f"  Domains: {rec.get('job_domains', 'N/A')}")
                    print(f"  Preview: {job_text[:150]}...")
                    
            except Exception as e:
                print(f"  ERROR: Failed to get recommendations - {e}")
    
    print("\n" + "="*60)
    print("SANITY CHECK COMPLETE")
    print("Review the above recommendations before proceeding with full evaluation")
    print("="*60)


def showcase_recommendation_system(
    evaluation_df: pd.DataFrame,
    resume_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame] = None,
    resume_id: Optional[int] = None,
    model_name: str = 'sbert',
    show_scores: bool = False
):
    """
    Showcase the recommendation system using cached evaluation data.

    Displays a single resume and the top 5 LLM-evaluated recommendations,
    with rich formatting and model performance stats. Uses existing evaluation
    data to avoid re-computation.

    Args:
        evaluation_df: DataFrame containing cached evaluations with scores/justifications
        resume_df: DataFrame containing resume data
        jobs_df: DataFrame containing job data
        metrics_df: Model performance metrics DataFrame (optional)
        resume_id: Specific resume ID to showcase (random if None)
        model_name: Model to showcase (default: 'sbert')
        show_scores: Whether to show similarity/semantic scores
    """

    # Select resume to showcase
    if resume_id is None:
        # Get available resume IDs from evaluation data
        available_resumes = evaluation_df['resume_id'].unique()
        if len(available_resumes) == 0:
            print("❌ No evaluation data available for showcase")
            return

        # Use a consistent random selection
        np.random.seed(42)
        resume_idx = np.random.choice(available_resumes)
    else:
        resume_idx = resume_id

    # Get resume details
    sample_resume = resume_df.loc[resume_idx] if resume_idx in resume_df.index else None
    if sample_resume is None:
        print(f"❌ Resume ID {resume_idx} not found in resume dataset")
        return

    # Get evaluations for this resume and model
    resume_evaluations = evaluation_df[
        (evaluation_df['resume_id'] == resume_idx) &
        (evaluation_df['model_name'] == model_name)
    ].copy()

    if resume_evaluations.empty:
        print(f"❌ No evaluation data found for resume {resume_idx} with model {model_name}")
        available_models = evaluation_df[evaluation_df['resume_id'] == resume_idx]['model_name'].unique()
        print(f"Available models for this resume: {list(available_models)}")
        return

    print("\n" + "="*80)
    print("🎯 JOB RECOMMENDATION SYSTEM SHOWCASE")
    print("="*80)

    # Display model performance stats if available
    if metrics_df is not None and not metrics_df.empty:
        print(f"\n📊 MODEL PERFORMANCE OVERVIEW")
        print("─" * 50)

        # Find the selected model stats
        model_stats = metrics_df[metrics_df['model'] == model_name]

        if not model_stats.empty:
            stats = model_stats.iloc[0]
            print(f"🏆 Featured Model: {model_name.upper()}")
            print(f"✅ Pass Rate: {stats['pass_rate']:.1%} | ❌ Fail Rate: {stats['fail_rate']:.1%}")
            print(f"🎯 nDCG@5: {stats['ndcg_at_5']:.3f} (ranking quality)")
            print(f"⚡ Avg Inference: {stats['avg_inference_time_ms']:.0f}ms")
            print(f"🌟 Quality: {stats['poor_rate']:.0%} poor, {stats['relevant_rate']:.0%} adequate, {stats['highly_relevant_rate']:.0%} excellent")

        # Display evaluator info from evaluation data
        total_evaluations = len(evaluation_df)
        valid_evaluations = len(evaluation_df[evaluation_df['relevance_score'] >= 0])
        avg_score = evaluation_df[evaluation_df['relevance_score'] >= 0]['relevance_score'].mean()

        print(f"\n🤖 LLM EVALUATION RESULTS:")
        print(f"📊 Total Evaluations: {total_evaluations:,}")
        print(f"✅ Valid Evaluations: {valid_evaluations:,}")
        print(f"🎯 Average LLM Score: {avg_score:.1f}/5")
    else:
        # Basic info without metrics
        print(f"\n🏆 Featured Model: {model_name.upper()}")
        print(f"🤖 Using cached LLM evaluations")

    # Display resume details
    print(f"\n📄 RESUME PROFILE")
    print("─" * 50)
    print(f"📋 Resume ID: {resume_idx}")
    print(f"🏷️ Category: {sample_resume.get('category', 'Unknown')}")

    # Skills
    skills = sample_resume.get('skills', [])
    if isinstance(skills, (list, np.ndarray)) and len(skills) > 0:
        skills_display = skills[:8] if len(skills) > 8 else skills
        skills_str = ", ".join(skills_display)
        if len(skills) > 8:
            skills_str += f" (+{len(skills)-8} more)"
        print(f"🛠️ Skills: {skills_str}")

    # Domains
    domains = sample_resume.get('domains', []) or sample_resume.get('domain_bridge', [])
    if isinstance(domains, (list, np.ndarray)) and len(domains) > 0:
        domains_str = ", ".join(domains[:5])
        print(f"🎯 Target Domains: {domains_str}")

    # Text preview
    text_preview = sample_resume.get('clean_text', '')[:300]
    print(f"📝 Profile Preview: {text_preview}...")

    # Sort recommendations by LLM relevance score first, then similarity score
    # This gives us the LLM's opinion on what's actually most relevant
    resume_evaluations_sorted = resume_evaluations.sort_values(
        ['relevance_score', 'similarity_score'],
        ascending=[False, False]
    ).head(5)

    print(f"\n\n🔍 TOP 5 LLM-EVALUATED RECOMMENDATIONS FROM {model_name.upper()} MODEL")
    print("═" * 70)

    # Display each recommendation using cached evaluation data
    for rank, (_, eval_row) in enumerate(resume_evaluations_sorted.iterrows(), 1):
        print(f"\n🏆 RANK #{rank}")
        print("─" * 30)

        # Get job details from evaluation row
        job_id = eval_row['job_id']
        relevance_score = eval_row['relevance_score']
        justification = eval_row['justification']
        similarity_score = eval_row['similarity_score']

        # Try to find job details from jobs_df
        job_details = None
        if hasattr(jobs_df, 'loc'):
            # Try to find job by job_id
            matching_jobs = jobs_df[jobs_df['job_id'] == job_id] if 'job_id' in jobs_df.columns else pd.DataFrame()
            if not matching_jobs.empty:
                job_details = matching_jobs.iloc[0]

        # Display job information
        if job_details is not None:
            # Use job title if available, otherwise extract from text
            job_title = job_details.get('title', job_details.get('clean_title', 'Job Title Not Available'))
            if pd.isna(job_title) or job_title == '':
                # Extract title-like info from job text
                job_text = job_details.get('clean_text', '')
                first_words = ' '.join(job_text.split()[:6])
                job_title = first_words.title() if first_words else f"Job {job_id}"

            print(f"💼 Title: {job_title}")
            print(f"🆔 Job ID: {job_id}")

            # Job domains
            job_domains = job_details.get('domains', [])
            if isinstance(job_domains, (list, np.ndarray)) and len(job_domains) > 0:
                domains_str = ", ".join(job_domains[:4])
                print(f"🎯 Domains: {domains_str}")

            # Job skills
            job_skills = job_details.get('skills', [])
            if isinstance(job_skills, (list, np.ndarray)) and len(job_skills) > 0:
                skills_display = job_skills[:6] if len(job_skills) > 6 else job_skills
                skills_str = ", ".join(skills_display)
                if len(job_skills) > 6:
                    skills_str += f" (+{len(job_skills)-6} more)"
                print(f"🛠️ Required Skills: {skills_str}")

            # Job text preview
            job_text = job_details.get('clean_text', 'No description available')
        else:
            # Fallback when job details not found
            job_title = f"Position {job_id}"
            print(f"💼 Title: {job_title}")
            print(f"🆔 Job ID: {job_id}")
            job_text = 'Job description not available in dataset'
            job_skills = []

        # Show similarity scores if requested
        if show_scores:
            print(f"🔢 Similarity Score: {similarity_score:.3f}")

        # Job description preview
        job_preview = job_text[:200] if job_text else "No description available"
        print(f"📋 Description: {job_preview}...")

        # Skill overlap analysis
        if isinstance(skills, (list, np.ndarray)) and isinstance(job_skills, (list, np.ndarray)):
            resume_skills_set = set(str(s).lower() for s in skills)
            job_skills_set = set(str(s).lower() for s in job_skills)
            skill_overlap = resume_skills_set.intersection(job_skills_set)

            if skill_overlap:
                overlap_display = list(skill_overlap)[:4]
                overlap_str = ", ".join(overlap_display)
                if len(skill_overlap) > 4:
                    overlap_str += f" (+{len(skill_overlap)-4} more)"
                print(f"✅ Skill Matches: {overlap_str}")
            else:
                print(f"⚠️ Skill Matches: Limited direct overlap")

        # Display cached LLM Evaluation
        print(f"\n🤖 AI EVALUATION:")

        # Score with emoji indicator
        if relevance_score >= 4:
            score_emoji = "🌟"
            score_label = "Excellent"
        elif relevance_score >= 3:
            score_emoji = "👍"
            score_label = "Good"
        elif relevance_score >= 2:
            score_emoji = "⚠️"
            score_label = "Fair"
        else:
            score_emoji = "❌"
            score_label = "Poor"

        print(f"{score_emoji} Score: {relevance_score}/5 ({score_label})")
        print(f"💭 AI Reasoning: {justification}")

    # Add summary statistics
    print(f"\n\n📊 EVALUATION SUMMARY FOR THIS RESUME:")
    print("─" * 50)
    avg_relevance = resume_evaluations['relevance_score'].mean()
    max_relevance = resume_evaluations['relevance_score'].max()
    pass_count = (resume_evaluations['relevance_score'] >= 3).sum()
    total_count = len(resume_evaluations)

    print(f"🎯 Average LLM Score: {avg_relevance:.1f}/5")
    print(f"🎆 Best LLM Score: {max_relevance}/5")
    print(f"✅ Recommendations Passed (≥3): {pass_count}/{total_count} ({pass_count/total_count:.1%})")

    print("\n" + "═" * 70)
    print("\n💡 INTERPRETATION:")
    print("• Rankings are based on LLM relevance scores, not similarity")
    print("• AI evaluation provides expert-level assessment of job fit")
    print("• Skill matches indicate direct technical compatibility")
    print("• Domain overlap suggests industry/functional fit")
    print("• LLM reasoning explains match quality in human terms")
    print("• Consider both exact matches and transferable skills")
    print("\n🚀 This demonstrates how AI can improve job recommendation ranking!")
    print("\n" + "="*80)
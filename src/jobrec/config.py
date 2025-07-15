"""
Centralised settings so you can swap components or paths by editing ONE file.
Every notebook / module should simply:
    from src import config
"""

import os
from pathlib import Path

# ╭──────────────────────────────────────────────────────────╮
# │                       PATHS                              │
# ╰──────────────────────────────────────────────────────────╯
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR         = ROOT_DIR / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR  = ROOT_DIR / "models"
EMB_DIR     = ROOT_DIR / "embeddings"
REPORT_DIR  = ROOT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

# create dirs if they don't exist (safe in import time)
for _p in (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    EMB_DIR, FIGURES_DIR
):
    _p.mkdir(parents=True, exist_ok=True)

# ╭──────────────────────────────────────────────────────────╮
# │                     NLP / ML                             │
# ╰──────────────────────────────────────────────────────────╯
SBERT_MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL_NAME      = "en_core_web_sm"

# Column produced by your keyword-tagger.
# Swap to "job_categories_ml" when you upgrade to a model-based tagger.
JOB_CATEGORY_COL      = "job_categories_kw"
TOP_N_RECOMMEND       = 5

# Harmonised résumé categories (update to match your mapping dict)
RESUME_CATEGORIES = [
    "data_science",
    "software_engineer",
    "marketing",
    "sales",
    "design",
    "product",
    "other",
]

# List of Job Skills
SKILLS = ['javascript', 'node.js', 'aws', 'kubernetes', 'go lang', 'ruby', 'python', 'sql', 'java', 
          'docker', 'html', 'management', 'engineering', 'marketing', 'design', 'sales', 'software', 
          'development', 'communication', 'leadership', 'installation', 'technical', 'automation', 'power systems']

# List of Industry Domains
DOMAINS = ['healthcare', 'finance', 'tech', 'education', 'manufacturing', 'retail', 'sales', 
           'construction', 'hospitality', 'engineering', 'legal', 'marketing', 'government']

# ╭──────────────────────────────────────────────────────────╮
# │                 Parallel / Random seed                   │
# ╰──────────────────────────────────────────────────────────╯
NUM_CORES      = os.cpu_count() or 4
RANDOM_STATE   = 42

# ╭──────────────────────────────────────────────────────────╮
# │                 Logging / Verbosity                      │
# ╰──────────────────────────────────────────────────────────╯
VERBOSE = False   # set False to silence helper-level printouts

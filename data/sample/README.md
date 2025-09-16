# Sample Data

This directory contains representative sample datasets for demonstration and testing purposes.

## Dataset Overview

### 📄 Files Included

- **`resumes_sample.csv`** (100 resumes) - Representative sample of professional resumes
- **`jobs_sample.csv`** (1,000 jobs) - Sample job postings across multiple industries
- **`bridge.csv`** (120 mappings) - Domain bridge mappings for career transitions

### 📊 Sample Statistics

**Resumes Sample:**
- **Size**: 100 resumes from 2,000 total
- **Categories**: 24 professional categories (Healthcare, Aviation, Teacher, etc.)
- **Features**: Category, clean text, skills, domains, text metrics
- **Avg Length**: ~793 words per resume
- **Skills**: Average 7.5 skills per resume

**Jobs Sample:**
- **Size**: 1,000 jobs from 9,893 total
- **Industries**: Healthcare, IT, Consulting, Retail, etc.
- **Features**: Job ID, title, industry, clean text, skills, domains
- **Avg Length**: ~526 words per job description
- **Skills**: Average 4.3 skills per job

**Bridge Data:**
- **Mappings**: 120 resume-category to job-domain connections
- **Similarity Scores**: Pre-computed relevance scores for career transitions
- **Coverage**: 24 resume categories → 23 job domains

## 🔧 Usage

### Quick Start
```python
import pandas as pd

# Load sample data
resumes = pd.read_csv('data/sample/resumes_sample.csv')
jobs = pd.read_csv('data/sample/jobs_sample.csv')
bridge = pd.read_csv('data/sample/bridge.csv')

print(f"Loaded {len(resumes)} resumes and {len(jobs)} jobs")
```

### Model Testing
```python
from src.jobrec.recommender import ModelFactory

# Initialize models with sample data
factory = ModelFactory()
models = factory.fit_all_models(
    baseline_jobs_df=jobs,
    refined_jobs_df=jobs  # Using same sample for demo
)

# Get recommendations
sbert_model = models['sbert']
sample_resume = resumes.iloc[0]
recommendations = sbert_model.recommend(
    resume_text=sample_resume['clean_text'],
    resume_domains=sample_resume['domain_bridge'],
    top_n=5
)
```

## 📝 Data Schema

### Resume Columns
- `category`: Professional category (e.g., 'HEALTHCARE', 'TEACHER')
- `clean_text`: Preprocessed resume text
- `skills`: Extracted skills list
- `domains`: Relevant professional domains
- `text_length`: Word count
- `domain_bridge`: Career transition mappings

### Job Columns
- `job_id`: Unique job identifier
- `title`: Job title
- `industry`: Industry classification
- `clean_text`: Preprocessed job description
- `skills`: Required skills
- `domains`: Job domain categories
- `text_length`: Description word count

### Bridge Columns
- `resume_category`: Source resume category
- `job_domain`: Target job domain
- `similarity_score`: Relevance score (0-1)
- `rank`: Ranking within category

## 🚫 Limitations

**Sample Bias:**
- Random sampling may not represent edge cases
- Limited to 100 resumes vs 2,000+ in full dataset
- Industry distribution may differ from complete data

**Use Cases:**
- ✅ Model testing and development
- ✅ Code demonstration and tutorials
- ✅ Performance benchmarking (relative comparisons)
- ❌ Production model training (use full dataset)
- ❌ Final performance evaluation (requires complete data)

## 📁 Full Dataset Access

For complete datasets and production use:

1. **Download Original Data:**
   - [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
   - [Job Descriptions (Kaggle)](https://www.kaggle.com/datasets/sudharsan13296/job-descriptions-dataset)

2. **Run Processing Pipeline:**
   ```bash
   # Follow installation guide in INSTALLATION.md
   # Run notebooks 01-03 to recreate full processed datasets
   ```

3. **Use Full Models:**
   - Production models should be trained on complete filtered datasets
   - Final evaluation requires full 10,545+ evaluation samples
   - See `model_metrics.md` for complete performance results

---

*Sample datasets created with fixed seed (42) for reproducible research*
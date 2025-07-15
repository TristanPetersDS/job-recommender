import os
import re, string, pandas as pd
import spacy, nltk
from tqdm import tqdm

from pandarallel import pandarallel

from .config import SKILLS, DOMAINS

# Initialize nlp pipeline
_nlp = spacy.load("en_core_web_sm")

# Initialize pandarallel for parallelization
cpu_cores = os.cpu_count()
pandarallel.initialize(nb_workers=cpu_cores)

def preprocessing_pipeline(
    df: pd.DataFrame,
    text_column: str,
    prefix: str = "",
    regex_func=None,
    lemmatize_func=None,
    extract_skills_func=None,
    extract_domains_func=None 
) -> pd.DataFrame:
    """
    Applies the preprocessing pipeline, splitting original and lemmatized into separate columns.
    """
    temp_df = df.copy()
    # Apply RegEx function
    assert regex_func is not None, "regex_func must be provided"
    temp_df[f"{prefix}clean"] = temp_df[text_column].parallel_apply(regex_func)

    # Lemmatize text
    if not lemmatize_func == None:
        temp_df[[f"{prefix}clean", f"{prefix}clean_lemmatized"]] = temp_df[f"{prefix}clean"].parallel_apply(lemmatize_func).apply(pd.Series)

        # Check to see if user supplied skill extraction function
        if not extract_skills_func == None:
            # Extract skills from original text
            temp_df[f"{prefix}skills"] = temp_df[f"{prefix}clean"].parallel_apply(extract_skills_func)
    
            # Check if user supplied domain extraction function
            if not extract_domains == None:
                # Extract domains from original text and skills
                temp_df[f"{prefix}domains"] = temp_df.parallel_apply(lambda row: extract_domains_func(row[f"{prefix}clean"], row[f"{prefix}skills"]), axis=1)

    return temp_df

def regex_text(text):
    if pd.isna(text):
        return ""
    # Lowercase
    text = text.lower()

    # Remove emails, URLs
    text = re.sub(r"(http\S+|www\S+)", " ", text)                
    text = re.sub(r"[\w._%+-]+@[\w.-]+", " ", text)              

    # Replace special chars and digits with space to preserve separation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                     
    text = re.sub(r"\s+", " ", text).strip()  

    return text

def lemmatize_text(text):
    if pd.isna(text):
        return "", ""

    doc = _nlp(text)
    original_terms = [token.text for token in doc if token.is_alpha]
    lemmatized_terms = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Return as tuple
    return " ".join(original_terms), " ".join(lemmatized_terms)


def extract_skills(cleaned_text: str, skills_list=None):
    """
    Extracts skills from the cleaned, tokenized text (as a string).
    skills_list: list of skills to check against.
    Returns a list of detected skills.
    """
    if pd.isna(cleaned_text) or not isinstance(cleaned_text, str):
        return []

    tokens = cleaned_text.split()

    if skills_list is None:
        skills_list=[]
        for skill in SKILLS:
            skills_list.append(skill.lower().strip()) # fallback to global SKILLS list

    found_skills = [skill for skill in skills_list if skill in tokens]
    return found_skills


def extract_domains(cleaned_text: str, skills=None, domains_list=None):
    """
    Extracts domains from the cleaned, tokenized text (as a string),
    optionally augmented by skill presence logic.
    Returns a list of detected domains.
    """
    if pd.isna(cleaned_text) or not isinstance(cleaned_text, str):
        return []

    tokens = cleaned_text.split()
        
    if domains_list is None:
        domains_list=[]
        for domain in DOMAINS:
            domains_list.append(domain.lower().strip()) # fallback to global SKILLS list

    found_domains = [domain for domain in domains_list if domain in tokens]

    # Domain inference based on detected skills
    # Needs overhaul
    if skills:
        if any(skill in ['aws', 'kubernetes', 'docker'] for skill in skills):
            found_domains.append('tech')
        if any(skill in ['management', 'leadership'] for skill in skills):
            found_domains.append('business')

    return list(set(found_domains))  # unique domains
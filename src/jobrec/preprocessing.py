import os, re, string, pandas as pd, gc
import spacy, nltk, psutil
from spacy.tokens import DocBin
from tqdm import tqdm
from pandarallel import pandarallel

from .config import NUM_CORES, SKILLS, DOMAINS, SPACY_MODEL_NAME

# Initialize once
pandarallel.initialize(nb_workers=NUM_CORES)
_nlp = spacy.load(SPACY_MODEL_NAME)


def preprocessing_pipeline(
    df: pd.DataFrame,
    text_column: str,
    prefix: str = "",
    regex_func=None,
    lemmatize_func=None,
    extract_skills_func=None,
    extract_domains_func=None,
    return_docbin: bool = False,
    batch_size: int = None
) -> pd.DataFrame:
    assert regex_func is not None, "regex_func must be provided"

    # Check to see if the dataframe needs special handlings
    if len(df) <= 5000:
        temp_df = df.copy()
        # Apply RegEx function
        assert regex_func is not None, "regex_func must be provided"
        temp_df[f"{prefix}clean"] = temp_df[text_column].parallel_apply(regex_func)
    
        # Lemmatize text
        if not lemmatize_func == None:
            temp_df[[f"{prefix}clean", f"{prefix}clean_lemmatized", f"{prefix}clean_tokens"]] = temp_df[f"{prefix}clean"].parallel_apply(lemmatize_func).apply(pd.Series)
    
            # Check to see if user supplied skill extraction function
            if not extract_skills_func == None:
                # Extract skills from original text
                temp_df[f"{prefix}skills"] = temp_df[f"{prefix}clean"].parallel_apply(extract_skills_func)
        
                # Check if user supplied domain extraction function
                if not extract_domains == None:
                    # Extract domains from original text and skills
                    temp_df[f"{prefix}domains"] = temp_df.parallel_apply(lambda row: extract_domains_func(row[f"{prefix}clean"], row[f"{prefix}skills"]), axis=1)
    
        return temp_df
    else:
        # Dynamic batch size
        if batch_size is None:
            total_memory = psutil.virtual_memory().total
            batch_size = max(100, min(5000, int(len(df) * 1e8 / total_memory)))
            batch_size = min(batch_size, 5000)
    
        print(f"[INFO] Using batch size: {batch_size}")
    
        chunks = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
        processed_chunks = []
    
        for i, chunk in enumerate(tqdm(chunks, desc="Processing batches")):
            temp_df = chunk.copy()
            temp_df[f"{prefix}clean"] = temp_df[text_column].parallel_apply(regex_func)
    
            if lemmatize_func:
                result = temp_df[f"{prefix}clean"].parallel_apply(lemmatize_func)
                unpacked = pd.DataFrame(result.tolist(), columns=[f"{prefix}clean", f"{prefix}clean_lemmatized", f"{prefix}clean_tokens"])
                
                # If storing docbin, serialize and drop token list
                if return_docbin:
                    docbin = DocBin(store_user_data=True)
                    for doc in unpacked[f"{prefix}clean_tokens"]:
                        if isinstance(doc, spacy.tokens.Doc):
                            docbin.add(doc)
                    temp_df[f"{prefix}clean_tokens"] = [docbin.to_bytes()] * len(temp_df)
                else:
                    temp_df[f"{prefix}clean_tokens"] = unpacked[f"{prefix}clean_tokens"]
    
                temp_df[f"{prefix}clean_lemmatized"] = unpacked[f"{prefix}clean_lemmatized"]
    
            if extract_skills_func:
                temp_df[f"{prefix}skills"] = temp_df[f"{prefix}clean"].parallel_apply(extract_skills_func)
    
                if extract_domains_func:
                    temp_df[f"{prefix}domains"] = temp_df.parallel_apply(
                        lambda row: extract_domains_func(row[f"{prefix}clean"], row[f"{prefix}skills"]),
                        axis=1
                    )
    
            # Drop unneeded objects & collect garbage
            del unpacked, result
            gc.collect()
    
            processed_chunks.append(temp_df)
            del temp_df
            gc.collect()
    
        final_df = pd.concat(processed_chunks, axis=0).reset_index(drop=True)
    
        # Final cleanup
        del processed_chunks
        gc.collect()
    
        return final_df


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
    return " ".join(original_terms), " ".join(lemmatized_terms), doc


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
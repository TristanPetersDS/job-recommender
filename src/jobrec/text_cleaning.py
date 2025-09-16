import os, re, string, numpy as np, pandas as pd, gc
import jsonlines
import spacy, nltk, psutil
import logging
from datetime import datetime
from spacy.tokens import DocBin
from tqdm import tqdm
from pandarallel import pandarallel

from .config import NUM_CORES, DOMAINS, SPACY_MODEL_NAME, MODELS_DIR

# Initialize once
pandarallel.initialize(nb_workers=NUM_CORES)
_nlp = spacy.load(SPACY_MODEL_NAME)
ruler = _nlp.add_pipe('entity_ruler')
ruler.from_disk(MODELS_DIR/'jz_skill_patterns.jsonl')

def setup_debug_logging(log_file=None):
    """Setup logging configuration for debug messages."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"preprocessing_debug_{timestamp}.log"
    
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    return logging.getLogger(__name__)

def text_cleaning_pipeline(
    df: pd.DataFrame,
    text_column: str,
    prefix: str = "",
    regex_func=None,
    lemmatize_func=None,
    extract_skills_func=None,
    extract_domains_func=None,
    batch_size: int = None,
    debug: bool = False,
    log_file: str = None
) -> pd.DataFrame:
    """
    Optimized preprocessing pipeline that handles both small and large datasets.
    
    Args:
        df: Input DataFrame
        text_column: Column name containing text to process
        prefix: Prefix for new column names
        regex_func: Function for regex cleaning
        lemmatize_func: Function for lemmatization
        extract_skills_func: Function for skill extraction
        extract_domains_func: Function for domain extraction
        batch_size: Size of batches for large datasets
        debug: Enable debug logging to file
        log_file: Custom log file path (if None, auto-generated)
    
    Returns:
        Processed DataFrame
    """
    assert regex_func is not None, "regex_func must be provided"
    
    # Setup logging if debug is enabled
    logger = None
    if debug:
        logger = setup_debug_logging(log_file)
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 50)
    
    # Create a copy to avoid modifying original
    temp_df = df.copy().reset_index(drop=True)
    
    if debug and logger:
        logger.debug(f"Input DataFrame shape: {temp_df.shape}")
        logger.debug(f"Processing column: {text_column}")
        logger.debug(f"Column data type: {temp_df[text_column].dtype}")
        logger.debug(f"Null values in column: {temp_df[text_column].isnull().sum()}")
    
    # Check to see if the dataframe needs special handling
    if len(temp_df) <= 5000:
        if debug and logger:
            logger.info("Using small dataframe processing path")
        temp_df = _process_small_dataframe(
            temp_df, text_column, prefix, regex_func, 
            lemmatize_func, extract_skills_func, extract_domains_func, logger
        )
    else:
        if debug and logger:
            logger.info("Using large dataframe processing path")
        temp_df = _process_large_dataframe(
            temp_df, text_column, prefix, regex_func, 
            lemmatize_func, extract_skills_func, extract_domains_func, 
            batch_size, logger
        )
    
    # Final cleanup and duplicate removal
    if debug and logger:
        logger.debug(f"Before duplicate removal: {temp_df.shape}")
    
    # Remove any potential duplicates that might have been introduced
    temp_df = _remove_duplicates_safe(temp_df, debug, logger)
    
    if debug and logger:
        logger.debug(f"After duplicate removal: {temp_df.shape}")
        logger.info("Preprocessing pipeline completed successfully")
        logger.info("=" * 50)
    
    return temp_df

def _process_small_dataframe(df, text_column, prefix, regex_func, 
                           lemmatize_func, extract_skills_func, extract_domains_func, logger):
    """Process small dataframes (≤5000 rows) using parallel processing."""
    
    if logger:
        logger.info("Processing small dataframe with parallel processing")
    
    # Create progress bar for regex cleaning
    print("Applying regex cleaning...")
    tqdm.pandas(desc="Regex cleaning", leave=False, dynamic_ncols=True)
    df[f"{prefix}clean"] = df[text_column].progress_apply(regex_func)
    
    # Lemmatize text
    if lemmatize_func is not None:
        if logger:
            logger.info("Applying lemmatization")
        
        print("Applying lemmatization...")
        tqdm.pandas(desc="Lemmatization", leave=False, dynamic_ncols=True)
        
        # Apply lemmatization and handle the tuple return properly
        lemma_results = df[f"{prefix}clean"].progress_apply(lemmatize_func)
        lemma_df = pd.DataFrame(lemma_results.tolist(), 
                               columns=[f"{prefix}clean_tokens", f"{prefix}clean_lemmatized"],
                               index=df.index)
        
        df = pd.concat([df, lemma_df], axis=1)
        
        if logger:
            logger.debug(f"Lemmatization completed, new columns added: {lemma_df.columns.tolist()}")
        
        # Extract skills from lemmatized tokens
        if extract_skills_func is not None:
            if logger:
                logger.info("Extracting skills")
            
            print("Extracting skills...")
            tqdm.pandas(desc="Skill extraction", leave=False, dynamic_ncols=True)
            df[f"{prefix}skills"] = df[f"{prefix}clean_tokens"].progress_apply(extract_skills_func)
            
            if logger:
                sample_skills = df[f"{prefix}skills"].iloc[0] if len(df) > 0 else []
                logger.debug(f"Sample extracted skills: {sample_skills}")
            
            # Extract domains from lemmatized tokens and skills
            if extract_domains_func is not None:
                if logger:
                    logger.info("Extracting domains")
                
                print("Extracting domains...")
                tqdm.pandas(desc="Domain extraction", leave=False, dynamic_ncols=True)
                df[f"{prefix}domains"] = df.progress_apply(
                    lambda row: extract_domains_func(row[f"{prefix}clean_tokens"], row[f"{prefix}skills"]), 
                    axis=1
                )
                
                if logger:
                    sample_domains = df[f"{prefix}domains"].iloc[0] if len(df) > 0 else []
                    logger.debug(f"Sample extracted domains: {sample_domains}")
    
    return df

def _process_large_dataframe(df, text_column, prefix, regex_func, 
                           lemmatize_func, extract_skills_func, extract_domains_func, 
                           batch_size, logger):
    """Process large dataframes (>5000 rows) using batching."""
    
    # Dynamic batch size calculation
    if batch_size is None:
        total_memory = psutil.virtual_memory().total
        batch_size = max(100, min(5000, int(len(df) * 1e8 / total_memory)))
        batch_size = min(batch_size, 5000)
    
    if logger:
        logger.info(f"Processing large dataframe with batch size: {batch_size}")
        logger.debug(f"Total memory available: {total_memory / (1024**3):.2f} GB")
    
    print(f"Processing {len(df):,} rows in batches of {batch_size:,}")
    
    # Create batches ensuring no overlap
    chunks = []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size].copy()
        chunks.append(chunk)
    
    if logger:
        logger.debug(f"Created {len(chunks)} chunks for processing")
    
    processed_chunks = []
    
    # Main processing loop with progress bar
    with tqdm(total=len(chunks), desc="Processing batches", 
              leave=False, dynamic_ncols=True, unit="batch") as pbar:
        
        for i, chunk in enumerate(chunks):
            if logger:
                logger.debug(f"Processing batch {i+1}/{len(chunks)}, shape: {chunk.shape}")
            
            # Apply regex cleaning
            chunk[f"{prefix}clean"] = chunk[text_column].parallel_apply(regex_func)
            
            # Lemmatization
            if lemmatize_func is not None:
                lemma_results = chunk[f"{prefix}clean"].parallel_apply(lemmatize_func)
                lemma_df = pd.DataFrame(lemma_results.tolist(), 
                                       columns=[f"{prefix}clean_tokens", f"{prefix}clean_lemmatized"],
                                       index=chunk.index)
                
                chunk = pd.concat([chunk, lemma_df], axis=1)
                
                if logger:
                    logger.debug(f"Batch {i+1}: Lemmatization completed")
                
                # Skills extraction
                if extract_skills_func is not None:
                    chunk[f"{prefix}skills"] = chunk[f"{prefix}clean_tokens"].parallel_apply(extract_skills_func)
                    
                    if logger:
                        logger.debug(f"Batch {i+1}: Skills extraction completed")
                    
                    # Domain extraction - Fixed parameter consistency
                    if extract_domains_func is not None:
                        chunk[f"{prefix}domains"] = chunk.parallel_apply(
                            lambda row: extract_domains_func(row[f"{prefix}clean_tokens"], row[f"{prefix}skills"]),
                            axis=1
                        )
                        
                        if logger:
                            logger.debug(f"Batch {i+1}: Domain extraction completed")
                
                # Clean up intermediate variables
                del lemma_results, lemma_df
            
            processed_chunks.append(chunk)
            
            # Update progress bar
            pbar.set_postfix({
                'Processed': f"{(i+1)*batch_size:,}/{len(df):,}", 
                'Memory': f"{psutil.Process().memory_info().rss / (1024**2):.0f}MB"
            })
            pbar.update(1)
            
            # Garbage collection after each batch
            del chunk
            gc.collect()
            
            if logger:
                memory_usage = psutil.Process().memory_info().rss / (1024**2)
                logger.debug(f"Batch {i+1} completed. Memory usage: {memory_usage:.2f} MB")
    
    # Concatenate all processed chunks
    if logger:
        logger.info(f"Concatenating {len(processed_chunks)} chunks")
    
    print("Concatenating processed batches...")
    final_df = pd.concat(processed_chunks, axis=0, ignore_index=True)
    
    # Final cleanup
    del processed_chunks
    gc.collect()
    
    if logger:
        final_memory = psutil.Process().memory_info().rss / (1024**2)
        logger.info(f"Final concatenation completed. Final memory usage: {final_memory:.2f} MB")
    
    return final_df

def _remove_duplicates_safe(df, debug=False, logger=None):
    """
    Safely remove duplicates from DataFrame that may contain list columns.
    """
    try:
        # Try normal drop_duplicates first (fastest if no list columns)
        return df.drop_duplicates().reset_index(drop=True)
    except TypeError as e:
        if "unhashable type: 'list'" in str(e):
            if debug and logger:
                logger.warning("Found unhashable list columns, using alternative duplicate removal method")
            
            # Identify list columns vs hashable columns
            list_columns = []
            hashable_columns = []
            
            for col in df.columns:
                try:
                    # Test if column is hashable by trying to hash first non-null value
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_val is not None:
                        hash(sample_val)
                    hashable_columns.append(col)
                except (TypeError, AttributeError):
                    list_columns.append(col)
            
            if debug and logger:
                logger.debug(f"Hashable columns: {hashable_columns}")
                logger.debug(f"List columns: {list_columns}")
            
            if not hashable_columns:
                # If no hashable columns, just reset index and return
                if debug and logger:
                    logger.warning("No hashable columns found, skipping duplicate removal")
                return df.reset_index(drop=True)
            
            # Drop duplicates based only on hashable columns
            temp_df = df.drop_duplicates(subset=hashable_columns).reset_index(drop=True)
            
            if debug and logger:
                original_shape = df.shape
                new_shape = temp_df.shape
                logger.debug(f"Duplicate removal: {original_shape} -> {new_shape}")
                logger.debug(f"Removed {original_shape[0] - new_shape[0]} duplicate rows")
            
            return temp_df
        else:
            # Re-raise if it's a different TypeError
            raise e

def regex_text(text):
    """Improved regex text cleaning function."""
    if pd.isna(text) or text is None:
        return ""
    
    # Ensure text is string
    text = str(text).lower()

    # Remove emails, URLs
    text = re.sub(r"(http\S+|www\S+)", " ", text)                
    text = re.sub(r"[\w._%+-]+@[\w.-]+", " ", text)              

    # Replace special chars and digits with space to preserve separation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                     
    text = re.sub(r"\s+", " ", text).strip()  
    
    # Extra words to filter out
    extra_words = {
        'education', 'business', 'experience', 'play', 'company', 
        'name', 'citi', 'state', 'work', 'manag', 'project', 
        'certificate', 'languages', 'color', 'vision'
    }
    
    # Filter out extra words efficiently
    words = text.split()
    filtered_words = [word for word in words if word not in extra_words]
    
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """Improved lemmatization function with better error handling."""
    if pd.isna(text) or text is None:
        return "", ""

    try:
        text = str(text)
        doc = _nlp(text)
        
        original_terms = [token.text for token in doc if token.is_alpha]
        lemmatized_terms = [token.lemma_ for token in doc 
                           if token.is_alpha and not token.is_stop]

        return " ".join(original_terms), " ".join(lemmatized_terms)
    except Exception as e:
        # Log to file if logger is available, otherwise silent fail
        if hasattr(logging.getLogger(__name__), 'handlers') and logging.getLogger(__name__).handlers:
            logging.getLogger(__name__).warning(f"Error in lemmatization: {e}")
        return "", ""

def extract_skills(cleaned_text: str, skills_list=None):
    """
    Extracts skills from the cleaned, tokenized text.
    Returns a list of detected skills.
    """
    if pd.isna(cleaned_text) or not isinstance(cleaned_text, str) or cleaned_text.strip() == "":
        return []

    try:
        doc = _nlp(cleaned_text)
        skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
        return list(set(skills))  # Remove duplicates
    except Exception as e:
        # Log to file if logger is available, otherwise silent fail
        if hasattr(logging.getLogger(__name__), 'handlers') and logging.getLogger(__name__).handlers:
            logging.getLogger(__name__).warning(f"Error in skill extraction: {e}")
        return []

def extract_domains(cleaned_text: str, skills=None, domains_list=None):
    """
    Extracts domains from the cleaned, tokenized text.
    Returns a list of detected domains.
    """
    if pd.isna(cleaned_text) or not isinstance(cleaned_text, str) or cleaned_text.strip() == "":
        return []

    try:
        tokens = cleaned_text.split()
        
        if domains_list is None:
            domains_list = [domain.lower().strip() for domain in DOMAINS]

        found_domains = [domain for domain in domains_list if domain in tokens]
        return list(set(found_domains))  # Remove duplicates
    except Exception as e:
        # Log to file if logger is available, otherwise silent fail
        if hasattr(logging.getLogger(__name__), 'handlers') and logging.getLogger(__name__).handlers:
            logging.getLogger(__name__).warning(f"Error in domain extraction: {e}")
        return []

def get_skills(text, nlp):
    """Extract skills from a given text using the spaCy model."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    try:
        doc = nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        return list(set(skills))
    except Exception as e:
        # Log to file if logger is available, otherwise silent fail
        if hasattr(logging.getLogger(__name__), 'handlers') and logging.getLogger(__name__).handlers:
            logging.getLogger(__name__).warning(f"Error in get_skills: {e}")
        return []

def unique_skills(skills):
    """Remove duplicate skills from a list."""
    if skills is None or not isinstance(skills, (list, tuple)):
        return []
    return list(set(skills))
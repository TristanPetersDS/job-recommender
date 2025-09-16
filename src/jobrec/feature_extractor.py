import pandas as pd
import numpy as np
import logging
import gc
import psutil
from datetime import datetime
from tqdm import tqdm
from pandarallel import pandarallel
from .config import NUM_CORES

# Initialize pandarallel
pandarallel.initialize(nb_workers=NUM_CORES)

def setup_feature_logging(log_file=None):
    """Setup logging configuration for feature extraction debug messages."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"feature_extraction_debug_{timestamp}.log"
    
    # Create a separate logger for feature extraction
    logger = logging.getLogger('feature_extraction')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

def compute_text_features_vectorized(df: pd.DataFrame, text_column: str, prefix: str = "") -> pd.DataFrame:
    """
    Vectorized computation of text features for better performance.
    Computes all features in a single pass to avoid redundant string operations.
    """
    temp_df = df.copy()
    
    # Convert to string and handle NaN values
    text_series = temp_df[text_column].fillna("").astype(str)
    
    # Split text into words once (vectorized operation)
    words_series = text_series.str.split()
    
    # Compute text length (word count)
    temp_df[f"{prefix}text_length"] = words_series.str.len()
    
    # Compute average word length (vectorized)
    word_lengths = words_series.apply(lambda words: [len(word) for word in words] if words else [0])
    temp_df[f"{prefix}avg_word_length"] = word_lengths.apply(lambda lengths: np.mean(lengths) if lengths else 0)
    
    # Compute unique word count (vectorized)
    temp_df[f"{prefix}unique_word_count"] = words_series.apply(lambda words: len(set(words)) if words else 0)
    
    # Compute lexical diversity (vectorized division)
    temp_df[f"{prefix}lexical_diversity"] = (
        temp_df[f"{prefix}unique_word_count"] / temp_df[f"{prefix}text_length"]
    ).fillna(0)
    
    return temp_df

def compute_text_features_parallel(df: pd.DataFrame, text_column: str, prefix: str = "") -> pd.DataFrame:
    """
    Parallel computation of text features using pandarallel.
    Used for medium-sized datasets where parallelization overhead is worth it.
    """
    temp_df = df.copy()
    
    def compute_features_single_row(text):
        """Compute all features for a single text in one function call."""
        if pd.isna(text) or text == "":
            return pd.Series([0, 0.0, 0, 0.0])
        
        text = str(text)
        words = text.split()
        
        if not words:
            return pd.Series([0, 0.0, 0, 0.0])
        
        text_length = len(words)
        avg_word_length = np.mean([len(word) for word in words])
        unique_word_count = len(set(words))
        lexical_diversity = unique_word_count / text_length if text_length > 0 else 0.0
        
        return pd.Series([text_length, avg_word_length, unique_word_count, lexical_diversity])
    
    # Apply parallel computation
    feature_columns = [
        f"{prefix}text_length",
        f"{prefix}avg_word_length", 
        f"{prefix}unique_word_count",
        f"{prefix}lexical_diversity"
    ]
    
    print("Computing text features...")
    tqdm.pandas(desc="Text features", leave=False, dynamic_ncols=True)
    features_df = temp_df[text_column].progress_apply(compute_features_single_row)
    features_df.columns = feature_columns
    
    # Concatenate with original dataframe
    result_df = pd.concat([temp_df, features_df], axis=1)
    
    return result_df

def text_features_pipeline(
    df: pd.DataFrame, 
    text_column: str, 
    prefix: str = "", 
    batch_size: int = None,
    debug: bool = False,
    log_file: str = None,
    use_parallel: bool = True
) -> pd.DataFrame:
    """
    Optimized pipeline to compute all derived text features for a dataframe.
    
    Adds columns:
    - <prefix>text_length: Total word count
    - <prefix>avg_word_length: Average length of words
    - <prefix>unique_word_count: Number of unique words
    - <prefix>lexical_diversity: Ratio of unique words to total words
    
    Args:
        df: Input DataFrame
        text_column: Column name containing text to process
        prefix: Prefix for new column names
        batch_size: Size of batches for large datasets
        debug: Enable debug logging to file
        log_file: Custom log file path
        use_parallel: Whether to use parallel processing for medium datasets
    
    Returns:
        DataFrame with additional feature columns
    """
    
    # Setup logging if debug is enabled
    logger = None
    if debug:
        logger = setup_feature_logging(log_file)
        logger.info("=" * 50)
        logger.info("Starting feature extraction pipeline")
        logger.info("=" * 50)
    
    # Create a copy to avoid modifying original
    temp_df = df.copy().reset_index(drop=True)
    
    if debug and logger:
        logger.debug(f"Input DataFrame shape: {temp_df.shape}")
        logger.debug(f"Processing column: {text_column}")
        logger.debug(f"Column data type: {temp_df[text_column].dtype}")
        logger.debug(f"Null values in column: {temp_df[text_column].isnull().sum()}")
        logger.debug(f"Use parallel processing: {use_parallel}")
    
    # Determine processing strategy based on dataset size
    if len(temp_df) <= 1000:
        # Small datasets: Use vectorized operations (fastest for small data)
        if debug and logger:
            logger.info("Using vectorized processing for small dataset")
        
        print("Computing text features (vectorized)...")
        temp_df = compute_text_features_vectorized(temp_df, text_column, prefix)
        
    elif len(temp_df) <= 10000 and use_parallel:
        # Medium datasets: Use parallel processing
        if debug and logger:
            logger.info("Using parallel processing for medium dataset")
        
        temp_df = compute_text_features_parallel(temp_df, text_column, prefix)
        
    else:
        # Large datasets: Use batching
        if debug and logger:
            logger.info("Using batch processing for large dataset")
        
        temp_df = _process_large_dataframe_features(
            temp_df, text_column, prefix, batch_size, logger
        )
    
    # Final cleanup and validation
    if debug and logger:
        logger.debug(f"Final DataFrame shape: {temp_df.shape}")
        
        # Log feature statistics
        feature_cols = [col for col in temp_df.columns if col.startswith(prefix) and 
                       any(feat in col for feat in ['text_length', 'avg_word_length', 'unique_word_count', 'lexical_diversity'])]
        
        for col in feature_cols:
            logger.debug(f"{col} - Mean: {temp_df[col].mean():.2f}, Std: {temp_df[col].std():.2f}")
        
        logger.info("Feature extraction pipeline completed successfully")
        logger.info("=" * 50)
    
    return temp_df

def _process_large_dataframe_features(df, text_column, prefix, batch_size, logger):
    """Process large dataframes using batching for feature extraction."""
    
    # Dynamic batch size calculation
    if batch_size is None:
        total_memory = psutil.virtual_memory().total
        batch_size = max(500, min(5000, int(len(df) * 1e8 / total_memory)))
        batch_size = min(batch_size, 5000)
    
    if logger:
        logger.info(f"Processing large dataframe with batch size: {batch_size}")
        logger.debug(f"Total memory available: {total_memory / (1024**3):.2f} GB")
    
    print(f"Processing {len(df):,} rows in batches of {batch_size:,}")
    
    # Create batches
    chunks = []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size].copy()
        chunks.append(chunk)
    
    if logger:
        logger.debug(f"Created {len(chunks)} chunks for processing")
    
    processed_chunks = []
    
    # Main processing loop with progress bar
    with tqdm(total=len(chunks), desc="Processing feature batches", 
              leave=False, dynamic_ncols=True, unit="batch") as pbar:
        
        for i, chunk in enumerate(chunks):
            if logger:
                logger.debug(f"Processing feature batch {i+1}/{len(chunks)}, shape: {chunk.shape}")
            
            # Use vectorized processing for each chunk
            processed_chunk = compute_text_features_vectorized(chunk, text_column, prefix)
            processed_chunks.append(processed_chunk)
            
            # Update progress bar
            pbar.set_postfix({
                'Processed': f"{(i+1)*batch_size:,}/{len(df):,}",
                'Memory': f"{psutil.Process().memory_info().rss / (1024**2):.0f}MB"
            })
            pbar.update(1)
            
            # Garbage collection
            del chunk, processed_chunk
            gc.collect()
            
            if logger:
                memory_usage = psutil.Process().memory_info().rss / (1024**2)
                logger.debug(f"Feature batch {i+1} completed. Memory usage: {memory_usage:.2f} MB")
    
    # Concatenate all processed chunks
    if logger:
        logger.info(f"Concatenating {len(processed_chunks)} feature chunks")
    
    print("Concatenating feature batches...")
    final_df = pd.concat(processed_chunks, axis=0, ignore_index=True)
    
    # Final cleanup
    del processed_chunks
    gc.collect()
    
    if logger:
        final_memory = psutil.Process().memory_info().rss / (1024**2)
        logger.info(f"Feature concatenation completed. Final memory usage: {final_memory:.2f} MB")
    
    return final_df

# Legacy functions for backward compatibility (now optimized)
def compute_text_length(df: pd.DataFrame, text_column: str, new_column: str = "text_length") -> pd.DataFrame:
    """
    Computes total word count for each document.
    Optimized version using vectorized operations.
    """
    temp_df = df.copy()
    temp_df[new_column] = temp_df[text_column].fillna("").astype(str).str.split().str.len()
    return temp_df

def compute_avg_word_length(df: pd.DataFrame, text_column: str, new_column: str = "avg_word_length") -> pd.DataFrame:
    """
    Computes the average word length in each document.
    Optimized version.
    """
    temp_df = df.copy()
    
    def avg_length_vectorized(text):
        if pd.isna(text) or text == "":
            return 0.0
        words = str(text).split()
        return np.mean([len(word) for word in words]) if words else 0.0
    
    temp_df[new_column] = temp_df[text_column].apply(avg_length_vectorized)
    return temp_df

def compute_unique_word_count(df: pd.DataFrame, text_column: str, new_column: str = "unique_word_count") -> pd.DataFrame:
    """
    Computes the number of unique words in each document.
    Optimized version using vectorized operations.
    """
    temp_df = df.copy()
    temp_df[new_column] = temp_df[text_column].fillna("").astype(str).str.split().apply(lambda words: len(set(words)) if words else 0)
    return temp_df

def compute_lexical_diversity(df: pd.DataFrame, unique_column: str = "unique_word_count",
                               length_column: str = "text_length", new_column: str = "lexical_diversity") -> pd.DataFrame:
    """
    Computes lexical diversity as the ratio of unique words to total words.
    Optimized version with better error handling.
    """
    temp_df = df.copy()
    temp_df[new_column] = (temp_df[unique_column] / temp_df[length_column]).fillna(0)
    
    # Handle any infinite values
    temp_df[new_column] = temp_df[new_column].replace([np.inf, -np.inf], 0)
    
    return temp_df
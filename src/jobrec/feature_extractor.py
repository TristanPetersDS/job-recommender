import pandas as pd
import numpy as np

def compute_text_length(df: pd.DataFrame, text_column: str, new_column: str = "text_length") -> pd.Series:
    """
    Computes total word count for each document.
    """
    df[new_column] = df[text_column].apply(lambda x: len(str(x).split()))
    return df

def compute_avg_word_length(df: pd.DataFrame, text_column: str, new_column: str = "avg_word_length") -> pd.Series:
    """
    Computes the average word length in each document.
    """
    def avg_length(text):
        words = str(text).split()
        return np.mean([len(word) for word in words]) if words else 0
    df[new_column] = df[text_column].apply(avg_length)
    return df

def compute_unique_word_count(df: pd.DataFrame, text_column: str, new_column: str = "unique_word_count") -> pd.Series:
    """
    Computes the number of unique words in each document.
    """
    df[new_column] = df[text_column].apply(lambda x: len(set(str(x).split())))
    return df

def compute_lexical_diversity(df: pd.DataFrame, unique_column: str = "unique_word_count",
                               length_column: str = "text_length", new_column: str = "lexical_diversity") -> pd.Series:
    """
    Computes lexical diversity as the ratio of unique words to total words.
    """
    df[new_column] = df[unique_column] / df[length_column]
    df[new_column] = df[new_column].fillna(0)  # handles division by zero if needed
    return df

def text_features_pipeline(df: pd.DataFrame, text_column: str, prefix: str = "") -> pd.DataFrame:
    """
    Computes all derived text features for a dataframe and a text column.
    Adds columns:
    - <prefix>text_length
    - <prefix>avg_word_length
    - <prefix>unique_word_count
    - <prefix>lexical_diversity
    """
    length_col = f"{prefix}text_length"
    avg_len_col = f"{prefix}avg_word_length"
    unique_col = f"{prefix}unique_word_count"
    diversity_col = f"{prefix}lexical_diversity"

    df = compute_text_length(df, text_column, length_col)
    df = compute_avg_word_length(df, text_column, avg_len_col)
    df = compute_unique_word_count(df, text_column, unique_col)
    df = compute_lexical_diversity(df, unique_col, length_col, diversity_col)
    return df
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def _ensure_label_list(x):
    """Return a list of string labels for a cell that may be str/list/ndarray/etc."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple, set)):
        return [str(e) for e in x if e is not None and not (isinstance(e, float) and np.isnan(e))]
    return [str(x)]  # scalar

def label_frequency(labels: pd.Series) -> pd.DataFrame:
    """Count labels for a (possibly multi-label) column and return a tidy frequency table."""
    ctr = Counter()
    for v in labels:
        ctr.update(_ensure_label_list(v))
    tbl = (pd.Series(ctr).sort_values(ascending=False)
           .rename('count').to_frame()
           .assign(pct=lambda d: (d['count']*100/d['count'].sum()).round(2)))
    tbl.index.name = 'label'
    return tbl

def tfidf_top_terms(texts: pd.Series,
                          labels: pd.Series,
                          k: int = 15,
                          min_df: int = 5,
                          min_docs: int = 3,
                          top_labels: int | None = None) -> dict[str, list[str]]:
    """
    Compute top-k TF‑IDF signature terms for each label in a (possibly multi‑label) column.
    - `texts`: cleaned/lemmatized text column
    - `labels`: category/domain column; each row may be a scalar or an array-like of labels
    - `min_docs`: skip labels with too few rows for stable estimates
    - `top_labels`: if set, restrict to the most frequent N labels (keeps output readable)
    """
    texts = texts.fillna("").astype(str)
    tfidf = TfidfVectorizer(min_df=min_df, stop_words="english")
    X = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())

    # Build label -> row indices map
    label_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, val in enumerate(labels):
        for lab in _ensure_label_list(val):
            if lab:
                label_to_rows[lab].append(i)

    # Keep only the top N labels
    kept_order = None
    if top_labels is not None:
        counts = sorted(((lab, len(rows)) for lab, rows in label_to_rows.items()),
                        key=lambda t: t[1], reverse=True)
        kept_order = [lab for lab, _ in counts[:top_labels]]
        keep = set(kept_order)
        label_to_rows = {lab: rows for lab, rows in label_to_rows.items() if lab in keep}

    out: dict[str, list[str]] = {}
    for lab, rows in label_to_rows.items():
        if len(rows) < min_docs:
            continue
        means = X[rows].mean(axis=0).A1
        if k >= means.size:
            top_idx_sorted = np.argsort(means)[::-1]
        else:
            cand = np.argpartition(means, -k)[-k:]
            top_idx_sorted = cand[np.argsort(means[cand])[::-1]]
        out[lab] = vocab[top_idx_sorted].tolist()

    # If we restricted labels, present results in frequency order
    if kept_order is not None:
        out = {lab: out[lab] for lab in kept_order if lab in out}
    return out

def overlay_hist_grid(jobs_df, resumes_df, job_cols, resume_cols, title_prefix):
    assert len(job_cols) == len(resume_cols), "Column lists must match in length"
    n = len(job_cols)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    
    for ax, job_col, resume_col in zip(axes, job_cols, resume_cols):
        sns.histplot(jobs_df[job_col], kde=True, bins=40, label='Jobs', ax=ax, color='steelblue', alpha=0.5)
        sns.histplot(resumes_df[resume_col], kde=True, bins=40, label='Resumes', ax=ax, color='red', alpha=0.5)
        ax.set_title(job_col.replace("desc_", "").replace("resume_", ""))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend()
    
    fig.suptitle(f"{title_prefix} Comparison Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

def hist_grid_seaborn(df, cols, title_prefix):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3))
    
    for ax, col in zip(axes, cols):
        sns.histplot(df[col], kde=True, bins=40, ax=ax, color='steelblue', alpha=0.6)
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    fig.suptitle(f"{title_prefix} distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_categories(df, col, n=5, cat_type=None, df_title=None, ext_type=None):
    '''
    This function plots the category distributions for a dataframe.

    Parameters:
    df: DataFrame object
    col: Column name for desired category.
    n: Number of categories to plot.
    cat_type: Title of category as a string. 
    df_title: Title of dataframe as a string.
    ext_type: Type of extraction type for extracted/derived features. 
    '''
    cat_freq = label_frequency(df[col])

    plt.figure(figsize=(12,6))
    sns.barplot(data=cat_freq.head(n), y='count', x=cat_freq.head(n).index, palette='viridis', hue=cat_freq.head(n).index)
    
    plt.title(f'Top {n} {ext_type} {df_title} {cat_type}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel(f'# {df_title}s')
    plt.tight_layout()
    plt.show()

    print(f"Top {n}{ext_type} {df_title} {cat_type}:")
    display(cat_freq.head(n))


def plot_missing_data(
    df,
    figsize=(10, 6), 
    color='skyblue', 
    title='Missing Data per Column', 
    save_plot=False,
    save_dir=None,
    filename_prefix=''):
    """
    Plots a bar chart showing the percentage of missing data per column.
    
    Parameters:
    - df: pandas DataFrame
    - figsize: tuple, size of the plot
    - color: str, color of the bars
    - title: str, title of the plot
    - save_plot: bool, whether to save the plot
    - save_dir: str, directory to save the plot if save_plot is True
    - filename_prefix: str, optional prefix for the saved filename
    """
    # Calculate missing percentage
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

    if missing_percent.empty:
        print("No missing values in the DataFrame.")
        return

    # Plot
    plt.figure(figsize=figsize)
    ax = missing_percent.plot(kind='bar', color=color)
    plt.title(title)
    plt.ylabel("Percentage of Missing Values (%)")
    plt.xlabel("Columns")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot if requested
    if save_plot:
        if not save_dir:
            save_dir = input("Enter the directory path to save the plot: ").strip()
        if not filename_prefix:
            filename_prefix = input("Enter a filename prefix (optional): ").strip()

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Clean title for filename
        safe_title = title.replace(' ', '_').lower()
        filename = f"{filename_prefix}_{safe_title}.png" if filename_prefix else f"{safe_title}.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, dpi=300)
    plt.show()

def wordcloud(df):
    txt = ' '.join(txt for txt in df['Resume'])
    wordcloud = WordCloud(
        height=2000,
        width=4000,
        colormap=WORDCLOUD_COLOR_MAP
    ).generate(txt)

    return wordcloud


def generate_wordcloud_from_df(df, category):
    txt = ' '.join(txt for txt in df['Resume'])
    wc = WordCloud(
        height=2000,
        width=4000,
        colormap=WORDCLOUD_COLOR_MAP
    ).generate(txt)
    return category, wc

def wordfreq(df):
    count = df['Resume'].str.split(expand=True).stack().value_counts().reset_index()
    count.columns = ['Word', 'Frequency']

    return count.head(10)    
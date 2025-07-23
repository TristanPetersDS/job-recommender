import os
import pandas

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud


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
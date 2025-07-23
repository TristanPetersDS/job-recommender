import pandas as pd
import numpy as np

def check_missing(df):
    result = pd.concat([df.isnull().sum(),
                   100 * (df.isnull().mean())], axis=1)
    result.columns = ['Missing', 'Total % Missing']
    return result.sort_values(by='Missing', ascending=False)

def check_duplicates(df):    
    present = df.notnull().sum()

    # Handle unhashable types gracefully
    def safe_nunique(series):
        try:
            return series.nunique()
        except TypeError:
            return series.apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x).nunique()

    unique = df.apply(safe_nunique)

    # Compute repeated count and duplicate count
    repeated = present - unique

    result = pd.concat([unique, repeated,  100 * (unique / present)], axis=1)
    result.columns = ['Unique Entries', 'Duplicate Count', 'Ratio Unique (%)']

    return result.sort_values(by=['Duplicate Count'], ascending=False)

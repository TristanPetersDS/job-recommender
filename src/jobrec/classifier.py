import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src import config

def train_resume_classifier(df: pd.DataFrame, text_col: str, label_col: str):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], stratify=df[label_col], test_size=0.2, random_state=42
    )
    vect = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vect.fit_transform(X_train)
    clf = LinearSVC()
    clf.fit(X_train_vec, y_train)
    # persist
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vect, config.MODELS_DIR / "resume_vectorizer.pkl")
    joblib.dump(clf,  config.MODELS_DIR / "resume_clf.pkl")
    return clf, vect, X_test, y_test

def load_resume_classifier():
    vect = joblib.load(config.MODELS_DIR / "resume_vectorizer.pkl")
    clf  = joblib.load(config.MODELS_DIR / "resume_clf.pkl")
    return clf, vect

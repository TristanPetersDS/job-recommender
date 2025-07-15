CATEGORY_KEYWORDS = {
    "data_science": ["data scientist", "machine learning", "nlp"],
    "software_engineer": ["software engineer", "backend", "full stack", "devops"],
    "marketing": ["seo", "content marketing", "digital ads"],
    "sales": ["sales representative", "account executive"],
    # …etc
}

def job_category_tagger(title: str, description: str) -> list[str]:
    text = f"{title} {description}".lower()
    return [
        cat for cat, kws in CATEGORY_KEYWORDS.items()
        if any(kw in text for kw in kws)
    ] or ["other"]

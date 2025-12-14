import re
import spacy
import pandas as pd

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# STOPWORDS from spaCy
STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

def preprocess_text(text):
    """Aggressive cleaning + spaCy lemmatization + negation-safe stopword removal"""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).lower()                               # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)             # remove URLs
    text = re.sub(r"<.*?>", "", text)                      # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                  # keep only letters (aggressive)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize with spaCy, remove stopwords (but keep negation words), lemmatize
    doc = nlp(text)
    tokens = []
    for token in doc:
        t = token.text.lower()
        # keep 'not' and other negations; adjust list if you want to preserve more function words
        keep_negations = (t in {"not", "no", "nor", "never"})
        if (t not in STOPWORDS or keep_negations) and (not token.is_punct) and (not token.is_space):
            tokens.append(token.lemma_)
    return " ".join(tokens)

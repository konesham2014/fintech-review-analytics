import pandas as pd
import re
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 1. LOAD DATA
# =========================
def load_data(path="data/raw/bank_reviews_cleaned.csv"):
    return pd.read_csv(path)


# =========================
# 2. SENTIMENT MODEL (DistilBERT)
# =========================
def build_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


def get_sentiment(text, model):
    try:
        result = model(str(text)[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        return label, score
    except:
        return "neutral", 0.0


def apply_sentiment(df, model):
    sentiments = df["review"].apply(lambda x: get_sentiment(x, model))
    df["sentiment_label"], df["sentiment_score"] = zip(*sentiments)
    return df


# =========================
# 3. TEXT CLEANING (spaCy)
# =========================
def load_spacy():
    return spacy.load("en_core_web_sm")


def clean_text(text, nlp):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and len(token) > 2
    ]

    return " ".join(tokens)


def apply_cleaning(df, nlp):
    df["clean_review"] = df["review"].apply(lambda x: clean_text(x, nlp))
    return df


# =========================
# 4. TF-IDF KEYWORDS
# =========================
def extract_keywords(df):
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(df["clean_review"])
    return tfidf, tfidf_matrix


# =========================
# 5. THEME CLASSIFICATION
# =========================
def assign_theme(text):
    text = str(text).lower()

    if any(x in text for x in ["login", "otp", "password", "sign"]):
        return "Account Access Issues"

    if any(x in text for x in ["transfer", "slow", "delay", "send money"]):
        return "Transaction Performance"

    if any(x in text for x in ["crash", "error", "bug", "freeze"]):
        return "App Stability"

    if any(x in text for x in ["ui", "design", "interface"]):
        return "UI & Design"

    if any(x in text for x in ["feature", "add", "request", "need"]):
        return "Feature Requests"

    return "General Feedback"


def apply_themes(df):
    df["identified_theme"] = df["clean_review"].apply(assign_theme)
    return df


# =========================
# 6. PIPELINE RUNNER
# =========================
def run_pipeline():
    df = load_data()

    # add ID
    df = df.reset_index().rename(columns={"index": "review_id"})

    print("Loading models...")
    sentiment_model = build_sentiment_model()
    nlp = load_spacy()

    print("Running sentiment analysis...")
    df = apply_sentiment(df, sentiment_model)

    print("Cleaning text...")
    df = apply_cleaning(df, nlp)

    print("Extracting keywords...")
    extract_keywords(df)

    print("Assigning themes...")
    df = apply_themes(df)

    # =========================
    # 7. OUTPUT (REQUIRED FORMAT)
    # =========================
    final_df = df[[
        "review_id",
        "review",
        "sentiment_label",
        "sentiment_score",
        "identified_theme"
    ]]

    output_path = "data/raw/task2_output.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Task 2 complete ✔ saved to {output_path}")
    print(final_df.head())


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_pipeline()
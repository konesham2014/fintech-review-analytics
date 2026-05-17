import pandas as pd
import re
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/raw/bank_reviews_cleaned.csv")

# Add review_id (required)
df = df.reset_index().rename(columns={"index": "review_id"})

# =========================
# 2. SENTIMENT MODEL (DistilBERT)
# =========================
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def get_sentiment(text):
    try:
        result = sentiment_model(str(text)[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        return label, score
    except:
        return "neutral", 0.0


df["sentiment_label"], df["sentiment_score"] = zip(
    *df["review"].apply(get_sentiment)
)

# =========================
# 3. TEXT CLEANING (spaCy)
# =========================
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and len(token) > 2
    ]

    return " ".join(tokens)


df["clean_review"] = df["review"].apply(clean_text)

# =========================
# 4. TF-IDF KEYWORDS
# =========================
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = tfidf.fit_transform(df["clean_review"])
keywords = tfidf.get_feature_names_out()

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

    if any(x in text for x in ["feature", "add", "need", "request"]):
        return "Feature Requests"

    return "General Feedback"


df["identified_theme"] = df["clean_review"].apply(assign_theme)

# =========================
# 6. FINAL OUTPUT FORMAT (REQUIRED)
# =========================
final_df = df[[
    "review_id",
    "review",
    "sentiment_label",
    "sentiment_score",
    "identified_theme"
]]

final_df.to_csv("data/raw/task2_output.csv", index=False)

print("Task 2 complete ✔")
print(final_df.head())
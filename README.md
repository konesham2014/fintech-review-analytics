# Task 2: Sentiment and Thematic Analysis

## Sentiment Analysis

This project uses the transformer model:

distilbert-base-uncased-finetuned-sst-2-english

via Hugging Face Transformers to classify reviews into:
- Positive
- Negative

The model also generates a confidence score for each prediction.

## Sentiment Model

The project uses Hugging Face's
distilbert-base-uncased-finetuned-sst-2-english
model for sentiment classification.

## Theme Extraction

Themes were identified using:
- spaCy preprocessing
- TF-IDF keyword extraction
- rule-based thematic grouping

## Output

Task 2 generates:
data/raw/task2_output.csv

Columns:
- review_id
- review
- sentiment_label
- sentiment_score
- identified_theme

## NLP Preprocessing

Text preprocessing includes:
- Lowercasing
- Removal of special characters
- Tokenization
- Stop-word removal
- Lemmatization using spaCy

## Thematic Analysis

TF-IDF keyword extraction and rule-based theme classification were used to identify recurring user concerns and satisfaction drivers.

Themes include:
- Account Access Issues
- Transaction Performance
- App Stability
- UI & Design
- Feature Requests
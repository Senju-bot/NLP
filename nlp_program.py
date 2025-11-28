import pandas as pd
import nltk
import spacy
import string
import re

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download NLTK requirements
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    doc = nlp(text)

    tokens = [
        token.lemma_ 
        for token in doc 
        if token.text not in stop_words and token.text not in string.punctuation
    ]

    return " ".join(tokens)

# Sample mini-dataset for training
data = {
    "text": [
        "I love this product, it's amazing!",
        "This is the worst experience ever.",
        "The service was okay, nothing special.",
        "Absolutely fantastic! Highly recommended.",
        "Terrible. I will never buy this again."
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative"]
}

df = pd.DataFrame(data)

df["clean_text"] = df["text"].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, preds))

# Predict sentiment from user
def predict_sentiment(text):
    clean = preprocess(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    return prediction

if __name__ == "__main__":
    user_input = input("\nEnter a sentence to analyze sentiment: ")
    result = predict_sentiment(user_input)
    print("\nPredicted Sentiment:", result)

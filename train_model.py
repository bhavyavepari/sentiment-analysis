import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Load data
data = pd.read_csv('data/sentiment_data.csv')

# Split data (no stratify for now)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42
)

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))  # Suppress warnings

# Save model
joblib.dump(model, 'model/sentiment_model.pkl')
print("Model saved successfully!")

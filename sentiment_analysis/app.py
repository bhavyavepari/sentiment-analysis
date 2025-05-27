# app.py

import streamlit as st
import joblib

# Load model
model = joblib.load('model/sentiment_model.pkl')

# Sentiment mapping
sentiment_dict = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# App UI
st.title("Sentiment Analysis (5-Class)")
st.markdown("Enter a sentence, and the model will predict the sentiment.")

text_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if text_input.strip() != "":
        prediction = model.predict([text_input])[0]
        sentiment = sentiment_dict[prediction]
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text.")


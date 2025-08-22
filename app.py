import streamlit as st
import pickle
import os

# File paths
model_path = "fake_news_model.pkl"
vectorizer_path = "vectorizer.pkl"

# Load model
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    st.error(f"Model file not found: {model_path}")

# Load vectorizer
if os.path.exists(vectorizer_path):
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)
else:
    st.error(f"Vectorizer file not found: {vectorizer_path}")

# Streamlit UI
st.title("Fake News Detection")

news_text = st.text_area("Enter news text here:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text!")
    else:
        try:
            vectorized_text = vectorizer.transform([news_text])
            prediction = model.predict(vectorized_text)[0]
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

import streamlit as st
import pickle
import re

st.title("Simple SMS Classifier Test")

# Load models
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    st.success(f"Models loaded! (Vectorizer type: {type(vectorizer).__name__})")
    
    # Simple preprocessing
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Test messages
    st.subheader("Test with examples:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test with Spam Example"):
            text = "WINNER! You have been selected for a $1000 prize"
            processed = preprocess(text)
            vector = vectorizer.transform([processed])
            prediction = model.predict(vector)[0]
            st.write(f"Message: {text}")
            st.write(f"Prediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
    
    with col2:
        if st.button("Test with Ham Example"):
            text = "I'll be home in 5 minutes"
            processed = preprocess(text)
            vector = vectorizer.transform([processed])
            prediction = model.predict(vector)[0]
            st.write(f"Message: {text}")
            st.write(f"Prediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
    
    # Custom input
    st.subheader("Try your own message:")
    message = st.text_input("Enter a message:")
    if st.button("Classify") and message:
        processed = preprocess(message)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        st.write(f"Prediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write(f"Error type: {type(e).__name__}")
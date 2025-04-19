import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Display info
st.title("Model Training Utility")
st.info("This page trains a new model directly on Streamlit Cloud")

# Simple preprocessing function
def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

try:
    # Train new model
    st.write("Loading dataset...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1': 'target', 'v2': 'text'})
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    
    st.write("Preprocessing text...")
    df['processed_text'] = df['text'].apply(simple_preprocess)
    
    st.write("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['target'], test_size=0.2, random_state=42
    )
    
    # Using CountVectorizer is more reliable than TfidfVectorizer for cloud
    st.write("Training vectorizer...")
    vectorizer = CountVectorizer(max_features=3000)
    all_texts = df['processed_text'].tolist()
    vectorizer.fit(all_texts)
    
    st.write("Training model...")
    X_train_vect = vectorizer.transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)
    
    st.write("Saving models...")
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, protocol=2)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=2)
    
    # Test the model
    st.write("Testing model...")
    test_text = "WINNER!! You have been selected to receive a prize!"
    processed = simple_preprocess(test_text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    
    st.success(f"Model trained successfully! Test prediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
    
    # Show model details
    st.write("**Model Details:**")
    st.code(f"Vectorizer type: {type(vectorizer).__name__}\nModel type: {type(model).__name__}")
    
    # Add a button to go back to the main app
    if st.button("Go to Spam Classifier App"):
        st.experimental_rerun()
    
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Error details:", type(e).__name__)
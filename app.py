import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import os

# Page configuration with dark pink theme
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

# Custom CSS for dark pink theme
st.markdown("""
<style>
    .stApp {
        background-color: #1A0112;
        color: white;
    }
    .stButton>button {
        background-color: #FF3366;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #8B0000;
    }
    h1, h2, h3 {
        color: #FF3366;
    }
    .success-box {
        background-color: #006400;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .error-box {
        background-color: #8B0000;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📱 SMS Spam Classifier")
st.markdown("Detect whether a message is spam or not")
st.markdown("---")

# Simple preprocessing function
def simple_preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to load models
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model, True
    except Exception as e:
        return None, None, False

# Function to train new model
def train_new_model():
    try:
        # Load dataset
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.rename(columns={'v1': 'target', 'v2': 'text'})
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
        df['target'] = df['target'].map({'ham': 0, 'spam': 1})
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(simple_preprocess)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['target'], test_size=0.2, random_state=42
        )
        
        # Create and train vectorizer and model
        vectorizer = CountVectorizer(max_features=3000)
        
        # Fit vectorizer on all text data
        all_texts = df['processed_text'].tolist()
        vectorizer.fit(all_texts)
        
        # Transform training data and train model
        X_train_vect = vectorizer.transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vect, y_train)
        
        # Save models
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f, protocol=4)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        return vectorizer, model, True
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, False

# Sidebar
with st.sidebar:
    st.title("About")
    st.info("This app detects spam messages using a machine learning model trained on SMS data.")
    st.markdown("---")
    
    # Model info section
    st.subheader("Model Information")
    vectorizer, model, models_loaded = load_models()
    
    if models_loaded:
        st.success(f"Model loaded successfully!")
        st.info(f"Vectorizer type: {type(vectorizer).__name__}")
        st.info(f"Model type: {type(model).__name__}")
    else:
        st.error("Model not loaded")
        
        if os.path.exists('spam.csv'):
            if st.button("Train New Model"):
                with st.spinner("Training model..."):
                    vectorizer, model, models_loaded = train_new_model()
                    if models_loaded:
                        st.success("Model trained successfully!")
                        st.experimental_rerun()
        else:
            st.warning("Training data not found")

# Main content
if models_loaded:
    # Example messages
    st.subheader("Try with examples")
    examples = {
        "Example (Ham)": "I'll be there in 5 minutes, wait for me",
        "Example (Spam)": "WINNER!! You have been selected to receive a $1000 cash prize! Call now to claim!"
    }
    
    example_choice = st.selectbox("Select an example or type your own:", 
                                ["Choose an example..."] + list(examples.keys()))
    
    # Input area
    if example_choice != "Choose an example...":
        message = examples[example_choice]
    else:
        message = ""
    
    message = st.text_area("Enter a message to classify:", value=message, height=100)
    
    # Classification logic
    if st.button("Classify Message", type="primary") and message:
        with st.spinner("Analyzing..."):
            # Preprocess text
            processed_text = simple_preprocess(message)
            
            # Make prediction
            try:
                vector = vectorizer.transform([processed_text])
                prediction = model.predict(vector)[0]
                
                # Display result
                if prediction == 1:
                    st.markdown("<div class='error-box'><h2>🚨 SPAM DETECTED</h2>This message has been classified as spam.</div>", 
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div class='success-box'><h2>✅ NOT SPAM</h2>This message appears to be legitimate.</div>", 
                                unsafe_allow_html=True)
                
                # Show details in expander
                with st.expander("See analysis details"):
                    st.write("**Original message:**")
                    st.code(message)
                    st.write("**Processed text:**")
                    st.code(processed_text)
            
            except Exception as e:
                st.error(f"Classification error: {str(e)}")
                if "idf" in str(e).lower():
                    st.warning("There seems to be an issue with the vectorizer. Try retraining the model.")
else:
    st.warning("⚠️ Model is not loaded. Please use the sidebar to train a new model.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray; font-size:12px;'>SMS Spam Classifier • Built with Streamlit</div>", 
            unsafe_allow_html=True)

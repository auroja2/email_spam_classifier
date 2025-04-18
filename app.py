# At the top of your app.py
# At the top of your app.py
import os
import sys
import streamlit as st

# Only try to install packages if they're not already available
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
except ImportError:
    st.error("Required packages are missing. Please make sure to install them with 'pip install -r requirements.txt'")
    st.stop()

import pickle
import string
import pandas as pd
import warnings

# Suppress version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

# Download NLTK resources with better error handling
@st.cache_resource
def download_nltk_resources():
    try:
        # Check if the data already exists to avoid redownloading
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        st.sidebar.success("NLTK resources loaded successfully!")
    except LookupError:
        try:
            # If not found, download the required NLTK data
            nltk.download('stopwords')
            nltk.download('punkt')
            st.sidebar.success("NLTK resources downloaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error downloading NLTK resources: {str(e)}")
            st.stop()

# Call the download function at startup
download_nltk_resources()

# Header
st.title("📱 SMS Spam Classifier")
st.markdown("#### Detect whether a message is spam or not")
st.divider()

# Define the transformation function
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load model and vectorizer with improved error handling
@st.cache_resource
def load_models():
    model_path = 'model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model files not found. Please make sure model.pkl and vectorizer.pkl exist in the application directory.")
        return None, None, False
        
    try:
        with open(vectorizer_path, 'rb') as f:
            tfidf = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return tfidf, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

tfidf, model, models_loaded = load_models()

# Input area
st.subheader("Enter a message")
example_texts = {
    "Example 1 (Ham)": "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Example 2 (Spam)": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Example 3 (Ham)": "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight",
    "Example 4 (Spam)": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."
}

# Example selector
example_option = st.selectbox(
    "Try an example or type your own message below:",
    ["Select an example..."] + list(example_texts.keys())
)

# Set default text
default_text = ""
if example_option != "Select an example...":
    default_text = example_texts[example_option]

# Input text area
message = st.text_area("", value=default_text, height=100, placeholder="Type or paste a message here...")

# Prediction section
if st.button('Classify Message', type="primary"):
    if not models_loaded:
        st.warning("Cannot classify without model files.")
    elif not message:
        st.warning("Please enter a message to classify.")
    else:
        try:
            with st.spinner('Analyzing message...'):
                # Process steps with visual feedback
                col1, col2, col3 = st.columns(3)
                
                # Step 1: Preprocess
                with col1:
                    transformed_message = transform(message)
                    st.success("✓ Preprocessing")
                
                # Step 2: Vectorize
                with col2:
                    vector_input = tfidf.transform([transformed_message])
                    st.success("✓ Vectorization")
                
                # Step 3: Predict
                with col3:
                    prediction = model.predict(vector_input)[0]
                    st.success("✓ Classification")
                
                # Show result with appropriate styling
                st.divider()
                if prediction == 1:
                    st.error("## 🚨 SPAM DETECTED 🚨")
                    st.markdown("This message has been classified as **spam**.")
                else:
                    st.success("## ✅ HAM (NOT SPAM)")
                    st.markdown("This message appears to be legitimate.")
                
                # Show processing details in an expander
                with st.expander("View processing details"):
                    st.markdown("**Original message:**")
                    st.info(message)
                    st.markdown("**Transformed text:**")
                    st.code(transformed_message)
        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")
            st.info("If you're seeing dimension mismatch errors, it's likely the model and vectorizer are incompatible.")

# Add helpful info at the bottom
st.divider()
st.markdown("""
**About this app:**  
This SMS Spam Classifier uses Natural Language Processing and Machine Learning to identify spam messages.
It preprocesses text by removing stopwords, stemming, and transforming it into a numerical representation 
that a Naive Bayes classifier can analyze.
""")

# Add a sidebar with accuracy info if available
try:
    # Try to load the model's accuracy from when it was trained
    with open('model_info.txt', 'r') as f:
        accuracy = f.read().strip()
    
    st.sidebar.header("Model Info")
    st.sidebar.metric("Model Accuracy", accuracy)
except:
    # If the file doesn't exist, show generic information
    st.sidebar.header("Model Info")
    st.sidebar.write("Using Naive Bayes classification with TF-IDF features")

# Add deployment info in the sidebar
st.sidebar.divider()
st.sidebar.markdown("**Project Deployment**")
st.sidebar.markdown("""
- GitHub: [View Source](https://github.com/yourusername/sms-spam-classifier)
- Created: April 2025
""")

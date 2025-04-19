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

# Check for required files at startup
if not os.path.exists('vectorizer.pkl') or not os.path.exists('model.pkl'):
    st.warning("⚠️ Model files not found. You'll need to train a new model.")
    MODEL_FILES_EXIST = False
else:
    MODEL_FILES_EXIST = True

if not os.path.exists('spam.csv'):
    st.warning("⚠️ Dataset file (spam.csv) not found. You won't be able to train a new model.")
    DATASET_EXISTS = False
else:
    DATASET_EXISTS = True

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

# Function to load models with caching for better performance
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Verify vectorizer works by testing it
        test_text = "test message"
        try:
            vectorizer.transform([test_text])
            return vectorizer, model, True
        except Exception as test_err:
            st.sidebar.error(f"Vectorizer test failed: {str(test_err)}")
            return None, None, False
            
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
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
        
        # Save models with protocol=2 for better compatibility
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f, protocol=2)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=2)
        
        # Verify the model works
        test_text = "test message"
        test_vector = vectorizer.transform([test_text])
        model.predict(test_vector)
        
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
    if MODEL_FILES_EXIST:
        vectorizer, model, models_loaded = load_models()
        
        if models_loaded:
            st.success(f"Model loaded successfully!")
            st.info(f"Vectorizer type: {type(vectorizer).__name__}")
            st.info(f"Model type: {type(model).__name__}")
        else:
            st.error("Model files found but could not be loaded properly.")
            
            if DATASET_EXISTS:
                if st.button("Train New Model"):
                    with st.spinner("Training model..."):
                        vectorizer, model, models_loaded = train_new_model()
                        if models_loaded:
                            st.success("Model trained successfully!")
                            st.experimental_rerun()
            else:
                st.warning("Dataset not found. Cannot train new model.")
    else:
        st.error("Model files not found.")
        
        if DATASET_EXISTS:
            if st.button("Train New Model"):
                with st.spinner("Training model..."):
                    vectorizer, model, models_loaded = train_new_model()
                    if models_loaded:
                        st.success("Model trained successfully!")
                        st.experimental_rerun()
        else:
            st.warning("Dataset not found. Cannot train new model.")
            
    # Version info
    st.markdown("---")
    st.caption("Version 1.1.0")

# Main content - only show if models are loaded
if MODEL_FILES_EXIST and 'models_loaded' in locals() and models_loaded:
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
            processed_text

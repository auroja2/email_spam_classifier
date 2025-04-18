import streamlit as st
import pickle
import re
import os
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

# Custom CSS for styling
def add_custom_css():
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
    h1, h2, h3 {
        color: #FF3366;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# Header
st.title("📱 SMS Spam Classifier")
st.markdown("#### Detect whether a message is spam or not")
st.divider()

# Simple preprocessing without NLTK
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

# Load model and vectorizer
@st.cache_resource
def load_models():
    # Get the directory of the current file
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
        model_path = os.path.join(current_dir, 'model.pkl')
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

# First try to load pre-trained models
vectorizer, model, models_loaded = load_models()

# If models aren't available, offer training option
if not models_loaded:
    st.warning("Pre-trained model not found. Would you like to train a new model?")
    
    # Only show this if the dataset is available
    if os.path.exists('spam.csv'):
        if st.button("Train new model"):
            with st.spinner("Training model... This may take a minute."):
                try:
                    # Import here to avoid dependency issues if sklearn isn't available
                    from sklearn.feature_extraction.text import CountVectorizer
                    from sklearn.model_selection import train_test_split
                    from sklearn.naive_bayes import MultinomialNB
                    
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
                    X_train_vect = vectorizer.fit_transform(X_train)
                    model = MultinomialNB()
                    model.fit(X_train_vect, y_train)
                    
                    # Save models in memory
                    models_loaded = True
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    else:
        st.error("Dataset not found. Please upload spam.csv file to train a model.")

# Example messages
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
message = ""
if example_option != "Select an example...":
    message = example_texts[example_option]

# Input text area
message = st.text_area("", value=message, height=100, placeholder="Type or paste a message here...")

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
                    processed_text = simple_preprocess(message)
                    st.success("✓ Preprocessing")
                
                # Step 2: Vectorize
                with col2:
                    vector_input = vectorizer.transform([processed_text])
                    st.success("✓ Vectorization")
                
                # Step 3: Predict
                with col3:
                    prediction = model.predict(vector_input)[0]
                    st.success("✓ Classification")
                
                # Show result with appropriate styling
                st.divider()
                if prediction == 1:
                    st.markdown("<div style='background-color:#8B0000; padding:20px; border-radius:10px;'><h2 style='color:white; text-align:center;'>🚨 SPAM DETECTED 🚨</h2><p style='color:white; text-align:center;'>This message has been classified as spam.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color:#006400; padding:20px; border-radius:10px;'><h2 style='color:white; text-align:center;'>✅ HAM (NOT SPAM)</h2><p style='color:white; text-align:center;'>This message appears to be legitimate.</p></div>", unsafe_allow_html=True)
                
                # Show processing details in an expander
                with st.expander("View processing details"):
                    st.markdown("**Original message:**")
                    st.info(message)
                    st.markdown("**Processed text:**")
                    st.code(processed_text)
        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")
            st.info("If you're seeing fitting errors, try retraining the model.")

# Add info at the bottom
st.divider()
st.markdown("""
**About this app:**  
This SMS Spam Classifier uses Machine Learning to identify spam messages.
It preprocesses text and uses a Naive Bayes classifier to detect patterns commonly found in spam messages.
""")

# Footer
st.markdown("<div style='text-align:center; color:gray; font-size:12px; margin-top:30px;'>SMS Spam Classifier • Deployed on Streamlit Cloud</div>", unsafe_allow_html=True)

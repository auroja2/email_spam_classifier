import streamlit as st
import pickle
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

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
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

vectorizer, model, models_loaded = load_models()

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
                    st.error("## 🚨 SPAM DETECTED 🚨")
                    st.markdown("This message has been classified as **spam**.")
                else:
                    st.success("## ✅ HAM (NOT SPAM)")
                    st.markdown("This message appears to be legitimate.")
                
                # Show processing details in an expander
                with st.expander("View processing details"):
                    st.markdown("**Original message:**")
                    st.info(message)
                    st.markdown("**Processed text:**")
                    st.code(processed_text)
        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")


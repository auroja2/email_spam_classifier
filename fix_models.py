import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

print("Starting model conversion process...")

# Simple preprocessing function
def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create new models with CountVectorizer (more reliable)
try:
    print("Loading dataset...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1': 'target', 'v2': 'text'})
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(simple_preprocess)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['target'], test_size=0.2, random_state=42
    )
    
    # Create CountVectorizer (more reliable than TF-IDF for cloud deployment)
    print("Creating and fitting vectorizer...")
    vectorizer = CountVectorizer(max_features=3000)
    all_texts = df['processed_text'].tolist()
    vectorizer.fit(all_texts)
    
    # Train model
    print("Training model...")
    X_train_vect = vectorizer.transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)
    
    # Save models with protocol=2 (for maximum compatibility)
    print("Saving models with protocol 2...")
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, protocol=2)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=2)
    
    # Test the models
    print("Testing models...")
    with open('vectorizer.pkl', 'rb') as f:
        test_vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    
    # Try prediction
    test_text = "WINNER! You have been selected to receive a $1000 prize!"
    processed_test = simple_preprocess(test_text)
    test_vector = test_vectorizer.transform([processed_test])
    prediction = test_model.predict(test_vector)[0]
    print(f"Test prediction for spam message: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
    
    # Try another prediction
    test_text2 = "I'll be home in 5 minutes"
    processed_test2 = simple_preprocess(test_text2)
    test_vector2 = test_vectorizer.transform([processed_test2])
    prediction2 = test_model.predict(test_vector2)[0]
    print(f"Test prediction for ham message: {'SPAM' if prediction2 == 1 else 'NOT SPAM'}")
    
    print("Models created and tested successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
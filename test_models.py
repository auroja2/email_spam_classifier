import pandas as pd
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Starting model training process...")

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

# Load dataset
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.rename(columns={'v1': 'target', 'v2': 'text'})
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Preprocess text
print("Preprocessing text...")
df['processed_text'] = df['text'].apply(simple_preprocess)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['target'], test_size=0.2, random_state=42
)

# Create and train vectorizer and model
print("Training vectorizer and model...")
vectorizer = TfidfVectorizer(max_features=3000)

# This is critical - fit the vectorizer on ALL texts before pickling
all_texts = list(df['processed_text'])  # Include ALL texts, not just training
vectorizer.fit(all_texts)  # Fit the vectorizer on all texts

# Now transform just the training data for the model
X_train_vect = vectorizer.transform(X_train)
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Evaluate model
X_test_vect = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save models using protocol=4 for better compatibility
print("Saving models...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f, protocol=4)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

print("Model and vectorizer saved successfully!")

# Verify models work
print("Testing model...")
test_message = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
processed_message = simple_preprocess(test_message)
test_vector = vectorizer.transform([processed_message])
test_prediction = model.predict(test_vector)[0]
print(f"Test message prediction: {'SPAM' if test_prediction == 1 else 'HAM'}")# test_models.py

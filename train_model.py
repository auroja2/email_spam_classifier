import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Simple preprocessing function (same as in your app)
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

# Load and prepare dataset
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.rename(columns={'v1': 'target', 'v2': 'text'})
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')

# Map labels to numbers
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Preprocess the text
print("Preprocessing text...")
df['processed_text'] = df['text'].apply(simple_preprocess)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], 
    df['target'],
    test_size=0.2,
    random_state=42
)

# Create and fit the TF-IDF vectorizer
print("Training the vectorizer...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_vect = tfidf.fit_transform(X_train)  # This is where the fitting happens
X_test_vect = tfidf.transform(X_test)

# Train the model
print("Training the model...")
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model and vectorizer
print("Saving model and vectorizer...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete! Files saved as 'model.pkl' and 'vectorizer.pkl'")

# Test the saved model to verify it works
print("\nVerifying the saved model...")
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Test with a sample message
test_message = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
test_processed = simple_preprocess(test_message)
test_vector = loaded_vectorizer.transform([test_processed])
prediction = loaded_model.predict(test_vector)[0]
print(f"Test message classified as: {'SPAM' if prediction == 1 else 'HAM'}")

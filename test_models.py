# test_models.py
import pickle

print("Loading models...")
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Vectorizer type: {type(vectorizer)}")
print(f"Model type: {type(model)}")

# Test with a sample message
message = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
vector = vectorizer.transform([message.lower()])
prediction = model.predict(vector)[0]
print(f"Test message prediction: {'SPAM' if prediction == 1 else 'HAM'}")

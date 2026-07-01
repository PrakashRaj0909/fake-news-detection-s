import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Keep only important columns
data = data[["text", "label"]]

# Text cleaning function
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()  # remove extra spaces
    return text

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# Show sample data
print("Cleaned Data Sample:")
print(data.head())

print("\nTotal records:", len(data))

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check shapes
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Train model using Logistic Regression
model = LogisticRegression()

model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pickle

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
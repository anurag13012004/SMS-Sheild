import os
import urllib.request
import zipfile
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Constants
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
MODEL_DIR = "models"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset ready.")
    else:
        print("Dataset already exists.")

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization is handled by CountVectorizer natively, returning clean space-separated string
    return text

def train():
    print("Loading dataset...")
    # The dataset is tab-separated with no header
    df = pd.read_csv(DATA_FILE, sep='\t', names=['label', 'message'])
    
    print(f"Dataset loaded: {len(df)} records.")
    
    print("Preprocessing data...")
    df['clean_message'] = df['message'].apply(preprocess_text)
    
    X = df['clean_message']
    y = df['label'] # 'ham' or 'spam'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing...")
    # CountVectorizer tokenizes by default
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("Exporting model.pkl and vectorizer.pkl...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Successfully saved to '{MODEL_DIR}/' directory.")

if __name__ == "__main__":
    download_data()
    train()

import os
import pandas as pandas
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/sample_data.json")

def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data

def preprocess_data(json_file):
    data = load_data(json_file)

    dataFrame = pandas.DataFrame(data)

    # Here we're going to extract the data from the sample
    descriptions = dataFrame["description"].values
    labels = dataFrame["account_code"].values

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    vectoriser = TfidfVectorizer(stop_words="english", max_features=5000)

    X_tfidf = vectoriser.fit_transform(descriptions)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectoriser, label_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectoriser, label_encoder = preprocess_data(DATA_PATH)

    print("TF-IDF Preprocessing Complete!")
    print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
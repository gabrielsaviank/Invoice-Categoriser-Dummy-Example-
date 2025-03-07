import torch
import os
import numpy as np
from preprocess import preprocess_data, DATA_PATH
from model import InvoiceClassifier

_, _, _, _, vectoriser, label_encoder = preprocess_data(DATA_PATH)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/invoice_classifier.pth")

input_size = len(vectoriser.get_feature_names_out())
num_classes = len(label_encoder.classes_)

# Initialise model
model = InvoiceClassifier(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predict_category(description):
    description_tfidf = vectoriser.transform([description])
    description_tensor = torch.tensor(description_tfidf.toarray(), dtype=torch.float32)

    with torch.no_grad():
        prediction = model(description_tensor)

        predicted_class = torch.argmax(prediction, dim=1).item()

        account_code = label_encoder.inverse_transform([predicted_class])[0]

        return account_code

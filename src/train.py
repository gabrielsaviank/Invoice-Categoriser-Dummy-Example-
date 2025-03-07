import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os

from preprocess import preprocess_data, DATA_PATH
from model import InvoiceClassifier

X_train, X_test, y_train, y_test, vectoriser, label_encoder = preprocess_data(DATA_PATH)

X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = InvoiceClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimiser.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred, dim=1).numpy()
    y_test_np = y_test_tensor.numpy()
    accuracy = accuracy_score(y_test_np, y_pred_classes)

print(f"Model Accuracy: {accuracy:.4f}")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/invoice_classifier.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
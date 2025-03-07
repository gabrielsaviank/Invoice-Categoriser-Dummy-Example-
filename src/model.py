import torch
import torch.nn as nn

class InvoiceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(InvoiceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer
        self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)
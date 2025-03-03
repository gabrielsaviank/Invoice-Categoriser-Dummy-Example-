import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data, DATA_PATH

class InvoiceClassifier(nn.Module):

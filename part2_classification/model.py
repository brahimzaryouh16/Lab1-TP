# Partie 2 - modele : MLP PyTorch multiclasses avec une interface sklearn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42


def build_mlp(n_features, n_classes, hidden, dropout=0.0):
    layers = []
    size = n_features
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        size = h
    layers.append(nn.Linear(size, n_classes))
    return nn.Sequential(*layers)


def make_optimizer(model, name, lr, weight_decay=0.0):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """MLP multiclasses avec les methodes fit/predict/predict_proba de sklearn.
    C'est ce qui permet d'utiliser GridSearchCV avec un modele PyTorch."""

    def __init__(self, hidden=(64, 32), lr=0.001, optimizer="adam", epochs=30,
                 batch_size=512, dropout=0.0, weight_decay=0.0):
        self.hidden = hidden
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay

    def fit(self, X, y):
        torch.manual_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.int64), device=device)

        self.n_classes_ = int(y_tensor.max().item()) + 1
        self.classes_ = np.arange(self.n_classes_)
        self.model_ = build_mlp(X_tensor.shape[1], self.n_classes_,
                                tuple(self.hidden), self.dropout).to(device)
        optimizer = make_optimizer(self.model_, self.optimizer, self.lr, self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        device = next(self.model_.parameters()).device
        self.model_.eval()
        probabilities = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i:i + 8192], device=device)
                probabilities.append(torch.softmax(self.model_(xb), dim=1))
        return torch.cat(probabilities).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

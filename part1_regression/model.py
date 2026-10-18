# Partie 1 - modele : MLP PyTorch avec une interface sklearn (pour GridSearchCV)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin

SEED = 42


def build_mlp(n_features, hidden, dropout=0.0):
    layers = []
    size = n_features
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def make_optimizer(model, name, lr, weight_decay=0.0):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """MLP de regression avec les methodes fit/predict de sklearn.
    C'est ce qui permet d'utiliser GridSearchCV avec un modele PyTorch."""

    def __init__(self, hidden=(64, 32), lr=0.001, optimizer="adam", epochs=30,
                 batch_size=1024, dropout=0.0, weight_decay=0.0, l1=0.0):
        self.hidden = hidden
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.l1 = l1

    def fit(self, X, y):
        torch.manual_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32), device=device).view(-1, 1)

        self.model_ = build_mlp(X_tensor.shape[1], tuple(self.hidden), self.dropout).to(device)
        optimizer = make_optimizer(self.model_, self.optimizer, self.lr, self.weight_decay)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                if self.l1 > 0:
                    loss = loss + self.l1 * sum(p.abs().sum() for p in self.model_.parameters())
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
            predictions = self.model_(X_tensor).view(-1).cpu().numpy()
        return predictions

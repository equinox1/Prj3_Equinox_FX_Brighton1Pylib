import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class PyTorchTuner:
    def __init__(self, **kwargs):
        self.params = kwargs.get("hypermodel_params", {})
        self.train_data = kwargs.get("traindataset")
        self.val_data = kwargs.get("valdataset")
        self.input_shape = self.params.get("mltune", {}).get("data_input_shape", (24, 1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_model(self, trial):
        input_dim = np.prod(self.input_shape)
        n_units1 = trial.suggest_int('n_units1', 32, 256, step=32)
        n_units2 = trial.suggest_int('n_units2', 32, 256, step=32)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1)
        )
        return model.to(self.device)

    def prepare_data(self):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
        return train_loader, val_loader

    def objective(self, trial):
        model = self.build_model(trial)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_loader, val_loader = self.prepare_data()

        for epoch in range(5):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb).squeeze()
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = model(xb).squeeze()
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def run(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=10)
        print("Best trial:", study.best_trial.params)
        return study

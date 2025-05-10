import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

class PyTorchTuner:
    def __init__(self, **kwargs):
        self.params = kwargs.get("hypermodel_params", {})
        self.train_data = kwargs.get("traindataset")
        self.val_data = kwargs.get("valdataset")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_data(self):
        def extract_xy(data):
            X, y = [], []
            for x_i, y_i in data:
                x_np = np.asarray(x_i.numpy() if hasattr(x_i, 'numpy') else x_i)
                y_np = np.asarray(y_i.numpy() if hasattr(y_i, 'numpy') else y_i)
                X.append(x_np)
                y.append(y_np)
            most_common_shape = max(set([x.shape for x in X]), key=[x.shape for x in X].count)
            X_clean = [x for x in X if x.shape == most_common_shape]
            y_clean = [y[i] for i in range(len(X)) if X[i].shape == most_common_shape]
            return torch.tensor(np.stack(X_clean), dtype=torch.float32), torch.tensor(np.stack(y_clean), dtype=torch.float32)

        X_train, y_train = extract_xy(self.train_data)
        X_val, y_val = extract_xy(self.val_data)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    def build_model(self, trial, input_dim):
        n_units1 = trial.suggest_int('n_units1', 32, 128, step=32)
        n_units2 = trial.suggest_int('n_units2', 32, 128, step=32)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1)
        )
        return model.to(self.device)

    def objective(self, trial):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loader, val_loader = self.prepare_data()
        for xb, _ in train_loader:
            input_dim = xb[0].numel()
            break

        model = self.build_model(trial, input_dim)

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        try:
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
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[PyTorchTuner] ⚠️ Trial pruned due to CUDA OOM")
                raise optuna.exceptions.TrialPruned()
            raise e

    def run_search(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=10)
        print("Best trial:", study.best_trial.params)
        return study

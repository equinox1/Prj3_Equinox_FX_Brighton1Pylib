import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PyTorchTuner:
    def __init__(self, **kwargs):
        self.params = kwargs.get("hypermodel_params", {})
        self.train_data = kwargs.get("traindataset")
        self.val_data = kwargs.get("valdataset")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_model(self, trial):
        train_loader, _ = self.prepare_data()
        for xb, _ in train_loader:
            input_dim = xb[0].numel()
            break

        n_units1 = trial.suggest_int('n_units1', 32, 256, step=32)
        n_units2 = trial.suggest_int('n_units2', 32, 256, step=32)

        model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1)
        )
        return model.to(self.device)

    def prepare_data(self):
        def extract_xy(data, name):
            X_list, y_list = [], []
            for x, y in data:
                x_arr = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
                y_arr = np.asarray(y.numpy() if hasattr(y, "numpy") else y)
                X_list.append(x_arr)
                y_list.append(y_arr)

            shape_counts = {}
            for shape in [x.shape for x in X_list]:
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            most_common_shape = max(shape_counts, key=shape_counts.get)

            cleaned_X = [x for x in X_list if x.shape == most_common_shape]
            cleaned_y = [y for i, y in enumerate(y_list) if X_list[i].shape == most_common_shape]

            if not cleaned_X:
                raise ValueError(f"[{name}] No valid X samples found matching most common shape {most_common_shape}")

            return np.stack(cleaned_X), np.stack(cleaned_y)

        X_train, y_train = extract_xy(self.train_data, "train")
        X_val, y_val = extract_xy(self.val_data, "val")

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
                try:
                    preds = model(xb).squeeze()
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("[PyTorchTuner] \u26a0\ufe0f Trial failed due to CUDA OOM")
                        raise optuna.TrialPruned()
                    raise e

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = model(xb).squeeze()
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def run_search(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=10)
        print("Best trial:", study.best_trial.params)
        return study

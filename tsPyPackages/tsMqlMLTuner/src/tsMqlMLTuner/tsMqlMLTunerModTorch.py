import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PyTorchTuner:
    def __init__(self, **kwargs):
        self.params = kwargs.get("hypermodel_params", {})
        self.train_data = kwargs.get("traindataset")
        self.val_data = kwargs.get("valdataset")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.oracle = kwargs.get("oracle", OracleClient())

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

    def build_model(self, hp, input_dim):
        n_units1 = hp.get('n_units1', 64)
        n_units2 = hp.get('n_units2', 64)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1)
        )
        return model.to(self.device)

    def objective_from_hp(self, hp):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loader, val_loader = self.prepare_data()
        sample_batch = next(iter(train_loader))[0]
        print(f"[DEBUG] sample_batch shape before flatten: {sample_batch.shape}")
        flat_sample = sample_batch.view(sample_batch.size(0), -1)
        input_dim = flat_sample.shape[1]
        print(f"[DEBUG] input_dim to Linear: {input_dim}")

        model = self.build_model(hp, input_dim)
        optimizer_name = hp.get("optimizer", "Adam")
        lr = hp.get("lr", 1e-3)
        optimizer = getattr(optim, optimizer_name, optim.Adam)(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        try:
            for epoch in range(hp.get("epochs", 5)):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    xb = xb.view(xb.size(0), -1)
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
                    xb = xb.view(xb.size(0), -1)
                    preds = model(xb).squeeze()
                    loss = loss_fn(preds, yb)
                    val_losses.append(loss.item())

            return np.mean(val_losses)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[PyTorchTuner] ⚠️ Trial failed due to CUDA OOM")
            raise e

    def run_search(self):
        tuner_id = os.environ.get("TUNER_ID", "worker")
        print(f"[PyTorchTuner] 🚀 Starting distributed search as {tuner_id}")

        while True:
            try:
                trial = self.oracle.get_trial()
                if not trial or "trial_id" not in trial:
                    print("[PyTorchTuner] 💤 No more trials. Exiting.")
                    break

                trial_id = trial["trial_id"]
                hp = trial["hyperparameters"]

                print(f"[PyTorchTuner] 🔍 Running trial {trial_id} with hyperparameters: {hp}")
                val_loss = self.objective_from_hp(hp)

                print(f"[PyTorchTuner] ✅ Trial {trial_id} completed. val_loss={val_loss:.5f}")
                self.oracle.report_trial_result(trial_id, val_loss)

            except Exception as e:
                print(f"[PyTorchTuner] ❌ Error during trial execution: {e}")
                if "trial_id" in locals():
                    self.oracle.update_trial_status(trial_id, "FAILED")
                break

        print("[PyTorchTuner] 🧹 Distributed search complete.")
        return True

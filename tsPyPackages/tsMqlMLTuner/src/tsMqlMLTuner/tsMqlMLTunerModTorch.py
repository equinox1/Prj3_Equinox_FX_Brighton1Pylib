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
        model_type = hp.get('model_type', 'cnn').lower()

        if model_type == 'cnn':
            num_layers = hp.get('num_cnn_layers', 1)
            layers = [nn.Unflatten(1, (1, input_dim))]
            in_channels = 1
            length = input_dim
            for i in range(num_layers):
                out_channels = hp.get(f'cnn_filters_{i}', 64)
                kernel_size = hp.get(f'cnn_kernel_size_{i}', 3)
                activation = hp.get(f'cnn_activation_{i}', 'relu')
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
                layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
                in_channels = out_channels
                length = max(1, length - kernel_size + 1)
            layers += [
                nn.Flatten(),
                nn.Linear(in_channels * length, hp.get('dense_1_units', 64)),
                nn.ReLU(),
                nn.Linear(hp.get('dense_1_units', 64), 1)
            ]
            return nn.Sequential(*layers).to(self.device)

        elif model_type == 'lstm':
            num_layers = hp.get('num_lstm_layers', 1)
            hidden_size = hp.get('lstm_units_0', 64)
            return nn.Sequential(
                nn.Unflatten(1, (1, input_dim)),
                nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True),
                nn.Flatten(),
                nn.Linear(hidden_size, 1)
            ).to(self.device)

        elif model_type == 'gru':
            num_layers = hp.get('num_gru_layers', 1)
            hidden_size = hp.get('gru_units_0', 64)
            return nn.Sequential(
                nn.Unflatten(1, (1, input_dim)),
                nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True),
                nn.Flatten(),
                nn.Linear(hidden_size, 1)
            ).to(self.device)

        elif model_type == 'transformer':
            num_blocks = hp.get('num_transformer_blocks', 1)
            key_dim = hp.get('key_dim_0', 64)
            num_heads = hp.get('num_heads_0', 4)
            class TransformerModel(nn.Module):
                def __init__(self, input_dim, key_dim, num_heads, num_blocks, dense_units):
                    super().__init__()
                    self.embedding = nn.Linear(input_dim, key_dim)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=key_dim, nhead=num_heads, batch_first=True),
                        num_layers=num_blocks
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(key_dim, dense_units),
                        nn.ReLU(),
                        nn.Linear(dense_units, 1)
                    )

                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    x = x.mean(dim=1)
                    return self.fc(x)
            return TransformerModel(input_dim=1, key_dim=key_dim, num_heads=num_heads,
                                    num_blocks=num_blocks, dense_units=hp.get('dense_1_units', 64)).to(self.device)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def objective_from_hp(self, hp):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loader, val_loader = self.prepare_data()
        sample_batch = next(iter(train_loader))[0]
        input_shape = sample_batch.shape[1:]
        input_dim = int(np.prod(input_shape))
        model_type = hp.get('model_type', 'cnn').lower()

        print(f"[PyTorchTuner] Model type: {model_type}, input shape: {input_shape}, flat input_dim: {input_dim}")
        model = self.build_model(hp, input_dim)

        optimizer_name = hp.get("optimizer", "Adam").capitalize()
        lr = hp.get("learning_rate", 1e-3)
        optimizer = getattr(optim, optimizer_name, optim.Adam)(model.parameters(), lr=lr)
        loss_name = hp.get("loss", "mse").lower()
        loss_fn = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()

        try:
            for epoch in range(hp.get("epochs", 5)):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    if model_type in ["lstm", "gru", "transformer"]:
                        xb = xb.view(xb.size(0), 1, -1)
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
                    if model_type in ["lstm", "gru", "transformer"]:
                        xb = xb.view(xb.size(0), 1, -1)
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
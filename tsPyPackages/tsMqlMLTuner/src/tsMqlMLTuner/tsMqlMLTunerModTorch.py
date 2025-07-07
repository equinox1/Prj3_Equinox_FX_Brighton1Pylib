import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from types import SimpleNamespace
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
    # CMdtunerTorch for PyTorch-based distributed tuning
import logging
import os
import torch


logger = logging.getLogger(__name__)


import os # Import os to access environment variables
import time # Import time for sleep
import requests # ADDED: Import requests for handling connection errors

import pathlib
import uuid  # Ensure uuid is imported for use in get_callbacks

from tsMqlSetup import CMqlSetup
# Initialize CMqlSetup for the launcher itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
# num_cores: Estimate physical cores. On systems with hyperthreading, this is often
#            half the logical core count (os.cpu_count()). If os.cpu_count() is not available
#            or is 1, default to 1.
# num_threads: Typically 1 per core for numerical workloads to avoid hyperthreading
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)

# --- Global Configuration ---
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {})

# Extract backend for logging path - crucial for correct log file path
# This will be passed to initialize_logging. It can also be obtained from env if passed by launcher.
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch')) # Default to pytorch if not specified


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate=0.2, num_lstm_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_lstm_layers = num_lstm_layers

        # Ensure hidden_layer_sizes is a list/tuple for consistent iteration
        if not isinstance(self.hidden_layer_sizes, (list, tuple)):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]

        # LSTM layers
        lstm_layers = []
        for i in range(self.num_lstm_layers):
            input_dim = input_size if i == 0 else self.hidden_layer_sizes[i-1]
            output_dim = self.hidden_layer_sizes[i] if i < len(self.hidden_layer_sizes) else self.hidden_layer_sizes[-1]
            lstm_layers.append(nn.LSTM(input_dim, output_dim, batch_first=True))
        self.lstm_layers = nn.ModuleList(lstm_layers)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers (dense layers)
        dense_layers = []
        # Input to the first dense layer is the output of the last LSTM layer
        current_input_dim = self.hidden_layer_sizes[-1]

        # Add dense layers based on configuration (e.g., from hypermodel_params)
        # For simplicity, let's assume a fixed structure or infer from hypermodel_params
        # If no specific dense layers are defined, just go straight to output layer
        
        # Example: Add one dense layer if needed, before the final output layer
        # For now, we'll assume the output of the last LSTM directly feeds into the final output layer
        
        self.fc = nn.Linear(current_input_dim, output_size)

        logger.info(f"[LSTMModel] Initialized with input_size={input_size}, hidden_layer_sizes={hidden_layer_sizes}, output_size={output_size}, dropout_rate={dropout_rate}, num_lstm_layers={num_lstm_layers}")


    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x) # x is (batch_size, sequence_length, hidden_size)
            # If it's the last LSTM layer, we might only need the last hidden state
            # For sequence-to-sequence, keep all outputs. For sequence-to-one, take last.
            # Assuming sequence-to-one for prediction, take the last time step's output
            if i == len(self.lstm_layers) - 1:
                x = x[:, -1, :] # Take output of the last time step: (batch_size, hidden_size)
            x = self.dropout(x)

        x = self.fc(x) # (batch_size, output_size)
        return x



class CMdtunerTorch:
    def __init__(self, is_chief=False, tuner_id='default_tuner', oracle_client=None, oracle_url=None, **kwargs):
        self.is_chief = is_chief
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.oracle_url = oracle_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.train_dataset = kwargs.get("train_data")
        self.val_dataset = kwargs.get("val_data")
        self.model_dir = kwargs.get("model_dir", os.path.join("Logdir", "tsneuromodel_1", "pytorch_models"))

        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"[PyTorchTuner] Using device: {self.device}")
        logger.info(f"[PyTorchTuner] Model save directory: {self.model_dir}")

    def build_model(self, hp):
        import torch.nn as nn
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(60 * 4, 1)
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        model = SimpleModel().to(self.device)
        model.device = self.device  # Attach device for external access
        return model

    def load_data(self):
        # No-op: datasets are already provided
        pass

    def run(self):
        logger.info("[CMdtunerTorch] Chief run method triggered")

        while True:
            trial_data = self.oracle_client.get_trial(self.tuner_id)
            if trial_data is None:
                logger.info("🛑 No more trials to run. Exiting.")
                break

            trial = SimpleNamespace(**trial_data)

            try:
                logger.info(f"🧪 Trial ID: {trial.trial_id}")
                self.load_data()

                model = self.build_model(trial.hyperparameters)
                optimizer = torch.optim.Adam(model.parameters())
                loss_fn = torch.nn.MSELoss()

                # Convert DataLoader to full batch tensors
                X_train, Y_train = zip(*[(x, y) for x, y in self.train_dataset])
                X_val, Y_val = zip(*[(x, y) for x, y in self.val_dataset])

                import numpy as np
                X_train = torch.tensor(np.stack([x.numpy() for x in X_train]), dtype=torch.float32).to(self.device)
                Y_train = torch.tensor(np.stack([y.numpy() for y in Y_train]), dtype=torch.float32).to(self.device)
                X_val = torch.tensor(np.stack([x.numpy() for x in X_val]), dtype=torch.float32).to(self.device)
                Y_val = torch.tensor(np.stack([y.numpy() for y in Y_val]), dtype=torch.float32).to(self.device)

                for epoch in range(self.epochs):
                    model.train()
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = loss_fn(output, Y_train)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val)
                    val_loss = loss_fn(val_preds, Y_val).item()

                # Save trial-specific model
                trial_model_path = os.path.join(self.model_dir, f"best_model_{trial.trial_id}.pth")
                torch.save(model.state_dict(), trial_model_path)

                # Also save as global best_model.pth
                global_model_path = os.path.join(self.model_dir, "best_model.pth")
                torch.save(model.state_dict(), global_model_path)

                self.oracle_client.report_result(trial.trial_id, val_loss)
                logger.info(f"✅ Trial {trial.trial_id} completed with score: {val_loss}")

            except Exception as e:
                logger.exception(f"💥 Trial {trial.trial_id} failed: {e}")
                self.oracle_client.report_result(trial.trial_id, float("inf"))

    def get_best_model_path(self):
        return os.path.join(self.model_dir, "best_model.pth")

    def get_best_model(self):
        import torch.nn as nn
        model = self.build_model(None)
        best_model_path = self.get_best_model_path()
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            model.eval()
            return model
        else:
            logger.warning("Best model path does not exist. Returning None.")
            return None

    def get_model_dir(self):
        return self.model_dir

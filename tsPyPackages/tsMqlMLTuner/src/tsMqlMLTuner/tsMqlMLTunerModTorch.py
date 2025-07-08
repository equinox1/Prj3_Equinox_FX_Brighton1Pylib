#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlMLTunerModTorch.py
File: tsPyPackages/tsMqlMLTuner/tsMqlMLTunerModTorch.py
Description: PyTorch Tuner module for machine learning models.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.4.0
License: MIT License
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure these imports are correct based on your project structure
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # Ensure correct import path

logger = logging.getLogger(__name__)

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})
base_params = all_params.get("base", {})

_logical_cores = os.cpu_count() or 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel=app_params.get('LOGLEVEL', 'INFO'),
    warn='ignore',
    precision=app_params.get('TF_PRECISION', 'mixed_float16'), # This is for TF, might need a PyTorch specific one
    tfdebug=app_params.get('TFDEBUG', False),
    num_cores=_estimated_physical_cores,
    num_threads=1
)

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(tune_params.get('seed', 42))

class OracleSyncCallbackTorch:
    """
    A custom callback to synchronize trial status and results with the Oracle Server for PyTorch.
    """
    def __init__(self, trial_id, oracle_client, objective_name, direction, logger_instance=None):
        self.trial_id = trial_id
        self.oracle_client = oracle_client
        self.objective_name = objective_name
        self.direction = direction
        self.best_val_score = float('inf') if direction == 'min' else float('-inf')
        self.logger = logger_instance or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"OracleSyncCallbackTorch initialized for trial {self.trial_id}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self.objective_name)
        if current_score is not None:
            metrics = {k: float(v) for k, v in logs.items()} # Ensure metrics are serializable
            self.oracle_client.update_trial(
                trial_id=self.trial_id,
                metrics=metrics,
                step=epoch,
                status="RUNNING"
            )
            self.logger.debug(f"Trial {self.trial_id} epoch {epoch+1}: Reported metrics to Oracle.")
            
            if self.direction == 'min':
                if current_score < self.best_val_score:
                    self.best_val_score = current_score
            else: # max
                if current_score > self.best_val_score:
                    self.best_val_score = current_score
        else:
            self.logger.warning(f"Objective '{self.objective_name}' not found in logs for trial {self.trial_id} epoch {epoch+1}.")

    def on_train_end(self, logs=None):
        logs = logs or {}
        final_score = logs.get(self.objective_name)
        if final_score is None:
            final_score = self.best_val_score if self.best_val_score != float('inf') and self.best_val_score != float('-inf') else None
            if final_score is None:
                self.logger.error(f"Trial {self.trial_id}: Final objective '{self.objective_name}' not found in logs and no best score recorded. Reporting FAILED.")
                self.oracle_client.update_trial_status(self.trial_id, status="FAILED")
                return

        self.oracle_client.report_trial_result(self.trial_id, float(final_score), status="COMPLETED")
        self.logger.info(f"Trial {self.trial_id} finished. Final score: {final_score:.4f}. Status: COMPLETED.")


# --- PyTorch Models ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate, num_lstm_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_lstm_layers = num_lstm_layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_layer_sizes[0], batch_first=True))
        
        # Additional LSTM layers
        for i in range(1, num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_layer_sizes[-1], output_size) # Fully connected layer after LSTM
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logger.info(f"[LSTMModel] Initialized with input_size={input_size}, hidden_layer_sizes={hidden_layer_sizes}, output_size={output_size}, dropout_rate={dropout_rate}, num_lstm_layers={num_lstm_layers}")
        logger.info(f"[LSTMModel] Using device: {self.device}")

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == len(self.lstm_layers) - 1: # Last LSTM layer, no return_sequences
                x, _ = lstm_layer(x)
                # If batch_first=True, output shape is (batch_size, seq_len, hidden_size)
                # We want the output of the last time step for the final layer
                x = x[:, -1, :] # Get the last hidden state of the sequence
            else: # Intermediate LSTM layers, return sequences
                x, _ = lstm_layer(x)

        x = self.dropout(x)
        x = self.fc(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes, filters, kernel_size, dropout_rate):
        super(CNNModel, self).__init__()
        # input_shape is (sequence_length, num_features)
        self.conv1 = nn.Conv1d(input_shape[1], filters[0], kernel_size[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate output size after conv1 to determine input for fc layer
        # Formula for Conv1d output size: (W - F + 2P) / S + 1
        # W = input_shape[0] (sequence_length)
        # F = kernel_size[0]
        # P = 0 (padding)
        # S = 1 (stride)
        conv1_output_len = input_shape[0] - kernel_size[0] + 1
        
        self.fc = nn.Linear(filters[0] * conv1_output_len, num_classes)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logger.info(f"[CNNModel] Initialized with input_shape={input_shape}, num_classes={num_classes}, filters={filters}, kernel_size={kernel_size}, dropout_rate={dropout_rate}")
        logger.info(f"[CNNModel] Using device: {self.device}")

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        # Conv1d expects (batch_size, num_features, sequence_length)
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten for the fully connected layer
        x = self.fc(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate, num_gru_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_gru_layers = num_gru_layers
        self.gru_layers = nn.ModuleList()

        self.gru_layers.append(nn.GRU(input_size, hidden_layer_sizes[0], batch_first=True))
        for i in range(1, num_gru_layers):
            self.gru_layers.append(nn.GRU(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_layer_sizes[-1], output_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logger.info(f"[GRUModel] Initialized with input_size={input_size}, hidden_layer_sizes={hidden_layer_sizes}, output_size={output_size}, dropout_rate={dropout_rate}, num_gru_layers={num_gru_layers}")
        logger.info(f"[GRUModel] Using device: {self.device}")

    def forward(self, x):
        for i, gru_layer in enumerate(self.gru_layers):
            if i == len(self.gru_layers) - 1:
                x, _ = gru_layer(x)
                x = x[:, -1, :]
            else:
                x, _ = gru_layer(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout_rate):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logger.info(f"[TransformerModel] Initialized with input_size={input_size}, d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, dim_feedforward={dim_feedforward}, output_size={output_size}, dropout_rate={dropout_rate}")
        logger.info(f"[TransformerModel] Using device: {self.device}")

    def forward(self, src):
        # src shape: (batch_size, sequence_length, input_size)
        src = self.encoder(src) * np.sqrt(self.encoder.out_features) # Scale by sqrt(d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :]) # Take output from the last token
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- Tuner Class for PyTorch ---
class CMdtunerTorch:
    def __init__(self, tuner_id, oracle_client, is_chief, **kwargs):
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.is_chief = is_chief
        self.kwargs = kwargs.copy()

        self.input_shape = kwargs.get('input_shape') # (sequence_length, num_features)
        self.num_classes = kwargs.get('num_classes') # Output size
        self.train_data = kwargs.get('train_data') # DataLoader
        self.val_data = kwargs.get('val_data')     # DataLoader
        self.test_data = kwargs.get('test_data')   # DataLoader

        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        self.app_params = self.hypermodel_params.get('app', {})
        self.tune_params = self.hypermodel_params.get('mltune', {})

        self.objective_name = self.tune_params.get('objective', 'val_loss')
        self.objective_direction = 'min' # Assuming 'val_loss' is minimized

        self.max_epochs = self.tune_params.get('max_epochs', 10)
        self.batch_size = self.tune_params.get('batch_size', 32)
        self.num_trials = self.tune_params.get('num_trials', 50)
        self.executions_per_trial = self.tune_params.get('executions_per_trial', 1)

        self.model_save_dir = Path(kwargs.get("model_save_dir", Path(base_params.get('mp_glob_sub_ml_src_modeldata'))  / "saved_models"))
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[PyTorchTuner] Model save directory: {self.model_save_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[PyTorchTuner] Using device: {self.device}")

        # Best model tracking for chief
        self.best_model = None
        self.best_score = float('inf') if self.objective_direction == 'min' else float('-inf')

    def _build_model(self, hyperparameters):
        """
        Builds a PyTorch model based on the given hyperparameters.
        This method should be flexible enough to build different model types
        (LSTM, CNN, GRU, Transformer) based on the hyperparameters.
        """
        model_type = self.tune_params.get('ml_model_name', 'lstm').lower() # Default to lstm if not specified
        
        # Common hyperparameters
        input_size = self.input_shape[1] if len(self.input_shape) > 1 else self.input_shape[0]
        output_size = self.num_classes

        # Dynamic hyperparameters from the oracle trial
        # Use .get() with defaults for robustness
        # Ensure these match the names used in Keras Tuner's HyperParameters
        hidden_units = hyperparameters.get('units', self.tune_params.get('lstm_units_default', 64))
        dropout_rate = hyperparameters.get('dropout', self.tune_params.get('dropout', 0.2))
        num_lstm_layers = hyperparameters.get('num_lstm_layers', self.tune_params.get('lstm_layers_default', 1)) # Assuming a default for num_lstm_layers

        # For models that use lists of hidden layer sizes (e.g., multi-layer LSTMs)
        # If 'units' is a single value, create a list for hidden_layer_sizes
        hidden_layer_sizes = [hidden_units] * num_lstm_layers # Example: [64, 64] for 2 layers

        logger.info(f"Building model with hyperparameters: {hyperparameters}") # Added debug log

        if model_type == 'lstm':
            model = LSTMModel(input_size, hidden_layer_sizes, output_size, dropout_rate, num_lstm_layers)
        elif model_type == 'cnn':
            filters = hyperparameters.get('filters', self.tune_params.get('cnn_units_default', 64))
            kernel_size = hyperparameters.get('kernel_size', self.tune_params.get('cnn_kernel_size_default', 3))
            model = CNNModel(self.input_shape, output_size, [filters], [kernel_size], dropout_rate)
        elif model_type == 'gru':
            num_gru_layers = hyperparameters.get('num_gru_layers', self.tune_params.get('gru_layers_default', 1))
            model = GRUModel(input_size, hidden_layer_sizes, output_size, dropout_rate, num_gru_layers)
        elif model_type == 'transformer':
            d_model = hyperparameters.get('d_model', self.tune_params.get('trans_dim_default', 64))
            nhead = hyperparameters.get('nhead', self.tune_params.get('trans_heads_default', 4))
            num_encoder_layers = hyperparameters.get('num_encoder_layers', self.tune_params.get('trans_layers_default', 2))
            dim_feedforward = hyperparameters.get('dim_feedforward', self.tune_params.get('trans_ff_default', 256))
            model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout_rate)
        else:
            logger.warning(f"Configured model type '{model_type}' is unsupported. Falling back to 'lstm' model.")
            model = LSTMModel(input_size, hidden_layer_sizes, output_size, dropout_rate, num_lstm_layers)
        
        return model.to(self.device)

    def _train_model(self, model, train_loader, val_loader, hyperparameters, trial_id):
        """
        Trains the given PyTorch model.
        """
        optimizer_choice = hyperparameters.get('optimizer', self.tune_params.get('optimizer', 'adam'))
        learning_rate = hyperparameters.get('learning_rate', self.tune_params.get('learning_rate', 1e-3))
        
        if optimizer_choice == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_choice == 'nadam':
            optimizer = optim.Nadam(model.parameters(), lr=learning_rate)
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_choice}. Defaulting to Adam.")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_name = self.app_params.get("mp_app_loss_function", "mean_squared_error")
        if loss_name == 'mean_squared_error':
            criterion = nn.MSELoss()
        elif loss_name == 'mean_absolute_error':
            criterion = nn.L1Loss()
        elif loss_name == 'binary_crossentropy':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_name == 'categorical_crossentropy':
            criterion = nn.CrossEntropyLoss()
        else:
            logger.warning(f"Unsupported loss function: {loss_name}. Defaulting to MSELoss.")
            criterion = nn.MSELoss()

        callbacks = [
            OracleSyncCallbackTorch(
                trial_id=trial_id,
                oracle_client=self.oracle_client,
                objective_name=self.objective_name,
                direction=self.objective_direction,
                logger_instance=logger
            )
        ]

        model.train()
        for epoch in range(self.max_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Reshape labels if necessary (e.g., (batch_size,) to (batch_size, 1))
                if labels.dim() == 1 and self.num_classes == 1:
                    labels = labels.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if labels.dim() == 1 and self.num_classes == 1:
                        labels = labels.unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            model.train() # Set back to train mode

            logs = {'loss': epoch_loss, 'val_loss': val_loss}
            logger.info(f"Trial {trial_id} - Epoch {epoch+1}/{self.max_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, logs)
        
        for callback in callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs)

        return val_loss # Return the final validation loss as the trial result

    def run(self):
        """
        Main method to run the tuning process.
        Handles both chief and worker logic.
        """
        if self.is_chief:
            self._run_chief()
        else:
            self._run_worker()

    def _run_chief(self):
        logger.info("🚀 Chief starting distributed tuning...")
        # Chief's responsibility: orchestrate trials, manage Oracle
        for i in range(self.num_trials):
            logger.info(f"[CMdtunerTorch] Chief requesting trial {i+1}...")
            # CORRECTED: Use get_trial method
            trial_data = self.oracle_client.get_trial(self.tuner_id)

            if trial_data and trial_data.get('trial_id'):
                trial_id = trial_data['trial_id']
                hyperparameters = trial_data['hyperparameters']
                logger.info(f"[CMdtunerTorch] Chief received trial {trial_id} with hyperparameters: {hyperparameters}")

                # Chief does not train, it just manages the Oracle.
                # The actual training is done by workers.
                # Chief will wait for workers to report results.
                # For a simple chief, it might just continuously request trials
                # and rely on workers to update the Oracle.
                # In a more complex setup, chief might monitor worker status.
                # For now, we just log that a trial was requested.
                pass # Chief just requests trials and waits for results via Oracle
            else:
                logger.info("[CMdtunerTorch] Chief received no new trial. All trials might be completed or no idle trials.")
                break # Exit loop if no new trials are available

        logger.info("✅ Chief finished tuning")

    def _run_worker(self):
        logger.info("👷 Worker starting trial execution loop...")
        while True:
            # Worker requests a trial from the Oracle
            # CORRECTED: Use get_trial method
            trial_data = self.oracle_client.get_trial(self.tuner_id)

            if trial_data and trial_data.get('trial_id'):
                trial_id = trial_data['trial_id']
                hyperparameters = trial_data['hyperparameters']
                logger.info(f"[CMdtunerTorch] Worker {self.tuner_id} received trial {trial_id} with hyperparameters: {hyperparameters}")

                try:
                    # Build and train model for the received trial
                    model = self._build_model(hyperparameters)
                    final_val_loss = self._train_model(model, self.train_data, self.val_data, hyperparameters, trial_id)
                    
                    # Report final result to Oracle
                    self.oracle_client.report_trial_result(trial_id, final_val_loss, status="COMPLETED")
                    logger.info(f"[CMdtunerTorch] Worker {self.tuner_id} completed trial {trial_id}. Reported result: {final_val_loss:.4f}")

                except Exception as e:
                    logger.error(f"[CMdtunerTorch] Worker {self.tuner_id} encountered error during trial {trial_id}: {e}", exc_info=True)
                    self.oracle_client.update_trial_status(trial_id, status="FAILED")
                    # Continue to next iteration to request another trial
            else:
                logger.info(f"[CMdtunerTorch] Worker {self.tuner_id} received no new trial from Oracle. Assuming all trials are processed or no more available.")
                break # Exit loop if no new trials are available

        logger.info("✅ Worker finished trial execution.")

    def get_best_model(self):
        """
        Retrieves the best model found during the tuning process.
        For PyTorch, this means loading the state_dict into a newly instantiated model
        that matches the best trial's hyperparameters.
        """
        logger.info("Attempting to retrieve best model from Oracle.")
        best_trial_info = self.oracle_client.get_best_trial()

        if not best_trial_info:
            logger.warning("No best trial found from Oracle. Cannot retrieve best model.")
            return None

        best_trial_id = best_trial_info.get('trial_id')
        best_hyperparameters = best_trial_info.get('hyperparameters', {})
        logger.debug(f"Best trial hyperparameters from Oracle for loading model: {best_hyperparameters}") # Debug log for hyperparameters
        best_model_path = self.model_save_dir / f"best_model_{best_trial_id}.pth" # Assuming this naming convention

        if not best_model_path.exists():
            logger.error(f"❌ Best model checkpoint not found at {best_model_path}. This might indicate an issue with saving or a worker not completing.")
            return None

        try:
            # Dynamically build the model using the best trial's hyperparameters
            model = self._build_model(best_hyperparameters)
            
            # Load the state dictionary
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            model.eval() # Set to evaluation mode
            logger.info(f"✅ Best model for trial {best_trial_id} loaded successfully from {best_model_path}.")
            self.best_model = model
            return self.best_model
        except Exception as e:
            logger.error(f"❌ Error loading best model from {best_model_path}: {e}", exc_info=True)
            return None

    def get_model_dir(self):
        return self.model_save_dir

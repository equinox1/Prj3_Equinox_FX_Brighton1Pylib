import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
# CORRECTED: Changed to relative import for OracleClient
from .tsMqlMLOracleClient import OracleClient

import logging # Ensure logging is imported
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

logger = logging.getLogger(__name__)

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


class PyTorchTuner:
    def __init__(self, oracle_client, train_dataset, val_dataset, input_shape, num_classes, hypermodel_params, max_trials, project_name, log_dir, is_chief, oracle_url, overwrite, tuner_id): # ADDED tuner_id to init
        self.oracle_client = oracle_client
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hypermodel_params = hypermodel_params
        self.max_trials = max_trials
        self.project_name = project_name
        self.log_dir = log_dir
        self.is_chief = is_chief
        self.oracle_url = oracle_url
        self.overwrite = overwrite
        self.tuner_id = tuner_id # Store tuner_id as an instance attribute

        self.objective_name = self.hypermodel_params.get('mltune', {}).get('objective', 'val_loss')
        self.objective_direction = self.hypermodel_params.get('mltune', {}).get('objective_direction', 'min')
        self.epochs = self.hypermodel_params.get('mltune', {}).get('max_epochs', 50)
        self.batch_size = self.hypermodel_params.get('mltune', {}).get('batch_size', 32)
        self.seed = self.hypermodel_params.get('mltune', {}).get('seed', 42)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[PyTorchTuner] Using device: {self.device}")

        # Directory for saving models
        self.model_save_dir = pathlib.Path(self.log_dir) / self.project_name / "pytorch_models"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[PyTorchTuner] Model save directory: {self.model_save_dir}")

        self.best_model_path = None
        self.best_score = float('inf') if self.objective_direction == 'min' else float('-inf')

    def build_model(self, hp):
        # hp is a dictionary of hyperparameters for PyTorch
        # It comes from the OracleClient's get_trial response
        lstm_units = hp.get('lstm_units_0', 64) # Default value if not found
        num_lstm_layers = hp.get('num_lstm_layers', 1)
        dropout_rate = hp.get('dropout_rate', 0.2)
        
        # Ensure hidden_layer_sizes matches num_lstm_layers length
        hidden_layer_sizes = [lstm_units] * num_lstm_layers # Simple approach: all LSTM layers have same units

        model = LSTMModel(
            input_size=self.input_shape[-1], # Assuming last dim is features
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=self.num_classes,
            dropout_rate=dropout_rate,
            num_lstm_layers=num_lstm_layers
        )
        model.to(self.device)
        logger.info(f"[PyTorchTuner] Model built with hp: {hp}")
        return model

    def fit(self, X_train, Y_train, X_val, Y_val, epochs, batch_size):
        logger.info(f"[PyTorchTuner] PyTorch Tuner fit method started. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Worker loop for PyTorch
        retry_interval = 5  # seconds
        max_retries = 10
        retries = 0

        while True:
            try:
                # Check if Oracle is available by attempting to get a trial.
                # If it's the chief, it might be running locally without OracleClient,
                # or it might also be using OracleClient to coordinate with an external OracleServer.
                # The condition `if not self.is_chief` ensures workers always try to connect.
                # For the chief, if self.oracle_client is None (meaning it's a local chief not using OracleServer),
                # it proceeds with its own trial generation logic.
                if self.oracle_client: # Only attempt if oracle_client is initialized
                    # Attempt to get a trial. ConnectionError will be caught if server is down.
                    trial_response = self.oracle_client.get_trial(self.tuner_id)
                else: # Chief not using OracleClient, simulate a single trial or use predefined HPs
                    logger.info("Chief is running without OracleClient. Simulating single trial.")
                    # In a real chief scenario without OracleClient, you'd define how it gets HPs
                    # For now, we'll break after one iteration if no OracleClient.
                    trial_response = None # No trial from Oracle if no client

                if trial_response is None or trial_response.get("trial_id") is None:
                    logger.info("No more trials from Oracle or Oracle is done. Exiting tuner.")
                    break # Exit if no trials left or Oracle signals completion

                trial_id = trial_response.get('trial_id')
                hyperparameters = trial_response.get('hyperparameters')
                status = trial_response.get('status')

                if status == 'STOPPED':
                    logger.info(f"Trial {trial_id} was stopped by Oracle. Skipping.")
                    continue

                if not hyperparameters:
                    logger.warning(f"Received trial {trial_id} with no hyperparameters. Skipping.")
                    if self.oracle_client:
                        self.oracle_client.update_trial_status(trial_id, status="INVALID")
                    continue

                logger.info(f"[PyTorchTuner] {self.tuner_id} running trial: {trial_id} with HPs: {hyperparameters}")
                if self.oracle_client:
                    self.oracle_client.update_trial_status(trial_id, status="RUNNING")

                model = self.build_model(hyperparameters)

                # Optimizer choice
                optimizer_choice = hyperparameters.get('optimizer', 'adam')
                learning_rate = hyperparameters.get('learning_rate', 1e-3)
                if optimizer_choice == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                elif optimizer_choice == 'nadam':
                    optimizer = optim.Nadam(model.parameters(), lr=learning_rate)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Default to Adam
                    logger.warning(f"Unsupported optimizer '{optimizer_choice}'. Defaulting to Adam.")

                # Loss function choice
                loss_name = self.hypermodel_params.get("app", {}).get("mp_app_loss_function", "mean_squared_error")
                if loss_name == "mean_absolute_error":
                    criterion = nn.L1Loss()
                elif loss_name == "categorical_crossentropy":
                    criterion = nn.CrossEntropyLoss()
                elif loss_name == "binary_crossentropy":
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.MSELoss() # Default
                    logger.warning(f"Unsupported loss function '{loss_name}'. Defaulting to MSELoss.")

                best_val_loss = float('inf')
                epochs_no_improve = 0
                early_stopping_patience = hyperparameters.get('es_patience', 10) # From tune_params or default

                for epoch in range(epochs):
                    model.train()
                    train_loss = 0.0
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * inputs.size(0)
                    
                    train_loss /= len(train_loader.dataset)

                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item() * inputs.size(0)
                    
                    val_loss /= len(val_loader.dataset)

                    logger.info(f"[PyTorchTuner] Trial {trial_id}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                    # Report intermediate results to Oracle (if not chief or if chief is also reporting)
                    if self.oracle_client:
                        self.oracle_client.report_trial_result(trial_id, {
                            self.objective_name: val_loss,
                            'epoch': epoch + 1,
                            'train_loss': train_loss
                        })

                    # Early stopping
                    if self.objective_direction == 'min':
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                            # Save best model weights
                            self.best_model_path = self.model_save_dir / f"{trial_id}_best_model.pth"
                            torch.save(model.state_dict(), self.best_model_path)
                            logger.info(f"Saved best model for trial {trial_id} to {self.best_model_path}")
                        else:
                            epochs_no_improve += 1
                    # Add max objective direction logic as well
                    elif self.objective_direction == 'max':
                        if val_loss > self.best_score: # Assuming val_loss is the objective, and we want to maximize it
                            self.best_score = val_loss
                            epochs_no_improve = 0
                            self.best_model_path = self.model_save_dir / f"{trial_id}_best_model.pth"
                            torch.save(model.state_dict(), self.best_model_path)
                            logger.info(f"Saved best model for trial {trial_id} to {self.best_model_path}")
                        else:
                            epochs_no_improve += 1


                    if epochs_no_improve >= early_stopping_patience:
                        logger.info(f"Early stopping triggered for trial {trial_id} at epoch {epoch+1}.")
                        if self.oracle_client:
                            self.oracle_client.update_trial_status(trial_id, status="STOPPED")
                        break
                
                # After training loop, report final status
                if self.oracle_client:
                    self.oracle_client.update_trial_status(trial_id, status="COMPLETED")
                logger.info(f"[PyTorchTuner] Trial {trial_id} completed.")

            except requests.exceptions.ConnectionError as ce:
                logger.error(f"Connection to Oracle server failed: {ce}. Retrying in {retry_interval}s...")
                time.sleep(retry_interval)
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max connection retries ({max_retries}) reached. Exiting tuner.")
                    break
            except Exception as e:
                logger.error(f"❌ An unexpected error occurred in PyTorchTuner.fit: {e}", exc_info=True)
                if 'trial_id' in locals() and trial_id and self.oracle_client:
                    self.oracle_client.update_trial_status(trial_id, status="FAILED")
                break # Exit tuner on unhandled error


    def get_best_model(self):
        # This method should load the best model found across all trials
        # For PyTorch, the best model path is updated during training
        if self.best_model_path and self.best_model_path.exists():
            logger.info(f"Attempting to load best PyTorch model from {self.best_model_path}")
            try:
                # Need to rebuild the model architecture first, then load state_dict
                # This requires knowing the hyperparameters of the best model.
                # If using Oracle, we can fetch the best trial's HPs.
                hp_values_for_loading = {
                    'lstm_units_0': 64, # Example default
                    'num_lstm_layers': 1, # Example default
                    'dropout_rate': 0.2, # Example default
                }
                # Attempt to get the actual best hyperparameters if available (e.g., from OracleClient)
                if self.oracle_client: # Use oracle_client instead of self.oracle
                    best_trial_info = self.oracle_client.get_best_trial()
                    if best_trial_info and 'hyperparameters' in best_trial_info:
                        hp_values_for_loading = best_trial_info['hyperparameters']
                        logger.info(f"Retrieved best hyperparameters for model loading: {hp_values_for_loading}")
                    else:
                        logger.warning("Could not retrieve best hyperparameters from Oracle. Using default/placeholder values for model loading.")


                model = self.build_model(hp_values_for_loading)
                model.load_state_dict(torch.load(self.best_model_path))
                model.eval() # Set to evaluation mode
                logger.info(f"PyTorch model loaded successfully from {self.best_model_path}")
                # You might want to print model summary here if PyTorch has a similar concept
                return model
            except Exception as e:
                logger.error(f"Error loading PyTorch model: {e}")
                return None
        else:
            logger.info(f"PyTorch best model file does not exist at {self.best_model_path}")
            return None

    def export_model(self, model, ftype="onnx"):
        if model is None:
            logger.warning("[PyTorchTuner] No model provided for export.")
            return None

        if ftype == "pth":
            if self.best_model_path:
                logger.info(f"[PyTorchTuner] Model already saved as .pth at {self.best_model_path}")
                return self.best_model_path
            else:
                # If not already saved as best model, save the current model
                save_path = self.model_save_dir / f"exported_model_{uuid.uuid4().hex}.pth"
                torch.save(model.state_dict(), save_path)
                logger.info(f"[PyTorchTuner] Model exported as .pth to {save_path}")
                return save_path
        elif ftype == "onnx":
            # ONNX export requires dummy input
            # Assuming input_shape is (sequence_length, num_features) or (batch_size, sequence_length, num_features)
            # For ONNX export, we need a dummy input with batch dimension
            if len(self.input_shape) == 2:
                dummy_input = torch.randn(1, self.input_shape[0], self.input_shape[1]).to(self.device)
            elif len(self.input_shape) == 1: # If input_shape is just (num_features,)
                 dummy_input = torch.randn(1, 1, self.input_shape[0]).to(self.device) # Assume sequence length of 1
            else:
                logger.error(f"[PyTorchTuner] Unsupported input_shape for ONNX export: {self.input_shape}")
                return None

            onnx_path = self.model_save_dir / f"{self.project_name}_model.onnx"
            try:
                torch.onnx.export(model,
                                  dummy_input,
                                  onnx_path,
                                  export_params=True,
                                  opset_version=11, # Common opset version
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output'],
                                  dynamic_axes={'input': {0: 'batch_size'}, # Allow dynamic batch size
                                                'output': {0: 'batch_size'}})
                logger.info(f"[PyTorchTuner] Model exported to ONNX: {onnx_path}")
                return onnx_path
            except Exception as e:
                logger.error(f"❌ Error exporting PyTorch model to ONNX: {e}", exc_info=True)
                return None
        else:
            logger.warning(f"[PyTorchTuner] Unsupported export file type for PyTorch: {ftype}. Only 'pth' and 'onnx' are supported.")
        return None

    def evaluate_model(self, model, test_data, test_labels):
        """Evaluates a given PyTorch model on test data."""
        if model is None or test_data is None or test_labels is None or test_data.size == 0:
            logger.warning("[PyTorchTuner] Model or test data/labels are None or empty. Cannot evaluate.")
            return {'val_loss': float('inf')}

        model.to(self.device)
        model.eval() # Set to evaluation mode

        test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.MSELoss() # Assuming MSE for evaluation, consistent with default loss
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
        
        avg_loss = total_loss / len(test_loader.dataset)
        logger.info(f"[PyTorchTuner] Evaluation Results - Loss: {avg_loss:.4f}")
        
        # You can add more metrics here if needed (e.g., MAE, R2_score)
        # For regression, MAE is often useful
        # For classification, accuracy, precision, recall, f1-score would be relevant
        
        return {'val_loss': avg_loss}
# filename: tsMqlMLTunerModTorch.py
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
from tsMqlSetup import CMqlSetup # Keep this import if CMqlSetup is used elsewhere
from tsMqlOverrides import CMqlOverrides
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient # Ensure correct import path
from tsMqlLogService import CMLogServiceSetup # Import the centralized logging setup

logger = logging.getLogger(__name__)

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})
base_params = all_params.get("base", {})

_logical_cores = os.cpu_count() or 1
_estimated_physical_cores = _logical_cores // 2 # Estimate physical cores
# Set default values for parameters if not found in tune_params
_default_batch_size = tune_params.get('batch_size', 32)
_default_max_epochs = tune_params.get('max_epochs', 100)
_default_objective_metric = tune_params.get('objective', 'val_loss') # Default objective

class OracleSyncCallback:
    """
    A custom callback for PyTorch training loops to synchronize with the OracleServer.
    Reports trial results (score and status) to the Oracle.
    """
    def __init__(self, oracle_client: OracleClient, trial_id: str, objective_metric: str, model_save_dir: Path):
        self.oracle_client = oracle_client
        self.trial_id = trial_id
        self.objective_metric = objective_metric
        self.best_score = float('inf') # Assuming lower is better (e.g., loss)
        self.model_save_dir = model_save_dir
        self.model_save_dir.mkdir(parents=True, exist_ok=True) # Ensure save directory exists
        logger.info(f"[OracleSyncCallback] Initialized for trial {self.trial_id}. Model save dir: {self.model_save_dir}")

    def on_epoch_end(self, epoch, logs=None, model=None):
        logs = logs or {}
        current_score = logs.get(self.objective_metric)
        
        if current_score is not None:
            # Report intermediate results to the OracleServer (optional, can be noisy)
            # self.oracle_client.report_trial_result(self.trial_id, current_score, status="RUNNING")

            # Save best model weights
            if current_score < self.best_score:
                self.best_score = current_score
                # Save the model state dictionary using a naming convention that includes trial_id
                save_path = self.model_save_dir / f"best_model_{self.trial_id}.pth"
                # It's common to save just the state_dict for PyTorch models
                if model: # Ensure model is passed
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"[OracleSyncCallback] Saved best model weights for trial {self.trial_id} at epoch {epoch} to {save_path} with {self.objective_metric}: {current_score:.4f}")
                else:
                    logger.warning(f"[OracleSyncCallback] Model not provided to save for trial {self.trial_id} at epoch {epoch}.")
        else:
            logger.warning(f"[OracleSyncCallback] Objective metric '{self.objective_metric}' not found in logs for trial {self.trial_id} at epoch {epoch}.")

    def on_train_end(self, logs=None):
        logs = logs or {}
        final_score = logs.get(self.objective_metric, self.best_score) # Use best score found if available
        status = "COMPLETED"
        if final_score is None:
            logger.warning(f"[OracleSyncCallback] Final objective metric '{self.objective_metric}' not found for trial {self.trial_id}. Reporting as FAILED or INCOMPLETE.")
            status = "FAILED" # Or "INCOMPLETE"
            final_score = -1.0 # Indicate an invalid score

        self.oracle_client.report_trial_result(self.trial_id, final_score, status=status)
        logger.info(f"[OracleSyncCallback] Reported final result for trial {self.trial_id} with score {final_score} and status {status} to OracleServer.")


class CMdtunerTorch:
    def __init__(self, tuner_id: str, oracle_client: OracleClient, is_chief: bool,
                 train_data: tuple, val_data: tuple, input_shape: tuple,
                 model_save_dir: Path, oracle_directory: Path, **kwargs):
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.is_chief = is_chief
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape # This is (timesteps, features) for sequence data
        self.model_save_dir = model_save_dir
        self.oracle_directory = oracle_directory
        self.kwargs = kwargs

        self.max_epochs = kwargs.get('tune_params', {}).get('max_epochs', _default_max_epochs)
        self.num_trials = kwargs.get('tune_params', {}).get('num_trials', 50)
        self.objective_metric = kwargs.get('tune_params', {}).get('objective', _default_objective_metric)
        self.batch_size = kwargs.get('tune_params', {}).get('batch_size', _default_batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[CMdtunerTorch] Using device: {self.device}")

        # Initialize best_model to None
        self.best_model = None

    def _build_model(self, hp):
        """
        Builds a PyTorch model based on hyperparameters.
        Handles both flat (2D) and sequence (3D) input shapes by flattening if necessary.
        """
        # Determine the input dimension for the first Linear layer
        # If input_shape is (timesteps, features_per_timestep), flatten to (timesteps * features_per_timestep)
        if len(self.input_shape) == 2: # This implies (timesteps, features_per_timestep)
            input_dim = self.input_shape[0] * self.input_shape[1]
            logger.info(f"Model input is 3D sequence data. Flattening input to {input_dim} features.")
            # Add a Flatten layer as the first module
            model = nn.Sequential(nn.Flatten())
        else: # Assume 2D input (features,)
            input_dim = self.input_shape[0]
            logger.info(f"Model input is 2D flat data with {input_dim} features.")
            model = nn.Sequential() # No initial Flatten layer needed

        # Hyperparameters for the model
        num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        activation_name = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

        def get_activation_layer(name):
            if name == 'relu':
                return nn.ReLU()
            elif name == 'tanh':
                return nn.Tanh()
            elif name == 'sigmoid':
                return nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation: {name}")

        # Add layers to the sequential model
        for i in range(num_layers):
            model.add_module(f"dense_{i}", nn.Linear(input_dim if i == 0 else units, units))
            model.add_module(f"activation_{i}", get_activation_layer(activation_name))
            if dropout_rate > 0:
                model.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))

        # Output layer
        model.add_module("output_layer", nn.Linear(units, 1)) # Assuming regression with single output

        return model.to(self.device)

    def _train_model(self, model, train_loader, val_loader, hp, trial_id):
        """
        Trains the PyTorch model for a given trial.
        """
        optimizer_name = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        criterion = nn.MSELoss() # Mean Squared Error for regression

        callback = OracleSyncCallback(self.oracle_client, trial_id, self.objective_metric, self.model_save_dir)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10 # Example patience

        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            logs = {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
            logger.info(f"[CMdtunerTorch] Trial {trial_id}, Epoch {epoch+1}/{self.max_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Call callback for epoch end
            callback.on_epoch_end(epoch, logs, model)

            # Early stopping logic (using val_loss as objective)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"[CMdtunerTorch] Early stopping triggered for trial {trial_id} at epoch {epoch+1}.")
                    break
        
        callback.on_train_end(logs={'val_loss': best_val_loss}) # Report final best validation loss
        return best_val_loss

    def run(self):
        """
        Main method to run the tuning process.
        """
        logger.info(f"[CMdtunerTorch] Tuner chief starting run loop. Is Chief: {self.is_chief}")

        # Convert numpy arrays to PyTorch tensors and create DataLoaders
        train_tensor_x = torch.tensor(self.train_data[0], dtype=torch.float32)
        train_tensor_y = torch.tensor(self.train_data[1], dtype=torch.float32)
        val_tensor_x = torch.tensor(self.val_data[0], dtype=torch.float32)
        val_tensor_y = torch.tensor(self.val_data[1], dtype=torch.float32)

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        val_dataset = TensorDataset(val_tensor_x, val_tensor_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # In a real distributed setting, workers would request trials from the Oracle.
        # For this setup, the chief acts as both the orchestrator and a worker.
        # The get_trial method now returns the full trial dictionary.
        trial_info = self.oracle_client.get_trial(self.tuner_id)
        if not trial_info:
            logger.error(f"[CMdtunerTorch] Failed to get trial from Oracle for tuner {self.tuner_id}.")
            return

        trial_id = trial_info.get('trial_id')
        trial_hyperparameters = trial_info.get('hyperparameters')

        if not trial_id or not trial_hyperparameters:
            logger.error(f"[CMdtunerTorch] Incomplete trial info received from Oracle for tuner {self.tuner_id}.")
            self.oracle_client.update_trial_status(trial_id, "FAILED")
            return

        logger.info(f"Received trial {trial_id} from OracleServer.")
        
        # Create a HyperParameters object from the dictionary
        from keras_tuner.engine.hyperparameters import HyperParameters
        hp = HyperParameters()
        for k, v in trial_hyperparameters.items(): # Use trial_hyperparameters directly
            if isinstance(v, int):
                hp.Int(k, min_value=v, max_value=v) # Fixed value
            elif isinstance(v, float):
                hp.Float(k, min_value=v, max_value=v) # Fixed value
            else:
                hp.Choice(k, [v]) # Fixed value

        logger.info(f"[CMdtunerTorch] Tuner chief received trial {trial_id} with HP: {hp.values}")

        try:
            model = self._build_model(hp)
            val_score = self._train_model(model, train_loader, val_loader, hp, trial_id)
            logger.info(f"[CMdtunerTorch] Worker chief completed trial {trial_id} with score {val_score}.")
            self.oracle_client.report_trial_result(trial_id, val_score, status="COMPLETED")
            self.best_model = model # Store the last trained model (which should be the best one for this trial)
            logger.info("✅ Worker finished trial execution for PyTorch.")
        except Exception as e:
            logger.error(f"[CMdtunerTorch] Worker chief encountered error during trial {trial_id}: {e}", exc_info=True)
            self.oracle_client.update_trial_status(trial_id, "FAILED")
            # Re-request the trial to ensure it's marked as failed and potentially retried or skipped
            # This is a simplified retry mechanism; a robust one would be in the launcher.
            # For now, just mark as failed and let the launcher decide.

    def get_best_model(self):
        """
        Retrieves the best model from the Oracle and loads its weights.
        """
        logger.info("[CMdtunerTorch] chief attempting to get best trial info from Oracle.")
        best_trial_info = self.oracle_client.get_best_trial()

        if not best_trial_info:
            logger.warning("No best trial found in Oracle. Cannot load best model.")
            return None

        best_trial_id = best_trial_info.get('trial_id')
        best_hyperparameters = best_trial_info.get('hyperparameters')
        
        if not best_trial_id or not best_hyperparameters:
            logger.warning("Best trial info is incomplete. Cannot load best model.")
            return None

        # Rebuild the model architecture using the best hyperparameters
        from keras_tuner.engine.hyperparameters import HyperParameters
        hp = HyperParameters()
        for k, v in best_hyperparameters.items():
            if isinstance(v, int):
                hp.Int(k, min_value=v, max_value=v)
            elif isinstance(v, float):
                hp.Float(k, min_value=v, max_value=v)
            else:
                hp.Choice(k, [v])

        model = self._build_model(hp)

        # Construct the path to the best model's weights
        best_model_path = self.model_save_dir / f"best_model_{best_trial_id}.pth" # Assuming this naming convention

        if not best_model_path.exists():
            logger.error(f"❌ Best model checkpoint not found at {best_model_path}. This might indicate an issue with saving or a worker not completing.")
            return None

        try:
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

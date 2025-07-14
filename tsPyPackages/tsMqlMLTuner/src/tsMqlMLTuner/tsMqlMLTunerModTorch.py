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
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

# Initialize logging for this module. This ensures logs from this module
# are correctly routed by the centralized logging system.
log_level = app_params.get('LOGLEVEL', 'INFO').upper()
CMLogServiceSetup.initialize_logging(
    app_params=app_params,
    tune_params=tune_params,
    base_params=base_params,
    role_hint='CMdtunerTorch_module', # A specific role hint for this module
    loglevel=log_level
)


# Corrected: Get LOGDIR directly from base_params
LOGDIR = Path(base_params.get('mp_glob_base_log_path'))
logger.info(f"tsMqlMLTunerModTorch initialized. LOGDIR: {LOGDIR}")


class CMdtunerTorch:
    """
    PyTorch Tuner for machine learning models, integrating with an external Oracle server.
    """
    def __init__(self, tuner_id, oracle_client, is_chief, train_data, val_data, input_shape,
                 oracle_directory, model_save_dir, app_params, tune_params):
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.is_chief = is_chief
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.oracle_directory = Path(oracle_directory) # KerasTuner's local directory
        self.model_save_dir = Path(model_save_dir) # Where final models are saved
        self.app_params = app_params
        self.tune_params = tune_params

        self.oracle_directory.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.max_epochs = tune_params.get('max_epochs', 10)
        self.objective = tune_params.get('objective', 'val_loss') # KerasTuner objective
        self.tuner_type = tune_params.get('tuner_type', 'hyperband') # 'hyperband', 'randomsearch', 'bayesian'
        self.num_trials = tune_params.get('num_trials', 50)
        self.executions_per_trial = tune_params.get('executions_per_trial', 1)
        self.overwrite = tune_params.get('overwrite', False)
        self.batch_size = tune_params.get('batch_size', 32)
        self.learning_rate = tune_params.get('learning_rate', 0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[CMdtunerTorch] Using device: {self.device}")

        self.best_model = None
        self.best_score = float('inf') if 'loss' in self.objective else float('-inf')
        self.best_trial_id = None
        self.best_hyperparameters = None

        # Placeholder for building the model dynamically based on hyperparameters
        # This will be refined when actual hyperparameter sampling is implemented
        self.model = None

    def _build_model(self, hp):
        """
        Dynamically builds a PyTorch model based on hyperparameters.
        This is a placeholder and should be expanded with actual model architecture
        and hyperparameter choices.
        """
        # Example: Simple MLP
        input_dim = self.input_shape[0]
        model = nn.Sequential(
            nn.Linear(input_dim, hp.Int('units_l1', min_value=32, max_value=128, step=32)),
            nn.ReLU(),
            nn.Dropout(hp.Float('dropout_l1', min_value=0.1, max_value=0.5, step=0.1)),
            nn.Linear(hp.Int('units_l1', min_value=32, max_value=128, step=32), 1) # Output layer
        ).to(self.device)
        logger.debug(f"Built model with hyperparameters: {hp.values}")
        return model

    def _train_model(self, model, train_loader, val_loader, hp, trial_id):
        """
        Trains the PyTorch model for a given trial.
        """
        optimizer = optim.Adam(model.parameters(), lr=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
        criterion = nn.MSELoss() # Assuming regression task

        for epoch in range(self.max_epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            logger.info(f"[CMdtunerTorch] Trial {trial_id}, Epoch {epoch+1}/{self.max_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Report intermediate results to Oracle (if needed, or just final)
            # For simplicity, we'll just report final in `run` method for now.
            # In a real scenario, you might report epoch-wise.

        return avg_val_loss # Return the final validation loss

    def run(self):
        """
        Main loop for the tuner to fetch trials from the Oracle and execute them.
        """
        logger.info(f"[CMdtunerTorch] Tuner {self.tuner_id} starting run loop. Is Chief: {self.is_chief}")

        # Convert numpy arrays to PyTorch tensors and create DataLoader
        train_tensor_x = torch.tensor(self.train_data[0], dtype=torch.float32)
        train_tensor_y = torch.tensor(self.train_data[1], dtype=torch.float32)
        val_tensor_x = torch.tensor(self.val_data[0], dtype=torch.float32)
        val_tensor_y = torch.tensor(self.val_data[1], dtype=torch.float32)

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        val_dataset = TensorDataset(val_tensor_x, val_tensor_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for i in range(self.num_trials): # Loop for a fixed number of trials as defined in tune_params
            trial_info = self.oracle_client.get_trial(self.tuner_id)

            if trial_info and trial_info.get('trial_id'):
                trial_id = trial_info['trial_id']
                hyperparameters = trial_info.get('hyperparameters', {})
                logger.info(f"[CMdtunerTorch] Tuner {self.tuner_id} received trial {trial_id} with HP: {hyperparameters}")

                try:
                    # Create a HyperParameters object from the dictionary for _build_model
                    from keras_tuner.engine.hyperparameters import HyperParameters
                    hp = HyperParameters()
                    for k, v in hyperparameters.items():
                        # This is a simplified way to add HPs. In a real scenario,
                        # you'd define the search space in _build_model and use
                        # hp.Choice, hp.Int, hp.Float etc.
                        # For now, we'll just set the values directly.
                        if isinstance(v, int):
                            hp.Int(k, min_value=v, max_value=v)
                        elif isinstance(v, float):
                            hp.Float(k, min_value=v, max_value=v)
                        else:
                            hp.Choice(k, [v])

                    model = self._build_model(hp)
                    val_score = self._train_model(model, train_loader, val_loader, hp, trial_id)

                    status = "COMPLETED"
                    if 'loss' in self.objective:
                        if val_score < self.best_score:
                            self.best_score = val_score
                            self.best_trial_id = trial_id
                            self.best_hyperparameters = hyperparameters
                            self.best_model = model # Keep track of the best model instance
                            # Save checkpoint of the best model if this is the chief
                            if self.is_chief:
                                checkpoint_path = self.model_save_dir / f"best_model_{trial_id}.pth"
                                torch.save(model.state_dict(), checkpoint_path)
                                logger.info(f"[CMdtunerTorch] Chief saved best model checkpoint for trial {trial_id} to {checkpoint_path}")
                    else: # Assuming higher is better for other objectives
                        if val_score > self.best_score:
                            self.best_score = val_score
                            self.best_trial_id = trial_id
                            self.best_hyperparameters = hyperparameters
                            self.best_model = model
                            if self.is_chief:
                                checkpoint_path = self.model_save_dir / f"best_model_{trial_id}.pth"
                                torch.save(model.state_dict(), checkpoint_path)
                                logger.info(f"[CMdtunerTorch] Chief saved best model checkpoint for trial {trial_id} to {checkpoint_path}")

                    self.oracle_client.report_trial_result(trial_id, val_score, status)
                    logger.info(f"[CMdtunerTorch] Worker {self.tuner_id} completed trial {trial_id} with score {val_score}.")

                except Exception as e:
                    logger.error(f"[CMdtunerTorch] Worker {self.tuner_id} encountered error during trial {trial_id}: {e}", exc_info=True)
                    self.oracle_client.update_trial_status(trial_id, status="FAILED")
            else:
                logger.info(f"[CMdtunerTorch] Worker {self.tuner_id} received no new trial from Oracle. Assuming all trials are processed or no more available.")
                break # Exit loop if no new trials are available

        logger.info("✅ Worker finished trial execution for PyTorch.")

    def get_best_model(self):
        """
        Retrieves the best model found during the tuning process from the Oracle.
        If this is the chief, it attempts to load the best model checkpoint.
        """
        logger.info(f"[CMdtunerTorch] {self.tuner_id} attempting to get best trial info from Oracle.")
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
            # Create a HyperParameters object from the dictionary for _build_model
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

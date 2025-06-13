<<<<<<< HEAD
=======

>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
<<<<<<< HEAD
# CORRECTED: Changed to relative import for OracleClient
from .tsMqlMLOracleClient import OracleClient

import logging # Ensure logging is imported
import os # Import os to access environment variables
import time # Import time for sleep

# --- Logging setup ---
# This script now *only* gets a logger. The root logger is configured by multiworker_launcher.py.
# This prevents repeated "Logging initialized" messages and ensures a consistent log file.
logger = logging.getLogger(__name__)
# -- end of logging setup ----


=======
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient

import logging
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
import os
import pathlib
import uuid  # Ensure uuid is imported for use in get_callbacks

<<<<<<< HEAD
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

=======

# -- start of logging setup --
from tsMqlSetup import CMqlSetup
# ✅ Logger and Logdir Setup
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
<<<<<<< HEAD
    num_cores=_estimated_physical_cores,
    num_threads=1
)


=======
    num_cores=8,
    num_threads=1
)
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
<<<<<<< HEAD

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
=======
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
<<<<<<< HEAD

app_params = mql_overrides.env.all_params().get("app", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')


class PyTorchTuner:
    # Added 'directory' as a parameter to the constructor
    def __init__(self, oracle_client, train_dataset, val_dataset, input_shape, num_classes, project_name, max_trials, hypermodel_params, directory="."):
        self.oracle = oracle_client
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.project_name = project_name
        self.max_trials = max_trials
        self.hypermodel_params = hypermodel_params
        self.directory = directory # Store the directory

        # Extract relevant parameters from hypermodel_params
        mltune_params = self.hypermodel_params.get('mltune', {})
        self.epochs = mltune_params.get('epochs', 10)
        self.batch_size = mltune_params.get('batch_size', 32)
        self.objective = mltune_params.get('objective', 'val_loss')

        self.best_model_state = None # To store the state_dict of the best model found
        self.best_val_loss = float('inf')


    def build_model(self, hp):
        # This method defines the PyTorch model architecture based on hyperparameters
        # It should return an instance of torch.nn.Module
        
        # Example: Simple LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x):
                # x shape: (batch_size, sequence_length, input_dim)
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                
                out, _ = self.lstm(x, (h0.detach(), c0.detach()))
                out = self.dropout(out[:, -1, :]) # Get the output of the last time step
                out = self.fc(out)
                return out

        # Hyperparameters for PyTorch model
        input_dim = self.input_shape[1] # Number of features
        output_dim = self.num_classes # Typically 1 for regression

        hidden_dim = hp.get('lstm_units_0', 64) # Example hyperparameter
        num_layers = hp.get('num_lstm_layers', 1) # Example hyperparameter
        dropout_rate = hp.get('dropout_rate', 0.2) # Example hyperparameter

        model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
        return model

    def objective_from_hp(self, hp_values):
        # This method trains a model for a given set of hyperparameters and returns the validation loss
        
        # Convert hp_values dict to a HyperParameters object if build_model expects it
        # For PyTorch, we can directly use the dict
        
        model = self.build_model(hp_values)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer_name = hp_values.get('optimizer', 'Adam').lower()
        learning_rate = hp_values.get('learning_rate', 1e-3)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Default to Adam

        loss_name = hp_values.get('loss', 'mse').lower()
        if loss_name == 'mse':
            criterion = nn.MSELoss()
        elif loss_name == 'mae':
            criterion = nn.L1Loss()
        elif loss_name == 'binary_crossentropy':
            criterion = nn.BCEWithLogitsLoss() # For binary classification
        else:
            criterion = nn.MSELoss() # Default to MSE

        # Prepare data loaders
        X_train_tensor, y_train_tensor = self.train_dataset
        X_val_tensor, y_val_tensor = self.val_dataset

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=self.batch_size, shuffle=False)

        num_epochs = hp_values.get('epochs', self.epochs)

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    val_loss += criterion(outputs_val, batch_y_val).item()
            
            val_loss /= len(val_loader)
            # print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.5f}") # For debugging

        # Update best model if current trial is better
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = model.state_dict() # Save the state dictionary

        return val_loss

    def run_distributed_search(self, train_dataset, val_dataset, epochs, batch_size):
        # This method will coordinate with the OracleClient to fetch trials and report results
        logger.info(f"[PyTorchTuner] Starting distributed search as {os.environ.get('TUNER_ID', 'worker_default')}")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size

        while True:
            try:
                # Direct assignment as OracleClient.get_trial() now returns the dict directly
                trial = self.oracle.get_trial() 

                if not trial or trial.get("trial_id") is None: # Check for None trial_id to signify no more trials
                    logger.info("[PyTorchTuner] 💤 No more trials available (or an error occurred). Exiting.")
                    break

                trial_id = trial["trial_id"]
                hp = trial.get("hyperparameters", {})

                if not isinstance(hp, dict) or not hp:
                    logger.warning(f"[PyTorchTuner] ⚠️ Trial {trial_id} has invalid or empty hyperparameters. Marking as FAILED.")
                    self.oracle.update_trial_status(trial_id, "FAILED")
                    continue

                logger.info(f"[PyTorchTuner] 🔍 Running trial {trial_id} with hyperparameters: {hp}")
                val_loss = self.objective_from_hp(hp)

                logger.info(f"[PyTorchTuner] ✅ Trial {trial_id} completed. val_loss = {val_loss:.5f}")
                self.oracle.report_trial_result(trial_id, {'val_loss': val_loss}) # Report result as dictionary
                self.oracle.update_trial_status(trial_id, "COMPLETED")

            except Exception as e:
                trial_name = trial_id if "trial_id" in locals() else "[UNKNOWN]"
                logger.error(f"[PyTorchTuner] ❌ Exception during trial {trial_name}: {e}", exc_info=True)
                if "trial_id" in locals():
                    self.oracle.update_trial_status(trial_id, "FAILED")
                # Do not break immediately, try to get next trial after a pause
                time.sleep(5) 

        logger.info("[PyTorchTuner] 🧹 Distributed search complete.")

    def get_best_model(self):
        # Rebuild the best model from the saved state dictionary
        if self.best_model_state:
            # You need to pass a dummy hp object or the best hp values to build_model
            # to reconstruct the model with the correct architecture
            # For simplicity, let's assume a fixed architecture for now or retrieve best hp from Oracle
            # For a real scenario, you'd fetch the best HP from Oracle or store it.
            # For demonstration, we'll use a placeholder for hp_values
            
            # This is a placeholder. In a real scenario, you'd fetch the best hyperparameters
            # from the Oracle or store them along with best_model_state.
            # For now, we'll use default values or infer from input_shape.
            hp_values_for_best_model = {
                'lstm_units_0': 64, # Example default
                'num_lstm_layers': 1, # Example default
                'dropout_rate': 0.2, # Example default
            }
            # Attempt to get the actual best hyperparameters if available (e.g., from OracleClient)
            if self.oracle:
                best_trial_info = self.oracle.get_best_trial()
                if best_trial_info and 'hyperparameters' in best_trial_info:
                    hp_values_for_best_model = best_trial_info['hyperparameters']
                    logger.info(f"Retrieved best hyperparameters for model reconstruction: {hp_values_for_best_model}")
                else:
                    logger.warning("Could not retrieve best hyperparameters from Oracle. Using default/placeholder values for model reconstruction.")


            best_model = self.build_model(hp_values_for_best_model)
            best_model.load_state_dict(self.best_model_state)
            best_model.eval() # Set to evaluation mode
            return best_model
        return None

    def export_best_model(self, ftype='pth'):
        best_model = self.get_best_model()
        if best_model:
            # Use self.hypermodel_params to get project_dir and modelname
            base_params = self.hypermodel_params.get('base', {})
            app_params = self.hypermodel_params.get('app', {})
            
            project_dir = base_params.get('mp_glob_base_ml_project_dir', os.path.join(os.getcwd(), "tuner_projects"))
            modelname = app_params.get('mp_glob_sub_ml_model_name', 'ts_mql_model')
            modeldatapath = base_params.get('mp_glob_sub_ml_src_modeldata', os.path.join(project_dir, "model_data"))

            os.makedirs(modeldatapath, exist_ok=True)
            
            if ftype == 'pth':
                export_filepath = os.path.join(modeldatapath, f"{modelname}.pth")
                torch.save(best_model.state_dict(), export_filepath)
                logger.info(f"PyTorch model state_dict saved to {export_filepath}")
                return export_filepath
            else:
                logger.warning(f"Unsupported export file type for PyTorch: {ftype}. Only 'pth' is supported.")
        else:
            logger.warning("No best PyTorch model found to export.")
        return None

    def check_and_load_model(self, lpbase_path, ftype='pth'):
        logger.info(f"Checking for PyTorch model file at base_path {lpbase_path}")
        
        # Use self.hypermodel_params to get modelname
        app_params = self.hypermodel_params.get('app', {})
        modelname = app_params.get('mp_glob_sub_ml_model_name', 'ts_mql_model')

        if ftype == 'pth':
            localmodel = modelname + '.pth'
            model_path = os.path.join(lpbase_path, localmodel)
            logger.info(f"Model path pth : {model_path}")
        else:
            logger.warning(f"Unsupported file type for loading PyTorch model: {ftype}. Only 'pth' is supported.")
            return None

        try:
            if os.path.exists(model_path):
                # To load the model, you need to first instantiate the model class
                # with the correct hyperparameters that were used to train it.
                # This is a placeholder, as the exact HP for loading isn't stored here.
                # In a real application, you'd save/load the best HP along with the model state.
                
                # For now, we'll use a dummy hp object for model reconstruction
                hp_values_for_loading = {
                    'lstm_units_0': 64, # Example default
                    'num_lstm_layers': 1, # Example default
                    'dropout_rate': 0.2, # Example default
                }
                # Attempt to get the actual best hyperparameters if available (e.g., from OracleClient)
                if self.oracle:
                    best_trial_info = self.oracle.get_best_trial()
                    if best_trial_info and 'hyperparameters' in best_trial_info:
                        hp_values_for_loading = best_trial_info['hyperparameters']
                        logger.info(f"Retrieved best hyperparameters for model loading: {hp_values_for_loading}")
                    else:
                        logger.warning("Could not retrieve best hyperparameters from Oracle. Using default/placeholder values for model loading.")


                model = self.build_model(hp_values_for_loading)
                model.load_state_dict(torch.load(model_path))
                model.eval() # Set to evaluation mode
                logger.info(f"PyTorch model loaded successfully from {model_path}")
                # You might want to print model summary here if PyTorch has a similar concept
                return model
            else:
                logger.info(f"PyTorch model file does not exist at {model_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return None
=======
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)

logger = setup_config.setup_global_logger(global_logfile, force_reset=True)
# -- end of logging setup ----

# Platform imports
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk         = run_platform.RunPlatform()
os_platform  = platform_checker.get_platform()
loadmql      = pchk.check_mql_state()


class PyTorchTuner:
    def __init__(self, **kwargs):
        self.params = kwargs.get("hypermodel_params", {})
        self.train_data = kwargs.get("traindataset")
        self.val_data = kwargs.get("valdataset")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.oracle = kwargs.get("oracle", OracleClient())
        self.logdev = kwargs.get("logdev", False)

    def prepare_data(self):
        def extract_xy(data):
            if isinstance(data, (list, tuple)) and len(data) == 2:
                x, y = data
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                if isinstance(y, np.ndarray):
                    y = torch.tensor(y, dtype=torch.float32)
                return x, y
            elif isinstance(data, TensorDataset):
                x_all, y_all = [], []
                for x, y in data:
                    x_all.append(x.unsqueeze(0))
                    y_all.append(y.unsqueeze(0))
                return torch.cat(x_all, dim=0), torch.cat(y_all, dim=0)
            raise ValueError("Unsupported dataset format. Expected (x, y) tuple or TensorDataset.")

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
        if self.logdev:
             print(f"[PyTorchTuner] 💻 Using device: {self.device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loader, val_loader = self.prepare_data()
        sample_batch = next(iter(train_loader))[0]
        input_dim = int(np.prod(sample_batch.shape[1:]))

        print(f"[PyTorchTuner] Input batch shape: {sample_batch.shape}, computed input_dim: {input_dim}")

        model = self.build_model(hp, input_dim)
        print(f"[PyTorchTuner] 🧠 Model initialized on device: {next(model.parameters()).device}")
        optimizer_name = hp.get("optimizer", "Adam")
        lr = hp.get("lr", 1e-3)
        optimizer = getattr(optim, optimizer_name, optim.Adam)(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        try:
            for epoch in range(5):
                model.train()
                for xb, yb in train_loader:
                    if self.logdev:
                        print(f"[PyTorchTuner] 📦 Training batch shape: {xb.shape}, target shape: {yb.shape}")
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    preds = model(xb).squeeze()
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            if self.device == "cuda":
                print(f"[PyTorchTuner] 🔋 GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                print(f"[PyTorchTuner] 🔋 GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
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
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2

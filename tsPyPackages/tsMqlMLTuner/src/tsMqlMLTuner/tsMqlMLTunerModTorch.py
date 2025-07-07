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
from pathlib import Path # <--- ADDED: Import Path from pathlib


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

# Load configuration for tuner module
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})
base_params = all_params.get("base", {})


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate, num_lstm_layers):
        super(LSTMModel, self).__init__()
        logger.info(f"[LSTMModel] Initialized with input_size={input_size}, hidden_layer_sizes={hidden_layer_sizes}, output_size={output_size}, dropout_rate={dropout_rate}, num_lstm_layers={num_lstm_layers}")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_lstm_layers = num_lstm_layers # Number of LSTM layers

        # LSTM Layer(s)
        lstm_layers = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if i == 0:
                lstm_layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
            else:
                lstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_size, batch_first=True))
            # Add dropout after each LSTM layer except the last one
            if dropout_rate > 0 and i < len(hidden_layer_sizes) - 1:
                lstm_layers.append(nn.Dropout(dropout_rate))
        self.lstm_layers = nn.ModuleList(lstm_layers)

        # Dropout layer (after LSTM, before Dense)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer after LSTM
        self.fc = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x):
        # Pass through LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            if isinstance(lstm_layer, nn.LSTM):
                x, _ = lstm_layer(x) # x is (batch_size, seq_len, hidden_size)
            elif isinstance(lstm_layer, nn.Dropout):
                x = lstm_layer(x)
        
        # Take the output of the last time step
        x = x[:, -1, :] # x is (batch_size, hidden_size)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate, num_gru_layers):
        super(GRUModel, self).__init__()
        logger.info(f"[GRUModel] Initialized with input_size={input_size}, hidden_layer_sizes={hidden_layer_sizes}, output_size={output_size}, dropout_rate={dropout_rate}, num_gru_layers={num_gru_layers}")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_gru_layers = num_gru_layers

        gru_layers = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if i == 0:
                gru_layers.append(nn.GRU(input_size, hidden_size, batch_first=True))
            else:
                gru_layers.append(nn.GRU(hidden_layer_sizes[i-1], hidden_size, batch_first=True))
            if dropout_rate > 0 and i < len(hidden_layer_sizes) - 1:
                gru_layers.append(nn.Dropout(dropout_rate))
        self.gru_layers = nn.ModuleList(gru_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x):
        for i, gru_layer in enumerate(self.gru_layers):
            if isinstance(gru_layer, nn.GRU):
                x, _ = gru_layer(x)
            elif isinstance(gru_layer, nn.Dropout):
                x = gru_layer(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, input_size, cnn_units, output_size, dropout_rate):
        super(CNNModel, self).__init__()
        logger.info(f"[CNNModel] Initialized with input_size={input_size}, cnn_units={cnn_units}, output_size={output_size}, dropout_rate={dropout_rate}")
        self.conv1 = nn.Conv1d(input_size, cnn_units, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        # Calculate the input size for the fully connected layer
        # Assuming input sequence length is 60 (from data_X.shape[1])
        # After conv1 (padding=1, kernel_size=3), sequence length is still 60.
        # After pool (kernel_size=2), sequence length becomes 60 // 2 = 30.
        self.fc = nn.Linear(cnn_units * (60 // 2), output_size) # Assuming seq_len=60

    def forward(self, x):
        # Input x is (batch_size, seq_len, input_size)
        # Conv1d expects (batch_size, input_channels, seq_len)
        x = x.permute(0, 2, 1) # -> (batch_size, input_size, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten for FC layer
        x = self.fc(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        logger.info(f"[MultiHeadSelfAttention] Initialized with embed_dim={embed_dim}, num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
        dk = torch.tensor(self.head_dim, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.embed_dim)
        output = self.dense(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        logger.info(f"[TransformerBlock] Initialized with embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}, dropout_rate={dropout_rate}")
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, ff_dim, output_size, num_transformer_blocks, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        logger.info(f"[TransformerModel] Initialized with input_size={input_size}, embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}, output_size={output_size}, num_transformer_blocks={num_transformer_blocks}, dropout_rate={dropout_rate}")
        self.embedding = nn.Linear(input_size, embed_dim) # Project input_size to embed_dim
        self.pos_encoding = self.positional_encoding(60, embed_dim) # Assuming seq_len=60
        
        transformer_blocks = []
        for _ in range(num_transformer_blocks):
            transformer_blocks.append(TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate))
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        self.fc = nn.Linear(embed_dim, output_size)

    def positional_encoding(self, position, d_model):
        pe = torch.zeros(position, d_model)
        position_tensor = torch.arange(0, position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position_tensor * div_term)
        pe[:, 1::2] = torch.cos(position_tensor * div_term)
        return pe.unsqueeze(0) # Add batch dimension

    def forward(self, x):
        # x is (batch_size, seq_len, input_size)
        x = self.embedding(x) # -> (batch_size, seq_len, embed_dim)
        
        # Add positional encoding
        # Ensure positional encoding is on the same device as x
        x += self.pos_encoding[:, :x.size(1), :].to(x.device)

        for block in self.transformer_blocks:
            x = block(x)
        
        x = x.mean(dim=1) # Global average pooling over the sequence dimension
        x = self.fc(x)
        return x


class CMdtunerTorch:
    def __init__(self, tuner_id, oracle_client, hypermodel_params, dataset_params, base_path, model_id, model_save_dir, train_data, val_data, input_shape, num_classes, is_chief=True):
        logger.info(f"[PyTorchTuner] Initializing with input_shape: {input_shape}, num_classes: {num_classes}")
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.hypermodel_params = hypermodel_params
        self.dataset_params = dataset_params
        self.base_path = Path(base_path)
        self.model_id = model_id
        self.model_dir = Path(model_save_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[PyTorchTuner] Model save directory: {self.model_dir}")

        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_chief = is_chief
        self.best_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[PyTorchTuner] Using device: {self.device}")

        # Extract hyperparameters ranges from tune_params
        self.tune_params = self.hypermodel_params.get("mltune", {})
        self.hp_ranges = {
            'trans_dim': (self.tune_params.get('trans_dim_min', 32), self.tune_params.get('trans_dim_max', 256), self.tune_params.get('trans_dim_step', 32)),
            'lstm_units': (self.tune_params.get('lstm_units_min', 32), self.tune_params.get('lstm_units_max', 128), self.tune_params.get('lstm_units_step', 32)),
            'gru_units': (self.tune_params.get('gru_units_min', 32), self.tune_params.get('gru_units_max', 128), self.tune_params.get('gru_units_step', 32)),
            'cnn_units': (self.tune_params.get('cnn_units_min', 32), self.tune_params.get('cnn_units_max', 128), self.tune_params.get('cnn_units_step', 32)),
            'trans_heads': (self.tune_params.get('trans_heads_min', 2), self.tune_params.get('trans_heads_max', 8), self.tune_params.get('trans_heads_step', 2)),
            'trans_ff': (self.tune_params.get('trans_ff_min', 64), self.tune_params.get('trans_ff_max', 512), self.tune_params.get('trans_ff_step', 64)),
            'dense_units': (self.tune_params.get('dense_units_min', 32), self.tune_params.get('dense_units_max', 128), self.tune_params.get('dense_units_step', 32)),
        }
        self.max_epochs = self.tune_params.get('max_epochs', 10)
        self.batch_size = self.tune_params.get('batch_size', 32)
        self.objective = self.tune_params.get('objective', 'val_loss')


    def build_model(self, hp):
        input_size = self.input_shape[1] # Features dimension
        output_size = self.num_classes
        dropout_rate = self.tune_params.get('dropout', 0.2)

        # Get the configured model type
        configured_model_type = self.tune_params.get('ml_model_name', 'lstm')
        supported_models = ['lstm', 'gru', 'cnn', 'transformer']

        # Check if the configured model type is supported, otherwise fallback to 'lstm'
        if configured_model_type not in supported_models:
            logger.warning(f"Configured model type '{configured_model_type}' is unsupported. Falling back to 'lstm' model.")
            model_type = 'lstm'
        else:
            model_type = configured_model_type

        if hp: # If hyperparameter object is provided by Oracle (during tuning)
            # Access hyperparameters directly from hp.values dictionary
            if model_type == 'lstm':
                lstm_units = hp.values.get('lstm_units', self.hp_ranges['lstm_units'][0]) # Use .get with a default from ranges
                num_lstm_layers = hp.values.get('num_lstm_layers', 1)
                model = LSTMModel(input_size, [lstm_units] * num_lstm_layers, output_size, dropout_rate, num_lstm_layers).to(self.device)
            elif model_type == 'gru':
                gru_units = hp.values.get('gru_units', self.hp_ranges['gru_units'][0])
                num_gru_layers = hp.values.get('num_gru_layers', 1)
                model = GRUModel(input_size, [gru_units] * num_gru_layers, output_size, dropout_rate, num_gru_layers).to(self.device)
            elif model_type == 'cnn':
                cnn_units = hp.values.get('cnn_units', self.hp_ranges['cnn_units'][0])
                model = CNNModel(input_size, cnn_units, output_size, dropout_rate).to(self.device)
            elif model_type == 'transformer':
                embed_dim = hp.values.get('trans_dim', self.hp_ranges['trans_dim'][0])
                num_heads = hp.values.get('trans_heads', self.hp_ranges['trans_heads'][0])
                ff_dim = hp.values.get('trans_ff', self.hp_ranges['trans_ff'][0])
                num_transformer_blocks = hp.values.get('num_transformer_blocks', 1)
                model = TransformerModel(input_size, embed_dim, num_heads, ff_dim, output_size, num_transformer_blocks, dropout_rate).to(self.device)
        else: # If hp is None (e.g., for loading best model or fixed run)
            # Use default values or values from tune_params directly
            if model_type == 'lstm':
                lstm_units = self.tune_params.get('lstm_units_default', 64)
                num_lstm_layers = 1 # Default number of layers if not tuning
                model = LSTMModel(input_size, [lstm_units] * num_lstm_layers, output_size, dropout_rate, num_lstm_layers).to(self.device)
            elif model_type == 'gru':
                gru_units = self.tune_params.get('gru_units_default', 64)
                num_gru_layers = 1
                model = GRUModel(input_size, [gru_units] * num_gru_layers, output_size, dropout_rate, num_gru_layers).to(self.device)
            elif model_type == 'cnn':
                cnn_units = self.tune_params.get('cnn_units_default', 64)
                model = CNNModel(input_size, cnn_units, output_size, dropout_rate).to(self.device)
            elif model_type == 'transformer':
                embed_dim = self.tune_params.get('trans_dim_default', 64)
                num_heads = self.tune_params.get('trans_heads_default', 2) # Assuming default in config
                ff_dim = self.tune_params.get('trans_ff_default', 128) # Assuming default in config
                num_transformer_blocks = 1
                model = TransformerModel(input_size, embed_dim, num_heads, ff_dim, output_size, num_transformer_blocks, dropout_rate).to(self.device)

        optimizer_choice = self.tune_params.get('optimizer', 'adam')
        lr = self.tune_params.get('learning_rate', 0.001) # Default LR if not tuning
        if optimizer_choice == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_choice == 'nadam':
            optimizer = optim.NAdam(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr) # Default to Adam

        loss_name = self.tune_params.get('loss', 'mean_squared_error')
        if loss_name == 'mean_squared_error':
            loss_fn = nn.MSELoss()
        elif loss_name == 'mean_absolute_error':
            loss_fn = nn.L1Loss()
        else:
            loss_fn = nn.MSELoss() # Default to MSELoss

        return model, optimizer, loss_fn


    def run(self):
        logger.info(f"[CMdtunerTorch] Chief run method triggered")
        if not self.is_chief:
            self._run_worker()
            return

        self.best_score = float('inf')
        self.best_hps = None
        self.best_trial_id = None

        # Chief's tuning loop
        # The chief requests trials from the OracleServer
        # The OracleServer (which holds the CustomOracle) manages trial creation and state
        for i in range(self.tune_params.get('num_trials', 50)):
            logger.info(f"[CMdtunerTorch] Chief requesting trial {i+1}...")
            # Changed from get_trial to request_trial
            trial_data = self.oracle_client.request_trial(self.tuner_id)
            
            # Check if trial_data is None or if trial_id is None
            if trial_data is None or trial_data.get("trial_id") is None:
                logger.info("🛑 No more trials to run or an invalid trial was returned. Exiting tuning loop.")
                break

            trial_id = trial_data["trial_id"]
            hps_values = trial_data["hyperparameters"]
            
            # Create a SimpleNamespace object to mimic KerasTuner's HyperParameters object
            # for easy access to hps_values within build_model
            hp = SimpleNamespace(values=hps_values)
            
            logger.info(f"⚡ Running trial {trial_id} with hyperparameters: {hps_values}")

            model, optimizer, loss_fn = self.build_model(hp)
            model.to(self.device)

            # Training loop
            for epoch in range(self.max_epochs):
                model.train()
                for batch_idx, (X_batch, y_batch) in enumerate(self.train_data):
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = loss_fn(outputs.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()

                # Validation step
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in self.val_data:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                        val_preds = model(X_val_batch)
                        val_loss += loss_fn(val_preds.squeeze(), y_val_batch).item()
                val_loss /= len(self.val_data)

                logger.info(f"Trial {trial_id}, Epoch {epoch+1}/{self.max_epochs}, Val Loss: {val_loss:.4f}")

                # Report intermediate results to Oracle (optional, but good for Hyperband/Bayesian)
                self.oracle_client.update_trial_status(trial_id, status="RUNNING") # Or a more granular update_trial method
                # Note: KerasTuner's Oracle.update_trial takes step and metrics.
                # For simplicity, we'll just report final result.

            # After training, report the final result to the Oracle
            self.oracle_client.report_result(trial_id, val_loss)
            logger.info(f"✅ Trial {trial_id} completed with score: {val_loss}")

            # Update best model if current trial is better
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_hps = hps_values
                self.best_trial_id = trial_id
                # Save trial-specific model
                trial_model_path = self.model_dir / f"best_model_{trial_id}.pth"
                torch.save(model.state_dict(), trial_model_path)
                logger.info(f"✨ New best model saved for trial {trial_id} with score: {val_loss}")
                # Also save as global best_model.pth
                global_model_path = self.model_dir / "best_model.pth"
                torch.save(model.state_dict(), global_model_path)


    def _run_worker(self):
        logger.info(f"[CMdtunerTorch] Worker {self.tuner_id} starting its trial execution loop...")
        while True:
            try:
                # Workers request trials from the OracleServer
                trial_data = self.oracle_client.request_trial(self.tuner_id)

                # Check if trial_data is None or if trial_id is None
                if trial_data is None or trial_data.get("trial_id") is None:
                    logger.info(f"🛑 Worker {self.tuner_id}: No more trials available or an invalid trial was returned. Exiting.")
                    break

                trial_id = trial_data["trial_id"]
                hps_values = trial_data["hyperparameters"]
                
                hp = SimpleNamespace(values=hps_values)

                logger.info(f"⚡ Worker {self.tuner_id} running trial {trial_id} with hyperparameters: {hps_values}")

                model, optimizer, loss_fn = self.build_model(hp)
                model.to(self.device)

                # Training loop
                for epoch in range(self.max_epochs):
                    model.train()
                    for batch_idx, (X_batch, y_batch) in enumerate(self.train_data):
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = loss_fn(outputs.squeeze(), y_batch)
                        loss.backward()
                        optimizer.step()

                # Validation step
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in self.val_data:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                        val_preds = model(X_val_batch)
                        val_loss += loss_fn(val_preds.squeeze(), y_val_batch).item()
                    val_loss /= len(self.val_data)

                    logger.info(f"Worker {self.tuner_id}, Trial {trial_id}, Epoch {epoch+1}/{self.max_epochs}, Val Loss: {val_loss:.4f}")
                    # Workers can report intermediate status if needed, but typically only final result.
                    # self.oracle_client.update_trial_status(trial_id, status="RUNNING")

                # Report the final result to the Oracle
                self.oracle_client.report_result(trial_id, val_loss)
                logger.info(f"✅ Worker {self.tuner_id}, Trial {trial_id} completed with score: {val_loss}")

            except Exception as e:
                logger.exception(f"💥 Worker {self.tuner_id}, Trial {trial_id} failed: {e}")
                self.oracle_client.report_result(trial_id, float("inf"), status="FAILED")
                # Consider a mechanism to stop the worker if too many failures
                time.sleep(self.tune_params.get('retry_delay', 5)) # Wait before requesting next trial


    def get_best_model_path(self):
        return self.model_dir / "best_model.pth"

    def get_best_model(self):
        model_type = self.tune_params.get('ml_model_name', 'lstm')
        input_size = self.input_shape[1]
        output_size = self.num_classes
        dropout_rate = self.tune_params.get('dropout', 0.2)

        # Build a model instance with default/best known hyperparameters (hp=None)
        # This will use the default values set in build_model if no tuning has occurred,
        # or the best found HPs if `self.best_hps` was set by the chief.
        # For simplicity, we'll build with defaults here. If you want to load a specific best HP model,
        # you'd need to pass `SimpleNamespace(values=self.best_hps)` to build_model.
        
        # If self.best_hps is available, use it to build the model structure
        if self.best_hps:
            hp_for_best_model = SimpleNamespace(values=self.best_hps)
            model, _, _ = self.build_model(hp_for_best_model)
        else:
            # Fallback to default model structure if no best_hps found (e.g., tuning didn't run)
            model, _, _ = self.build_model(None)


        best_model_path = self.get_best_model_path()
        if os.path.exists(best_model_path):
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                model.eval() # Set to evaluation mode
                logger.info(f"✅ Loaded best model from {best_model_path}")
                self.best_model = model # Store the loaded model
                return model
            except Exception as e:
                logger.error(f"❌ Error loading best model from {best_model_path}: {e}", exc_info=True)
                return None
        else:
            logger.warning("Best model path does not exist. Returning None.")
            return None

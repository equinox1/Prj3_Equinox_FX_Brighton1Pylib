#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlMLTunerParams.py
File: tsPyPackages/tsMqlMLTunerParams/tsMqlMLTunerParams.py
Description: Load and set machine learning tuner parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
License: MIT License
"""

import os
from datetime import datetime
from tsMqlEnvCore import CEnvCore
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)

def get_current_datetime() -> str:
    """Return the current date and time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class CMqlEnvMLTunerParams(CEnvCore):
    """Manage machine learning tuner parameters."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the machine learning tuner parameters.
        
        Keyword arguments allow overriding of default values.
        """
        # Initialize parent with custom parameters.
        super().__init__(custom_params=kwargs)

        # Get a consistent datetime value.
        current_datetime: str = kwargs.get('today', get_current_datetime())
        self.params: Dict[str, Any] = {}  # Ensure the params dict exists
        self.params["today"] = current_datetime
        self.today: str = current_datetime

        # General settings.
        self.seed: int = kwargs.get('seed', 42)
        self.tuner_id: Optional[str] = kwargs.get('tuner_id', None)

        # Tuner mode settings.
        self.tunemode: str = kwargs.get('tunemode', 'Hyperband')
        self.tunemodeepochs: int = kwargs.get('tunemodeepochs', 100)
        self.modelsummary: bool = kwargs.get('modelsummary', False)

        # Dataset paths.
        self.train_dataset: Optional[str] = kwargs.get('train_dataset', None)
        self.val_dataset: Optional[str] = kwargs.get('val_dataset', None)
        self.test_dataset: Optional[str] = kwargs.get('test_dataset', None)

        # Input configuration.
        self.input_shape: Optional[List[int]] = kwargs.get('input_shape', None)
        self.data_input_shape: Optional[List[int]] = kwargs.get('data_input_shape', None)
        self.multi_inputs: bool = kwargs.get('multi_inputs', False)
        self.batch_size: int = kwargs.get('batch_size', 32)

        # Model selection flags.
        self.cnn_model: bool = kwargs.get('cnn_model', True)
        self.lstm_model: bool = kwargs.get('lstm_model', True)
        self.gru_model: bool = kwargs.get('gru_model', True)
        self.transformer_model: bool = kwargs.get('transformer_model', True)
        self.multiactivate: bool = kwargs.get('multiactivate', True)
        self.multi_branches: bool = kwargs.get('multi_branches', True)
        self.multi_outputs: bool = kwargs.get('multi_outputs', False)

        # Label and windowing configuration.
        self.label_columns: Optional[List[str]] = kwargs.get('label_columns', None)
        self.input_width: Optional[int] = kwargs.get('input_width', None)
        self.shift: Optional[int] = kwargs.get('shift', None)
        self.total_window_size: Optional[int] = kwargs.get('total_window_size', None)
        self.label_width: Optional[int] = kwargs.get('label_width', None)

        # Tuner application settings.
        self.keras_tuner: str = kwargs.get('keras_tuner', 'Hyperband')
        self.hyperband_iterations: int = kwargs.get('hyperband_iterations', 1)

        # Epoch settings.
        self.max_epochs: int = kwargs.get('max_epochs', 100)
        self.min_epochs: int = kwargs.get('min_epochs', 10)
        self.tf_param_epochs: int = kwargs.get('tf_param_epochs', 10)
        self.epochs: int = kwargs.get('epochs', 2)
        self.tune_new_entries: bool = kwargs.get('tune_new_entries', True)
        self.allow_new_entries: bool = kwargs.get('allow_new_entries', True)

        # Core tuner parameters.
        self.num_trials: int = kwargs.get('num_trials', 3)
        self.max_retries_per_trial: int = kwargs.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials: int = kwargs.get('max_consecutive_failed_trials', 3)
        self.steps_per_execution: int = kwargs.get('steps_per_execution', 50)
        self.executions_per_trial: int = kwargs.get('executions_per_trial', 1)
        self.overwrite: bool = kwargs.get('overwrite', True)

        # Extra tuner parameters.
        self.factor: int = kwargs.get('factor', 10)
        self.objective: str = kwargs.get('objective', 'val_loss')

        # Logging and TensorFlow version flags.
        self.logger: Optional[Any] = kwargs.get('logger', None)
        self.tf1: bool = kwargs.get('tf1', False)
        self.tf2: bool = kwargs.get('tf2', False)

        # Evaluator settings.
        self.optimizer: str = kwargs.get('optimizer', 'adam')
        self.loss: str = kwargs.get('loss', 'mean_squared_error')
        self.metrics: List[str] = kwargs.get('metrics', ['mean_squared_error'])
        self.dropout: float = kwargs.get('dropout', 0.2)

        # Checkpointing options.
        self.chk_fullmodel: bool = kwargs.get('chk_fullmodel', True)
        self.chk_verbosity: int = kwargs.get('chk_verbosity', 1)
        self.chk_mode: str = kwargs.get('chk_mode', 'min')
        self.chk_monitor: str = kwargs.get('chk_monitor', 'val_loss')
        self.chk_sav_freq: str = kwargs.get('chk_sav_freq', 'epoch')
        self.chk_patience: int = kwargs.get('chk_patience', 3)

        # Model tuner parameters.
        self.unitmin: int = kwargs.get('unitmin', 32)
        self.unitmax: int = kwargs.get('unitmax', 512)
        self.unitstep: int = kwargs.get('unitstep', 32)
        self.defaultunits: int = kwargs.get('defaultunits', 128)

        # Model scales.
        self.all_modelscale: float = kwargs.get('all_modelscale', 1.0)
        self.cnn_modelscale: float = kwargs.get('cnn_modelscale', 1.0)
        self.lstm_modelscale: float = kwargs.get('lstm_modelscale', 1.0)
        self.gru_modelscale: float = kwargs.get('gru_modelscale', 1.0)
        self.trans_modelscale: float = kwargs.get('trans_modelscale', 1.0)
        self.transh_modelscale: float = kwargs.get('transh_modelscale', 1.0)
        self.transff_modelscale: float = kwargs.get('transff_modelscale', 1.0)
        self.dense_modelscale: float = kwargs.get('dense_modelscale', 1.0)

        # Additional model parameters (ensure integer values where applicable).
        self.trans_dim_min: int = int(kwargs.get('trans_dim_min', 32 / self.trans_modelscale))
        self.trans_dim_max: int = int(kwargs.get('trans_dim_max', 256 / self.trans_modelscale))
        self.trans_dim_step: int = int(kwargs.get('trans_dim_step', 32 / self.trans_modelscale))
        self.trans_dim_default: int = int(kwargs.get('trans_dim_default', 64 / self.trans_modelscale))

        self.lstm_units_min: int = int(kwargs.get('lstm_units_min', 32 / self.lstm_modelscale))
        self.lstm_units_max: int = int(kwargs.get('lstm_units_max', 128 / self.lstm_modelscale))
        self.lstm_units_step: int = int(kwargs.get('lstm_units_step', 32 / self.lstm_modelscale))
        self.lstm_units_default: int = int(kwargs.get('lstm_units_default', 64 / self.lstm_modelscale))

        self.gru_units_min: int = int(kwargs.get('gru_units_min', 32 / self.gru_modelscale))
        self.gru_units_max: int = int(kwargs.get('gru_units_max', 128 / self.gru_modelscale))
        self.gru_units_step: int = int(kwargs.get('gru_units_step', 32 / self.gru_modelscale))
        self.gru_units_default: int = int(kwargs.get('gru_units_default', 64 / self.gru_modelscale))

        self.cnn_units_min: int = int(kwargs.get('cnn_units_min', 32 / self.cnn_modelscale))
        self.cnn_units_max: int = int(kwargs.get('cnn_units_max', 128 / self.cnn_modelscale))
        self.cnn_units_step: int = int(kwargs.get('cnn_units_step', 32 / self.cnn_modelscale))
        self.cnn_units_default: int = int(kwargs.get('cnn_units_default', 64 / self.cnn_modelscale))

        self.trans_heads_min: int = int(kwargs.get('trans_heads_min', 2 / self.transh_modelscale))
        self.trans_heads_max: int = int(kwargs.get('trans_heads_max', 8 / self.transh_modelscale))
        self.trans_heads_step: int = int(kwargs.get('trans_heads_step', 2 / self.transh_modelscale))

        self.trans_ff_min: int = int(kwargs.get('trans_ff_min', 64 / self.transff_modelscale))
        self.trans_ff_max: int = int(kwargs.get('trans_ff_max', 512 / self.transff_modelscale))
        self.trans_ff_step: int = int(kwargs.get('trans_ff_step', 64 / self.transff_modelscale))

        self.dense_units_min: int = int(kwargs.get('dense_units_min', 32 / self.dense_modelscale))
        self.dense_units_max: int = int(kwargs.get('dense_units_max', 128 / self.dense_modelscale))
        self.dense_units_step: int = int(kwargs.get('dense_units_step', 32 / self.dense_modelscale))

        # Base path for file operations.
        self.base_path: str = kwargs.get('base_path', os.getcwd())

        # Update the internal parameter dictionary with all default tuner parameters.
        self.params.update(self.get_default_params())

    def get_tuner_params(self) -> Dict[str, Any]:
        """
        Retrieve tuner-specific parameters from the environment.
        
        Returns:
            A dictionary containing the tuner parameters.
        """
        return self.params.get("tuner", self.params)

    def get_default_params(self) -> Dict[str, Any]:
        """
        Compile and return a dictionary of default parameters based on the current instance attributes.
        
        Returns:
            A dictionary with all the default tuner and model parameters.
        """
        return {
            # General settings
            "today": self.today,
            "seed": self.seed,
            "tuner_id": self.tuner_id,
            
            # Tuner mode settings
            "tunemode": self.tunemode,
            "tunemodeepochs": self.tunemodeepochs,
            "modelsummary": self.modelsummary,
            
            # Dataset paths
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "test_dataset": self.test_dataset,
            
            # Input configuration
            "input_shape": self.input_shape,
            "data_input_shape": self.data_input_shape,
            "multi_inputs": self.multi_inputs,
            "batch_size": self.batch_size,
            
            # Model selection flags
            "cnn_model": self.cnn_model,
            "lstm_model": self.lstm_model,
            "gru_model": self.gru_model,
            "transformer_model": self.transformer_model,
            "multiactivate": self.multiactivate,
            "multi_branches": self.multi_branches,
            "multi_outputs": self.multi_outputs,
            
            # Label and windowing configuration
            "label_columns": self.label_columns,
            "input_width": self.input_width,
            "shift": self.shift,
            "total_window_size": self.total_window_size,
            "label_width": self.label_width,
            
            # Tuner application settings
            "keras_tuner": self.keras_tuner,
            "hyperband_iterations": self.hyperband_iterations,
            
            # Epoch settings
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "tf_param_epochs": self.tf_param_epochs,
            "epochs": self.epochs,
            "tune_new_entries": self.tune_new_entries,
            "allow_new_entries": self.allow_new_entries,
            
            # Core tuner parameters
            "num_trials": self.num_trials,
            "max_retries_per_trial": self.max_retries_per_trial,
            "max_consecutive_failed_trials": self.max_consecutive_failed_trials,
            "steps_per_execution": self.steps_per_execution,
            "executions_per_trial": self.executions_per_trial,
            "overwrite": self.overwrite,
            
            # Extra tuner parameters
            "factor": self.factor,
            "objective": self.objective,
            
            # Logging and TensorFlow flags
            "logger": self.logger,
            "tf1": self.tf1,
            "tf2": self.tf2,
            
            # Evaluator settings
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
            "dropout": self.dropout,
            
            # Checkpointing options
            "chk_fullmodel": self.chk_fullmodel,
            "chk_verbosity": self.chk_verbosity,
            "chk_mode": self.chk_mode,
            "chk_monitor": self.chk_monitor,
            "chk_sav_freq": self.chk_sav_freq,
            "chk_patience": self.chk_patience,
            
            # Model tuner parameters
            "unitmin": self.unitmin,
            "unitmax": self.unitmax,
            "unitstep": self.unitstep,
            "defaultunits": self.defaultunits,
            
            # Model scales
            "all_modelscale": self.all_modelscale,
            "cnn_modelscale": self.cnn_modelscale,
            "lstm_modelscale": self.lstm_modelscale,
            "gru_modelscale": self.gru_modelscale,
            "trans_modelscale": self.trans_modelscale,
            "transh_modelscale": self.transh_modelscale,
            "transff_modelscale": self.transff_modelscale,
            "dense_modelscale": self.dense_modelscale,
            
            # Additional model parameters
            "trans_dim_min": self.trans_dim_min,
            "trans_dim_max": self.trans_dim_max,
            "trans_dim_step": self.trans_dim_step,
            "trans_dim_default": self.trans_dim_default,
            
            "lstm_units_min": self.lstm_units_min,
            "lstm_units_max": self.lstm_units_max,
            "lstm_units_step": self.lstm_units_step,
            "lstm_units_default": self.lstm_units_default,
            
            "gru_units_min": self.gru_units_min,
            "gru_units_max": self.gru_units_max,
            "gru_units_step": self.gru_units_step,
            "gru_units_default": self.gru_units_default,
            
            "cnn_units_min": self.cnn_units_min,
            "cnn_units_max": self.cnn_units_max,
            "cnn_units_step": self.cnn_units_step,
            "cnn_units_default": self.cnn_units_default,
            
            "trans_heads_min": self.trans_heads_min,
            "trans_heads_max": self.trans_heads_max,
            "trans_heads_step": self.trans_heads_step,
            
            "trans_ff_min": self.trans_ff_min,
            "trans_ff_max": self.trans_ff_max,
            "trans_ff_step": self.trans_ff_step,
            
            "dense_units_min": self.dense_units_min,
            "dense_units_max": self.dense_units_max,
            "dense_units_step": self.dense_units_step,
            
            # Base path
            "base_path": self.base_path,
        }

# Example of returning parameters to an environment manager:
if __name__ == "__main__":
    # Instantiate the parameters class, possibly with overrides.
    tuner_params = CMqlEnvMLTunerParams(seed=123, tunemode='Hyperband', max_epochs=200)
    
    # Retrieve the complete dictionary of parameters.
    params_dict = tuner_params.get_tuner_params()
    
    # Example: pass the parameters dictionary to your environment manager.
    # envmgr.set_params(params_dict)
    
    # For demonstration, print the parameters.
    for key, value in params_dict.items():
        print(f"{key}: {value}")

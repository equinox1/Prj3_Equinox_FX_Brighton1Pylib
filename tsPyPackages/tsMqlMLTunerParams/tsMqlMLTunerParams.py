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
    """Manage machine learning tuner parameters using a merged parameter dictionary."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the machine learning tuner parameters.
        Keyword arguments allow overriding of default values.
        """

        # --- Group default parameters into dictionaries ---
        self.GENERAL_PARAMS = {
            "today": get_current_datetime(),
            "seed": 42,
            "tuner_id": None,
        }

        self.TUNER_MODE_PARAMS = {
            "tunemode": "hyperband",
            "tunemodeepochs": 100,
            "modelsummary": False,
            "keras_tuner": "hyperband",
            "hyperband_iterations": 1,
        }

        self.DATASET_PARAMS = {
            "train_dataset": None,
            "val_dataset": None,
            "test_dataset": None,
            "mp_ml_show_plot": False,
        }

        self.INPUT_CONFIG_PARAMS = {
            "input_shape": None,
            "data_input_shape": None,
            "multi_inputs": False,
            "batch_size": 32,
        }

        self.MODEL_SELECTION_PARAMS = {
            "cnn_model": True,
            "lstm_model": True,
            "gru_model": True,
            "transformer_model": True,
            "multiactivate": True,
            "multi_branches": True,
            "multi_outputs": False,
        }

        self.LABEL_WINDOW_PARAMS = {
            "label_columns": None,
            "input_width": None,
            "shift": None,
            "total_window_size": None,
            "label_width": None,
        }

        self.EPOCH_PARAMS = {
            "max_epochs": 100,
            "min_epochs": 10,
            "tf_param_epochs": 10,
            "epochs": 2,
            "tune_new_entries": True,
            "allow_new_entries": True,
        }

        self.CORE_TUNER_PARAMS = {
            "num_trials": 3,
            "max_retries_per_trial": 5,
            "max_consecutive_failed_trials": 3,
            "steps_per_execution": 50,
            "executions_per_trial": 1,
            "overwrite": True,
            "max_retries_per_trial": 5,
            "max_consecutive_failed_trials": 3,
            "executions_per_trial": 1,
        }

        self.EXTRA_TUNER_PARAMS = {
            "factor": 10,
            "objective": "val_loss",
        }

        self.LOGGING_TF_PARAMS = {
            "logger": None,
            "tf1": False,
            "tf2": False,
        }

        self.EVALUATOR_PARAMS = {
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metrics": ['mean_squared_error'],
            "dropout": 0.2,
        }

        self.CHECKPOINT_PARAMS = {
            "chk_fullmodel": True,
            "chk_verbosity": 1,
            "chk_mode": "min",
            "chk_monitor": "val_loss",
            "chk_sav_freq": "epoch",
            "chk_patience": 3,
        }

        self.MODEL_TUNER_PARAMS = {
            "unitmin": 32,
            "unitmax": 512,
            "unitstep": 32,
            "defaultunits": 128,
        }

        self.MODEL_SCALE_PARAMS = {
            "all_modelscale": 1.0,
            "cnn_modelscale": 1.0,
            "lstm_modelscale": 1.0,
            "gru_modelscale": 1.0,
            "trans_modelscale": 1.0,
            "transh_modelscale": 1.0,
            "transff_modelscale": 1.0,
            "dense_modelscale": 1.0,
        }

        self.ADDITIONAL_MODEL_PARAMS = {
            # These defaults assume scales of 1.0; if scales differ, they can be overridden via kwargs.
            "trans_dim_min": 32,
            "trans_dim_max": 256,
            "trans_dim_step": 32,
            "trans_dim_default": 64,
            "lstm_units_min": 32,
            "lstm_units_max": 128,
            "lstm_units_step": 32,
            "lstm_units_default": 64,
            "gru_units_min": 32,
            "gru_units_max": 128,
            "gru_units_step": 32,
            "gru_units_default": 64,
            "cnn_units_min": 32,
            "cnn_units_max": 128,
            "cnn_units_step": 32,
            "cnn_units_default": 64,
            "trans_heads_min": 2,
            "trans_heads_max": 8,
            "trans_heads_step": 2,
            "trans_ff_min": 64,
            "trans_ff_max": 512,
            "trans_ff_step": 64,
            "dense_units_min": 32,
            "dense_units_max": 128,
            "dense_units_step": 32,
        }

        self.FILE_PATH_PARAMS = {
            "base_path": os.getcwd(),
        }

        # --- Merge all parameters ---
        merged_params: Dict[str, Any] = {
            **self.GENERAL_PARAMS,
            **self.TUNER_MODE_PARAMS,
            **self.DATASET_PARAMS,
            **self.INPUT_CONFIG_PARAMS,
            **self.MODEL_SELECTION_PARAMS,
            **self.LABEL_WINDOW_PARAMS,
            **self.EPOCH_PARAMS,
            **self.CORE_TUNER_PARAMS,
            **self.EXTRA_TUNER_PARAMS,
            **self.LOGGING_TF_PARAMS,
            **self.EVALUATOR_PARAMS,
            **self.CHECKPOINT_PARAMS,
            **self.MODEL_TUNER_PARAMS,
            **self.MODEL_SCALE_PARAMS,
            **self.ADDITIONAL_MODEL_PARAMS,
            **self.FILE_PATH_PARAMS,
            **kwargs  # Override defaults with any provided keyword arguments
        }

        # Initialize the parent with the merged parameters.
        super().__init__(custom_params=merged_params)

    def get_tuner_params(self) -> Dict[str, Any]:
        """
        Retrieve all tuner parameters.
        
        Returns:
            A dictionary containing the merged tuner parameters.
        """
        return self.custom_params  # Assuming CEnvCore stores the provided dict as custom_params

# Example demonstration:
if __name__ == "__main__":
    # Instantiate the parameters class with some overrides.
    tuner_params = CMqlEnvMLTunerParams(seed=123, tunemode='hyperband', max_epochs=200)
    
    # Retrieve and print the complete dictionary of parameters.
    params_dict = tuner_params.get_tuner_params()
    for key, value in params_dict.items():
        print(f"{key}: {value}")

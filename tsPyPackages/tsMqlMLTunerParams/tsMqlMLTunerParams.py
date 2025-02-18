"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlMLTunerParams.py
File: tsPyPackages/tsMqlMLTunerParams/tsMqlMLTunerParams.py
Description: Load and set machine learning tuner parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""


import logging
from datetime import date
from tsMqlPlatform import run_platform, platform_checker

# Initialize logger
logger = logging.getLogger("tsMqlDataParams")
logging.basicConfig(level=logging.INFO)

# Run platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")


def get_global_param(all_params, param_name, default=None):
    """Safely fetch global environment parameters."""
    if hasattr(all_params, 'get_params') and callable(all_params.get_params):
        return all_params.get_params().get(param_name, default)
    logger.warning(f"Global parameter '{param_name}' is not available.")
    return default


class CMqlEnvMLTunerParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # get global environment parameters
        all_params = global_setter.run_service()
        globalenv = all_params['genparams']['globalenv']  # Access the global env params
        dataenv = all_params['dataparams']['dataenv']  # Access the data params
        mlenv = all_params['mlearnparams']['mlenv']  # Access the ml params
        tuneenv = all_params['tunerparams']['tuneenv']  # Access the tuner params

        # General Settings
        self.tunemode = kwargs.get('mp_ml_tunemode', True)
        self.tunemodeepochs = kwargs.get('tunemodeepochs', 100)
        self.modelsummary = kwargs.get('modelsummary', False)
        self.today = kwargs.get('today', date.today().strftime('%Y-%m-%d %H:%M:%S'))
        self.seed = kwargs.get('seed', 42)
        self.tuner_id = kwargs.get('tuner_id', 1)

        # Dataset Splits
        self.train_split = kwargs.get('train_split', 0.7)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.test_split = kwargs.get('test_split', 0.1)

        # Paths
        self.mp_ml_def_base_path = get_global_param(globalenv, 'mp_ml_def_base_path')
        self.mp_ml_num_base_path = get_global_param(globalenv, 'mp_ml_num_base_path')
        self.mp_ml_checkpoint_filepath = kwargs.get('mp_ml_checkpoint_filepath', self.mp_ml_def_base_path)
        self.modeldatapath = get_global_param(globalenv, 'mp_data_path')
        
        logger.info(f"Default base path: {self.mp_ml_def_base_path}")
        logger.info(f"Numbered base path: {self.mp_ml_num_base_path}")

        # Model selection
        self.model_types = {
            'cnn': kwargs.get('cnn_model', True),
            'lstm': kwargs.get('lstm_model', True),
            'gru': kwargs.get('gru_model', True),
            'transformer': kwargs.get('transformer_model', True)
        }
        self.multi_branches = kwargs.get('multi_branches', True)
        self.multi_inputs = kwargs.get('multi_inputs', False)
        self.multi_outputs = kwargs.get('multi_outputs', False)
        
        # Model parameters
        self.input_shape = kwargs.get('input_shape', None)
        self.batch_size = kwargs.get('batch_size', 8)
        self.label_columns = kwargs.get('label_columns', None)

        # Training window settings
        self.input_width = kwargs.get('input_width', 24)
        self.shift = kwargs.get('shift', 24)
        self.total_window_size = self.input_width + self.shift
        self.label_width = kwargs.get('label_width', 1)
        
        # Tuning parameters
        self.keras_tuner = kwargs.get('keras_tuner', 'Hyperband')
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.min_epochs = kwargs.get('min_epochs', 10)
        self.epochs = kwargs.get('epochs', 2)
        self.num_trials = kwargs.get('num_trials', 3)
        self.overwrite = kwargs.get('overwrite', True)
        
        # Optimization settings
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.metrics = kwargs.get('metrics', ['mean_squared_error'])
        self.dropout = kwargs.get('dropout', 0.2)
        
        # Checkpointing
        self.chk_settings = {
            'full_model': kwargs.get('chk_fullmodel', True),
            'verbosity': kwargs.get('chk_verbosity', 1),
            'mode': kwargs.get('chk_mode', 'min'),
            'monitor': kwargs.get('chk_monitor', 'val_loss'),
            'save_freq': kwargs.get('chk_sav_freq', 'epoch'),
            'patience': kwargs.get('chk_patience', 3)
        }
        
        # Model scaling factors
        self.model_scales = {
            'cnn': kwargs.get('cnn_modelscale', 1.0),
            'lstm': kwargs.get('lstm_modelscale', 1.0),
            'gru': kwargs.get('gru_modelscale', 1.0),
            'transformer': kwargs.get('trans_modelscale', 1.0),
            'dense': kwargs.get('dense_modelscale', 1.0)
        }
    
    def get_params(self):
        """Returns a dictionary of all set parameters."""
        return {**self.__dict__, **self.model_types, **self.model_scales, **self.chk_settings}

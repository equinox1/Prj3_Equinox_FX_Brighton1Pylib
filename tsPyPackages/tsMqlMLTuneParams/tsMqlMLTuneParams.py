#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"

from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk=run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras_tuner as kt
import os
import pathlib
import numpy as np
from datetime import date
import random

class CMdtunerHyperModel:
   def __init__(self, **kwargs):
        self.input_width = kwargs.get('input_width', 24)
        self.shift = kwargs.get('shift', 24)
        self.label_width = kwargs.get('label_width', 1)
        self.train_df = kwargs.get('train_df', None)
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df', None)
        self.label_columns = kwargs.get('label_columns', None)
        self.batch_size = kwargs.get('batch_size', 8)
        self.column_indices = {}
        self.label_columns_indices = {}
        self.total_window_size = self.input_width + self.shift
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.min_epochs = kwargs.get('min_epochs', 10)
        self.factor = kwargs.get('factor', 10)
        self.seed = kwargs.get('seed', 42)
        np.random.seed(self.seed)  # Ensuring reproducibility
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)
        self.tune_new_entries = kwargs.get('tune_new_entries', False)
        self.allow_new_entries = kwargs.get('allow_new_entries', False)
        self.max_retries_per_trial = kwargs.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = kwargs.get('max_consecutive_failed_trials', 3)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.epochs = kwargs.get('epochs', 2)
        self.dropout = kwargs.get('dropout', 0.2)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.metrics = kwargs.get('metrics', ['mean_squared_error'])
        self.directory = kwargs.get('directory', None)
        self.logger = kwargs.get('logger', None)
        self.tuner_id = kwargs.get('tuner_id', None)
        self.overwrite = kwargs.get('overwrite', True)
        self.executions_per_trial = kwargs.get('executions_per_trial', 1)
        self.chk_fullmodel = kwargs.get('chk_fullmodel', True)
        self.chk_verbosity = kwargs.get('chk_verbosity', 1)
        self.chk_mode = kwargs.get('chk_mode', 'min')
        self.chk_monitor = kwargs.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = kwargs.get('chk_sav_freq', 'epoch')
        self.chk_patience = kwargs.get('chk_patience', 3)
        self.modeldatapath = kwargs.get('modeldatapath', None)
        self.project_name = kwargs.get('project_name', "prjEquinox1_prod.keras")
        self.today = kwargs.get('today', None)
        self.random = kwargs.get('random', None)
        self.baseuniq = kwargs.get('baseuniq', None)
        self.basepath = kwargs.get('basepath', None)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', None)
        self.unitmin = kwargs.get('unitmin', 32)
        self.unitmax = kwargs.get('unitmax', 512)
        self.unitstep = kwargs.get('unitstep', 32)
        self.defaultunits = kwargs.get('defaultunits', 128)
        self.num_trials = kwargs.get('num_trials', 3)
        self.steps_per_execution = kwargs.get('steps_per_execution', 50)
        self.keras_tuner = kwargs.get('keras_tuner', 'Hyperband') # 'Hyperband' or 'RandomSearch'
        self.all_modelscale = kwargs.get('all_modelscale', 1.0)
        self.cnn_modelscale = kwargs.get('cnn_modelscale', 1.0)
        self.lstm_modelscale = kwargs.get('lstm_modelscale', 1.0)
        self.gru_modelscale = kwargs.get('gru_modelscale', 1.0)
       

   def get_hypermodel_params(self, basepath=None, **kwargs):
        today_date = date.today().strftime('%Y-%m-%d %H:%M:%S')
        random_seed = self.seed  # Using self.seed for consistency
        base_path = pathlib.Path(basepath) if basepath else pathlib.Path(os.getcwd())
        project_name = self.project_name or "prjEquinox1_prod.keras"

        subdir = base_path / 'tshybrid_ensemble_tuning_prod' / '1'
        subdir.mkdir(parents=True, exist_ok=True)

        return {
            'directory': str(subdir),
            'basepath': str(subdir),
            'checkpoint_filepath': str(base_path / 'tshybrid_ensemble_tuning_prod' / self.project_name),
            'objective': self.objective,
            'max_epochs': self.max_epochs,
            'min_epochs': self.min_epochs,
            'factor': self.factor,
            'seed': self.seed,
            'hyperband_iterations': self.hyperband_iterations,
            'tune_new_entries': self.tune_new_entries,
            'allow_new_entries': self.allow_new_entries,
            'max_retries_per_trial': self.max_retries_per_trial,
            'max_consecutive_failed_trials': self.max_consecutive_failed_trials,
            'validation_split': self.validation_split,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout': self.dropout,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'metrics': self.metrics,
            'logger': self.logger,
            'tuner_id': self.tuner_id,
            'overwrite': self.overwrite,
            'executions_per_trial': self.executions_per_trial,
            'chk_fullmodel': self.chk_fullmodel,
            'chk_verbosity': self.chk_verbosity,
            'chk_mode': self.chk_mode,
            'chk_monitor': self.chk_monitor,
            'chk_sav_freq': self.chk_sav_freq,
            'chk_patience': self.chk_patience,
            'modeldatapath': self.modeldatapath,
            'project_name': self.project_name,
            'steps_per_execution': self.steps_per_execution,
            'unitmin': self.unitmin,
            'unitmax': self.unitmax,
            'unitstep': self.unitstep,
            'defaultunits': self.defaultunits,
            'num_trials': self.num_trials
        }

   # Initialize the tuner class
   def initialize_tuner(hypermodel_params, train_dataset, val_dataset, test_dataset):
      try:
         print("Creating an instance of the tuner class")
         mt = CMdtuner(
               # tf datasets
               traindataset=train_dataset,
               valdataset=val_dataset,
               testdataset=test_dataset,
               # Model selection
               cnn_model=mp_ml_cnn_model,
               lstm_model=mp_ml_lstm_model,
               gru_model=mp_ml_gru_model,
               transformer_model=mp_ml_transformer_model,
               multiactivate=True,
               # Model inputs directly from the data traindataset shape
               data_input_shape=input_shape,
               # Model inputs from the shape selection options 1-4
               main_custom_shape_selector=mp_ml_custom_input_shape,
               cnn_custom_shape_selector=mp_ml_custom_input_cnn_shape,
               lstm_custom_shape_selector=mp_ml_custom_input_lstm_shape,
               gru_custom_shape_selector=mp_ml_custom_input_gru_shape,
               transformer_custom_shape_selector=mp_ml_custom_input_transformer_shape,
               # Use merge of different shapes in final input output
               multi_inputs=mp_ml_multi_inputs,
               multi_outputs=mp_ml_multi_outputs,
               multi_branches=mp_ml_multi_branches,
               #Logging
               tf1=False,
               tf2T=False,
               # Model parameters and hypermodel params
               step=mp_ml_tf_param_steps,
               objective=hypermodel_params['objective'],
               max_epochs=hypermodel_params['max_epochs'],
               min_epochs=mp_ml_tf_param_min_epochs,
               factor=hypermodel_params['factor'],
               seed=hypermodel_params['seed'],
               hyperband_iterations=hypermodel_params['hyperband_iterations'],
               tune_new_entries=hypermodel_params['tune_new_entries'],
               allow_new_entries=hypermodel_params['allow_new_entries'],
               max_retries_per_trial=hypermodel_params['max_retries_per_trial'],
               max_consecutive_failed_trials=hypermodel_params['max_consecutive_failed_trials'],
               validation_split=hypermodel_params['validation_split'],
               epochs=hypermodel_params['epochs'],
               batch_size=hypermodel_params['batch_size'],
               dropout=hypermodel_params['dropout'],
               optimizer=hypermodel_params['optimizer'],
               loss=hypermodel_params['loss'],
               metrics=hypermodel_params['metrics'],
               directory=hypermodel_params['directory'],
               basepath=hypermodel_params['basepath'],
               project_name=hypermodel_params['project_name'],
               logger=hypermodel_params['logger'],
               tuner_id=hypermodel_params['tuner_id'],
               overwrite=hypermodel_params['overwrite'],
               executions_per_trial=hypermodel_params['executions_per_trial'],
               chk_fullmodel=hypermodel_params['chk_fullmodel'],
               chk_verbosity=hypermodel_params['chk_verbosity'],
               chk_mode=hypermodel_params['chk_mode'],
               chk_monitor=hypermodel_params['chk_monitor'],
               chk_sav_freq=hypermodel_params['chk_sav_freq'],
               chk_patience=hypermodel_params['chk_patience'],
               checkpoint_filepath=hypermodel_params['checkpoint_filepath'],
               modeldatapath=hypermodel_params['modeldatapath'],
               tunemode =  mp_ml_tunemode,
               tunemodeepochs = mp_ml_tunemodeepochs,
               modelsummary = mp_ml_modelsummary,
               unitmin=hypermodel_params['unitmin'],
               unitmax=hypermodel_params['unitmax'],
               unitstep=hypermodel_params['unitstep'],
               defaultunits=hypermodel_params['defaultunits'],
               num_trials=hypermodel_params['num_trials'],
               steps_per_execution=mp_ml_steps_per_execution,
               keras_tuner=hypermodel_params['keras_tuner'],
               all_modelscale=hypermodel_params['all_modelscale'],
               cnn_modelscale=hypermodel_params['cnn_modelscale'],
               lstm_modelscale=hypermodel_params['lstm_modelscale'],
               gru_modelscale=hypermodel_params['gru_modelscale'],
               trans_modelscale = hypermodel_params['trans_modelscale'],
               transh_modelscale=hypermodel_params['transh_modelscale'],
               transff_modelscale=hypermodel_params['transff_modelscale'],
               dense_modelscale=hypermodel_params['dense_modelscale']  
         )
         print("Tuner initialized successfully.")
         return mt
      except Exception as e:
         print(f"Error initializing the tuner: {e}")
         raise
   
   def get_params(self):
            return self.__dict__  # Returns all attributes as a dictionary

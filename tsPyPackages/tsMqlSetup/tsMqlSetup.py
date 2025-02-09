#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"

from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk=run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

import tensorflow as tf
from tensorflow.keras.layers import Input
import os
import warnings
import posixpath  # For path handling

class CMqlSetup:
    def __init__(self, tflog='2', warn='ignore', **kwargs):
        self.kwargs = kwargs
        self.tflog = kwargs.get('tflog', tflog)
        self.warn = kwargs.get('warn', warn)
        self.tfdebug = kwargs.get('tfdebug', False)
        self.precision = kwargs.get('precision', 'mixed_float16')

        # Initialize the MT5 api
        self.os = kwargs.get('os', 'windows')  # windows or linux or macos
        if self.os == 'windows':
            import MetaTrader5 as mt5
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', mt5.COPY_TICKS_ALL)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        else:
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', 0)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        
        warnings.filterwarnings(self.warn)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog
        
        # Set the global policy for mixed precision
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy(self.precision)
        
        print("TensorFlow Version:", tf.__version__)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

    def get_computation_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("✅ Running on TPU")
            return tf.distribute.TPUStrategy(tpu)
        except Exception as e:
            print("⚠️ TPU not found, using GPU/CPU:", e)
            return tf.distribute.get_strategy()

    def set_setup_tfdebug(self):
        if self.tfdebug:
            tf.debugging.set_log_device_placement(True)
            tf.config.experimental_run_functions_eagerly(True)
            tf.config.run_functions_eagerly(True)
            tf.config.optimizer.set_jit(True)
            # List available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            print("GPUs Available:", gpus)

            # Enable memory growth to avoid sudden crashes
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.debugging.set_log_device_placement(True)

            if tf.config.list_physical_devices('GPU'):
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                print("Current GPU Memory Usage:", memory_info)
            import psutil
            print("RAM Usage:", psutil.virtual_memory().used / 1e9, "GB")

            import gc
            tf.keras.backend.clear_session()
            gc.collect()


class CMqlEnvData:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
        # Load configuration parameters with default values
        self.mp_data_loadapiticks = kwargs.get('mp_data_loadapiticks', True)
        self.mp_data_loadapirates = kwargs.get('mp_data_loadapirates', True)
        self.mp_data_loadfileticks = kwargs.get('mp_data_loadfileticks', True)
        self.mp_data_loadfilerates = kwargs.get('mp_data_loadfilerates', True)
        self.mp_data_data_label = kwargs.get('mp_data_data_label', 3)
        self.mv_data_dfname1 = kwargs.get('mv_data_dfname1', "df_rates1")
        self.mv_data_dfname2 = kwargs.get('mv_data_dfname2', "df_rates2")
        self.mp_data_rows = kwargs.get('mp_data_rows', 1000)
        self.mp_data_rowcount = kwargs.get('mp_data_rowcount', 10000)
        self.mp_data_history_size = kwargs.get('mp_data_history_size', 5)
        self.mp_data_cfg_usedata = kwargs.get('mp_data_cfg_usedata', 'loadfilerates')

         # Initialize the MT5 api
        self.os = kwargs.get('os', 'windows')  # windows or linux or macos
        if self.os == 'windows':
            import MetaTrader5 as mt5
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', mt5.COPY_TICKS_ALL)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        else:
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', 0)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        
        self.mp_data_timeframe = kwargs.get('mp_data_timeframe', None)  # Ensure this is set externally
        self.mp_data_tab_rows = kwargs.get('mp_data_tab_rows', 10)
        self.mp_data_tab_width = kwargs.get('mp_data_tab_width', 30)
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', False)
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', False)
        self.mp_data_show_head = kwargs.get('mp_data_show_head', False)

        self.mp_hl_avg_col = kwargs.get('mp_hl_avg_col', 'HLAvg')
        self.mp_ma_col = kwargs.get('mp_ma_col', 'SMA')
        self.mp_returns_col = kwargs.get('mp_returns_col', 'LogReturns')
        self.mp_returns_col_scaled = kwargs.get('mp_returns_col_scaled', 'LogReturns_scaled')
        self.mp_create_label = kwargs.get('mp_create_label', False)
        self.mp_create_label_scaled = kwargs.get('mp_create_label_scaled', False)
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', False)
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', False)

    def get_params(self):
        return self.__dict__  # Returns all attributes as a dictionary


class CMqlEnvML:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
         # Initialize the MT5 api
        self.os = kwargs.get('os', 'windows')  # windows or linux or macos
        if self.os == 'windows':
            import MetaTrader5 as mt5
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', mt5.COPY_TICKS_ALL)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        else:
            self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', 0)
            self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        
        # File Paths
        self.mp_ml_model_datapath = kwargs.get('mp_ml_model_datapath', os.path.expanduser("~/tsModelData/"))
        self.mp_ml_directory = kwargs.get('mp_ml_directory', "tshybrid_ensemble_tuning_prod")
        self.mp_ml_project_name = kwargs.get('mp_ml_project_name', "prjEquinox1_prod.keras")
        self.mp_ml_baseuniq = kwargs.get('mp_ml_baseuniq', "1")
        self.mp_ml_base_path = os.path.join(self.mp_ml_model_datapath, self.mp_ml_directory, self.mp_ml_baseuniq)
        self.mp_ml_mbase_path = os.path.join(self.mp_ml_model_datapath, self.mp_ml_directory)
        self.mp_ml_checkpoint_filepath = posixpath.join(self.mp_ml_base_path, self.mp_ml_directory, self.mp_ml_project_name)
        
        # Model settings
        self.mp_ml_shuffle = kwargs.get('mp_ml_shuffle', False)
        self.mp_ml_cnn_model = kwargs.get('mp_ml_cnn_model', True)
        self.mp_ml_lstm_model = kwargs.get('mp_ml_lstm_model', True)
        self.mp_ml_gru_model = kwargs.get('mp_ml_gru_model', True)
        self.mp_ml_transformer_model = kwargs.get('mp_ml_transformer_model', True)
        self.mp_ml_multi_inputs = kwargs.get('mp_ml_multi_inputs', False)
        self.mp_ml_multi_outputs = kwargs.get('mp_ml_multi_outputs', False)
        self.mp_ml_multi_branches = kwargs.get('mp_ml_multi_branches', True)
        self.mp_ml_tunemode = kwargs.get('mp_ml_tunemode', True)
        self.mp_ml_tunemodeepochs = kwargs.get('mp_ml_tunemodeepochs', True)
        self.mp_ml_batch_size = kwargs.get('mp_ml_batch_size', 8)
        self.mp_ml_train_split = kwargs.get('mp_ml_train_split', 0.7)
        self.mp_ml_validation_split = kwargs.get('mp_ml_validation_split', 0.2)
        self.mp_ml_test_split = kwargs.get('mp_ml_test_split', 0.1)
        
        # ML Parameters
        self.mp_ml_num_trials = kwargs.get('mp_ml_num_trials', 3)
        self.mp_ml_steps_per_execution = kwargs.get('mp_ml_steps_per_execution', 50)
        self.mp_ml_tf_param_steps = kwargs.get('mp_ml_tf_param_steps', 10)
        self.mp_ml_tf_param_max_epochs = kwargs.get('mp_ml_tf_param_max_epochs', 100)
        self.mp_ml_tf_param_min_epochs = kwargs.get('mp_ml_tf_param_min_epochs', 1)
        self.mp_ml_tf_param_epochs = kwargs.get('mp_ml_tf_param_epochs', self.mp_ml_tf_param_max_epochs)
        self.mp_ml_tf_param_chk_patience = kwargs.get('mp_ml_tf_param_chk_patience', 3)
        
        # Data Processing
        self.mp_ml_custom_input_keyfeat = kwargs.get('mp_ml_custom_input_keyfeat', {'Close'})
        self.mp_ml_custom_output_label = kwargs.get('mp_ml_custom_output_label', {'Label'})

        self.mp_ml_custom_input_keyfeat_scaled = {feat + '_Scaled' for feat in self.mp_ml_custom_input_keyfeat}  # the feature to predict
        self.mp_ml_custom_output_label_scaled = {targ + '_Scaled' for targ in self.mp_ml_custom_output_label}  # the label shifted to predict
        self.mp_ml_custom_output_label_count=len(self.mp_ml_custom_output_label)
        
        self.mp_ml_cfg_period = kwargs.get('mp_ml_cfg_period', 24) # Lookahead periods
        self.mp_ml_tf_ma_windowin = kwargs.get('mp_ml_tf_ma_windowin', 24) # Moving average window
        self.mp_ml_tf_shiftin = kwargs.get('mp_ml_tf_shiftin', 1) # Shift in the data
        self.mp_ml_tf_shiftout = kwargs.get('mp_ml_tf_shiftout', 1) # Shift out the data
        self.mp_ml_tf_ma_windowout = kwargs.get('mp_ml_tf_ma_windowout', 5) # Moving average window for output
   
    def get_params(self):
        return self.__dict__  # Returns all attributes as a dictionary


class CMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        

    def get_params(self):
            return self.__dict__  # Returns all attributes as a dictionary
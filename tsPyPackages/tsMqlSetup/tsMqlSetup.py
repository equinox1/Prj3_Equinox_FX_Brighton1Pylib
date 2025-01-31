import tensorflow as tf
from tensorflow.keras.layers import Input
import os
import warnings
import MetaTrader5 as mt5  # Ensure this is installed and imported
import posixpath  # For path handling

class tsMqlSetup:
    def __init__(self, tflog='2', warn='ignore', **kwargs):
        self.kwargs = kwargs
        self.tflog = tflog
        self.warn = warn
        
        warnings.filterwarnings(self.warn)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog
        
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


class tsMqlEnvData:
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
        self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', mt5.COPY_TICKS_ALL)
        self.mp_data_command_rates = kwargs.get('mp_data_command_rates', None)
        self.mp_data_timeframe = kwargs.get('mp_data_timeframe', None)  # Ensure this is set externally
        self.mp_data_tab_rows = kwargs.get('mp_data_tab_rows', 10)
        self.mp_data_tab_width = kwargs.get('mp_data_tab_width', 30)
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', False)
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', False)
        self.mp_data_show_head = kwargs.get('mp_data_show_head', False)
    
    def get_data_params(self):
        return self.__dict__  # Returns all attributes as a dictionary


class tsMqlEnvML:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
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
        
    def get_ml_params(self):
        return self.__dict__  # Returns all attributes as a dictionary


class tsMqlEnvGlobal:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

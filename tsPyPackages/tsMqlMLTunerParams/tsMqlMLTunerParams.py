import logging
from tsMqlPlatform import run_platform, platform_checker, logger as ts_logger, config

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

class CMqlEnvTuneML:
    def __init__(self,globalenv, **kwargs):
        self.kwargs = kwargs
        self.globalenv = globalenv

        # Tunemode
        self.tunemode = kwargs.get('mp_ml_tunemode', True)
        self.tunemodeepochs = kwargs.get('tunemodeepochs', 100)
        self.modelsummary = kwargs.get('modelsummary', False)

        # Id parameters
        self.today = kwargs.get('today', date.today().strftime('%Y-%m-%d %H:%M:%S'))
        self.seed = kwargs.get('seed', 42)
        self.tuner_id = kwargs.get('tuner_id', 1)

        # Datasets
        self.train_dataset = kwargs.get('train_dataset', None)
        self.val_dataset = kwargs.get('val_dataset', None)
        self.test_dataset = kwargs.get('test_dataset', None)

        # Data split
        self.mp_ml_train_split = kwargs.get('mp_ml_train_split', 0.7)
        self.mp_ml_validation_split = kwargs.get('mp_ml_validation_split', 0.2)
        self.mp_ml_test_split = kwargs.get('mp_ml_test_split', 0.1)

        self.train_split = kwargs.get('train_split', self.mp_ml_train_split)
        self.validation_split = kwargs.get('validation_split', self.mp_ml_validation_split)
        self.test_split = kwargs.get('test_split', self.mp_ml_test_split)

        # Tuner parameters file path names
       
        self.mp_ml_src_base = self.globalenv.get_params().get('mp_ml_src_base', None)
        self.mp_ml_src_lib = self.globalenv.get_params().get('mp_ml_src_lib', None)
        self.mp_ml_src_data = self.globalenv.get_params().get('mp_ml_src_data', None)
        self.mp_ml_directory = self.globalenv.get_params().get('mp_ml_directory', None)
        self.mp_ml_project_name = self.globalenv.get_params().get('mp_ml_project_name', None)
        self.mp_ml_baseuniq = self.globalenv.get_params().get('mp_ml_baseuniq', None)
        self.mp_ml_def_base_path = self.globalenv.get_params().get('mp_ml_def_base_path', None)
        self.mp_ml_num_base_path = self.globalenv.get_params().get('mp_ml_num_base_path', None)

        logger.info(f"Default base path Attr: {self.mp_ml_def_base_path}")
        logger.info(f"Numbered base path Attr: {self.mp_ml_num_base_path}")
        
        # checkpoint File paths
        self.mp_ml_checkpoint_filepath = kwargs.get('mp_ml_checkpoint_filepath', self.mp_ml_def_base_path)
        
        # Modeldata path
        self.modeldatapath = self.globalenv.get_params().get('mp_data_path', None)
        
        # Model selection
        self.cnn_model = kwargs.get('cnn_model', True)
        self.lstm_model = kwargs.get('lstm_model', True)
        self.gru_model = kwargs.get('gru_model', True)
        self.transformer_model = kwargs.get('transformer_model', True)
        self.multiactivate = kwargs.get('multiactivate', True)
        self.multi_branches = kwargs.get('multi_branches', True)

        # Input shape
        self.input_shape = kwargs.get('input_shape', None)
        self.data_input_shape = kwargs.get('data_input_shape', None)
        self.multi_inputs = kwargs.get('multi_inputs', False)
        self.batch_size = kwargs.get('batch_size', 32)

        # Output shape
        self.multi_outputs = kwargs.get('multi_outputs', False)

        # Label columns
        self.label_columns = kwargs.get('label_columns', None)

        # Estimation window
        self.input_width = kwargs.get('input_width', 24)
        self.shift = kwargs.get('shift', 24)
        self.total_window_size = self.input_width + self.shift
        self.label_width = kwargs.get('label_width', 1)
        self.batch_size = kwargs.get('batch_size', 8)

        # Tuner app
        self.keras_tuner = kwargs.get('keras_tuner', 'Hyperband')  # 'Hyperband' or 'RandomSearch'
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)

        # Tuner epochs
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.min_epochs = kwargs.get('min_epochs', 10)
        self.tf_param_epochs = kwargs.get('tf_param_epochs', 10)
        self.epochs = kwargs.get('epochs', 2)
        self.tune_new_entries = kwargs.get('tune_new_entries', True)
        self.allow_new_entries = kwargs.get('allow_new_entries', True)

        # Tuner parameters core
        self.num_trials = kwargs.get('num_trials', 3)
        self.max_retries_per_trial = kwargs.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = kwargs.get('max_consecutive_failed_trials', 3)
        self.steps_per_execution = kwargs.get('steps_per_execution', 50)
        self.executions_per_trial = kwargs.get('executions_per_trial', 1)
        self.overwrite = kwargs.get('overwrite', True)

        # Tuner parameters extra
        self.factor = kwargs.get('factor', 10)
        self.objective = kwargs.get('objective', 'val_loss')

        # Logging
        self.logger = kwargs.get('logger', None)
        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)

        # Evaluators
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.metrics = kwargs.get('metrics', ['mean_squared_error'])
        self.dropout = kwargs.get('dropout', 0.2)

        # Checkpoint parameters
        
        self.chk_fullmodel = kwargs.get('chk_fullmodel', True)
        self.chk_verbosity = kwargs.get('chk_verbosity', 1)
        self.chk_mode = kwargs.get('chk_mode', 'min')
        self.chk_monitor = kwargs.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = kwargs.get('chk_sav_freq', 'epoch')
        self.chk_patience = kwargs.get('chk_patience', 3)

        # Model tuners
        self.unitmin = kwargs.get('unitmin', 32)
        self.unitmax = kwargs.get('unitmax', 512)
        self.unitstep = kwargs.get('unitstep', 32)
        self.defaultunits = kwargs.get('defaultunits', 128)

        # Model scales
        self.all_modelscale = kwargs.get('all_modelscale', 1.0)
        self.cnn_modelscale = kwargs.get('cnn_modelscale', 1.0)
        self.lstm_modelscale = kwargs.get('lstm_modelscale', 1.0)
        self.gru_modelscale = kwargs.get('gru_modelscale', 1.0)
        self.trans_modelscale = kwargs.get('trans_modelscale', 1.0)
        self.transh_modelscale = kwargs.get('transh_modelscale', 1.0)
        self.transff_modelscale = kwargs.get('transff_modelscale', 1.0)
        self.dense_modelscale = kwargs.get('dense_modelscale', 1.0)

    
    def get_params(self):
        """Returns a dictionary of all set parameters."""
        return self.params
